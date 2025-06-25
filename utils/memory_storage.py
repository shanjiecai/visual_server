#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视觉记忆存储器
用于存储和管理视觉记忆数据，支持按类别存储图像和检索
"""

import time
import json
import base64
import shutil
import os
from typing import Dict, List, Optional, Any, Set
from loguru import logger
from pathlib import Path
import threading
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MemoryFrame:
    """记忆帧数据结构"""
    frame_id: str
    timestamp: float
    base64_data: str
    categories: List[str]
    metadata: Dict[str, Any]
    created_at: float


class VisualMemoryStorage:
    """视觉记忆存储器"""
    
    def __init__(self, storage_config: Dict[str, Any]):
        self.config = storage_config
        self.storage_dir = Path(storage_config.get("storage_dir", "memory_storage"))
        self.max_frames_per_category = storage_config.get("max_frames_per_category", 100)
        self.max_total_frames = storage_config.get("max_total_frames", 1000)
        self.cleanup_interval = storage_config.get("cleanup_interval", 300)  # 5分钟
        self.frame_ttl = storage_config.get("frame_ttl", 3600)  # 1小时
        
        # 内存数据结构
        self._category_to_frames: Dict[str, List[str]] = defaultdict(list)
        self._frame_to_data: Dict[str, MemoryFrame] = {}
        self._frame_completion_time: Dict[str, float] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 初始化存储目录
        self._init_storage()
        
        logger.info(f"Visual memory storage initialized at {self.storage_dir}")
    
    def _init_storage(self) -> None:
        """初始化存储目录结构"""
        try:
            # 创建主存储目录
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建子目录
            frames_dir = self.storage_dir / "frames"
            categories_dir = self.storage_dir / "categories"
            
            frames_dir.mkdir(exist_ok=True)
            categories_dir.mkdir(exist_ok=True)
            
            # 从持久化存储加载数据
            self._load_from_storage()
            
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise
    
    def _load_from_storage(self) -> None:
        """从持久化存储加载数据"""
        try:
            categories_dir = self.storage_dir / "categories"
            frames_dir = self.storage_dir / "frames"
            
            # 加载类别映射
            for category_file in categories_dir.glob("*_frames.json"):
                category_name = category_file.stem.replace("_frames", "")
                
                try:
                    with open(category_file, 'r', encoding='utf-8') as f:
                        frame_ids = [line.strip() for line in f if line.strip()]
                        self._category_to_frames[category_name] = frame_ids
                except Exception as e:
                    logger.warning(f"Failed to load category {category_name}: {e}")
            
            # 加载frame数据（从JPG图片和元数据文件）
            frame_count = 0
            for metadata_file in frames_dir.glob("*_meta.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                    
                    frame_id = frame_data["frame_id"]
                    
                    # 尝试加载对应的JPG图片
                    image_file = frames_dir / f"{frame_id}.jpg"
                    base64_data = ""
                    
                    if image_file.exists():
                        try:
                            with open(image_file, 'rb') as f:
                                image_bytes = f.read()
                            base64_data = base64.b64encode(image_bytes).decode('utf-8')
                            logger.debug(f"Loaded image for frame {frame_id}")
                        except Exception as e:
                            logger.warning(f"Failed to load image for {frame_id}: {e}")
                    
                    # 创建记忆帧对象
                    memory_frame = MemoryFrame(
                        frame_id=frame_id,
                        timestamp=frame_data["timestamp"],
                        base64_data=base64_data,
                        categories=frame_data["categories"],
                        metadata=frame_data["metadata"],
                        created_at=frame_data["created_at"]
                    )
                    
                    # 只有在对应的类别映射中存在时才加载
                    should_load = False
                    for category in frame_data["categories"]:
                        if frame_id in self._category_to_frames.get(category, []):
                            should_load = True
                            break
                    
                    if should_load:
                        self._frame_to_data[frame_id] = memory_frame
                        self._frame_completion_time[frame_id] = frame_data["created_at"]
                        frame_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load frame from {metadata_file}: {e}")
            
            logger.info(f"Loaded {len(self._category_to_frames)} categories and {frame_count} frames from storage")
            
        except Exception as e:
            logger.warning(f"Failed to load from storage: {e}")
    
    def store_frame(self, frame_id: str, timestamp: float, base64_data: str, 
                   categories: List[str], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """存储一帧视觉记忆"""
        try:
            with self._lock:
                # 创建记忆帧对象
                memory_frame = MemoryFrame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    base64_data=base64_data,
                    categories=categories,
                    metadata=metadata or {},
                    created_at=time.time()
                )
                
                # 存储到内存
                self._frame_to_data[frame_id] = memory_frame
                self._frame_completion_time[frame_id] = time.time()
                
                # 按类别索引
                for category in categories:
                    if frame_id not in self._category_to_frames[category]:
                        self._category_to_frames[category].append(frame_id)
                
                # 持久化frame数据到磁盘
                self._persist_frame_data(memory_frame)
                
                # 执行清理策略
                self._cleanup_frames()
                
                # 持久化类别映射
                self._persist_category_mappings()
                
                logger.debug(f"Stored frame {frame_id} with categories {categories}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store frame {frame_id}: {e}")
            return False
    
    def get_frames_by_category(self, category: str, max_count: Optional[int] = None) -> List[MemoryFrame]:
        """根据类别获取记忆帧"""
        try:
            with self._lock:
                frame_ids = self._category_to_frames.get(category, [])
                
                if max_count:
                    frame_ids = frame_ids[-max_count:]  # 获取最新的frames
                
                frames = []
                for frame_id in frame_ids:
                    if frame_id in self._frame_to_data:
                        frames.append(self._frame_to_data[frame_id])
                
                return frames
                
        except Exception as e:
            logger.error(f"Failed to get frames for category {category}: {e}")
            return []
    
    def get_latest_frame_by_category(self, category: str) -> Optional[MemoryFrame]:
        """获取指定类别的最新帧"""
        frames = self.get_frames_by_category(category, max_count=1)
        return frames[0] if frames else None
    
    def get_frames_by_categories(self, categories: List[str], max_count_per_category: int = 2) -> Dict[str, List[MemoryFrame]]:
        """根据多个类别获取记忆帧"""
        result = {}
        for category in categories:
            result[category] = self.get_frames_by_category(category, max_count_per_category)
        return result
    
    def get_all_categories(self) -> List[str]:
        """获取所有可用的类别"""
        with self._lock:
            return list(self._category_to_frames.keys())
    
    def get_category_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取每个类别的统计信息"""
        stats = {}
        with self._lock:
            for category, frame_ids in self._category_to_frames.items():
                # 计算最新帧的时间
                latest_time = 0
                frame_count = len(frame_ids)
                
                if frame_ids:
                    for frame_id in frame_ids[-5:]:  # 检查最近5帧
                        if frame_id in self._frame_to_data:
                            latest_time = max(latest_time, self._frame_to_data[frame_id].timestamp)
                
                stats[category] = {
                    "frame_count": frame_count,
                    "latest_timestamp": latest_time,
                    "latest_frame_id": frame_ids[-1] if frame_ids else None
                }
        
        return stats
    
    def get_latest_frame_info(self) -> Optional[Dict[str, Any]]:
        """获取最新处理完成的帧信息"""
        with self._lock:
            if not self._frame_completion_time:
                return None
            
            # 找到最新完成的帧
            latest_frame_id = max(self._frame_completion_time, 
                                key=lambda k: self._frame_completion_time[k])
            
            if latest_frame_id in self._frame_to_data:
                frame = self._frame_to_data[latest_frame_id]
                return {
                    "frame_id": latest_frame_id,
                    "timestamp": frame.timestamp,
                    "categories": frame.categories,
                    "base64_data": frame.base64_data,
                    "metadata": frame.metadata
                }
        
        return None
    
    def query_frames_for_question(self, available_categories: List[str], 
                                 question: str) -> List[MemoryFrame]:
        """根据问题查询相关的记忆帧（简化版本，主要基于类别匹配）"""
        # 这里可以集成更复杂的查询逻辑，比如语义匹配
        # 目前简单实现：返回问题中可能涉及的类别的最新帧
        
        relevant_frames = []
        question_lower = question.lower()
        
        with self._lock:
            # 简单的关键词匹配
            for category in available_categories:
                if category.lower() in question_lower:
                    latest_frames = self.get_frames_by_category(category, max_count=2)
                    relevant_frames.extend(latest_frames)
        
        # 如果没有直接匹配，返回所有类别的最新帧
        if not relevant_frames:
            for category in available_categories[:3]:  # 限制类别数量
                latest_frames = self.get_frames_by_category(category, max_count=1)
                relevant_frames.extend(latest_frames)
        
        return relevant_frames
    
    def _cleanup_frames(self) -> None:
        """清理过期和超量的帧"""
        try:
            current_time = time.time()
            total_frames = len(self._frame_to_data)
            
            # 按时间清理过期帧
            expired_frames = []
            for frame_id, frame in self._frame_to_data.items():
                if current_time - frame.created_at > self.frame_ttl:
                    expired_frames.append(frame_id)
            
            # 删除过期帧
            for frame_id in expired_frames:
                self._remove_frame(frame_id)
            
            # 按数量限制清理
            if total_frames > self.max_total_frames:
                # 按创建时间排序，删除最旧的帧
                sorted_frames = sorted(
                    self._frame_to_data.items(),
                    key=lambda x: x[1].created_at
                )
                
                frames_to_remove = total_frames - self.max_total_frames
                for i in range(frames_to_remove):
                    frame_id = sorted_frames[i][0]
                    self._remove_frame(frame_id)
            
            # 清理每个类别的帧数量限制
            for category in list(self._category_to_frames.keys()):
                frame_ids = self._category_to_frames[category]
                if len(frame_ids) > self.max_frames_per_category:
                    # 保留最新的帧
                    frames_to_keep = frame_ids[-self.max_frames_per_category:]
                    self._category_to_frames[category] = frames_to_keep
            
            if expired_frames:
                logger.debug(f"Cleaned up {len(expired_frames)} expired frames")
                
        except Exception as e:
            logger.error(f"Failed to cleanup frames: {e}")
    
    def _remove_frame(self, frame_id: str) -> None:
        """删除指定帧"""
        # 从磁盘删除frame文件
        try:
            frames_dir = self.storage_dir / "frames"
            
            # 删除JPG图片文件
            image_file = frames_dir / f"{frame_id}.jpg"
            if image_file.exists():
                image_file.unlink()
                logger.debug(f"Deleted image file for {frame_id}")
            
            # 删除元数据文件
            metadata_file = frames_dir / f"{frame_id}_meta.json"
            if metadata_file.exists():
                metadata_file.unlink()
                logger.debug(f"Deleted metadata file for {frame_id}")
                
        except Exception as e:
            logger.warning(f"Failed to delete frame files for {frame_id}: {e}")
        
        # 从内存中删除
        if frame_id in self._frame_to_data:
            del self._frame_to_data[frame_id]
        
        if frame_id in self._frame_completion_time:
            del self._frame_completion_time[frame_id]
        
        # 从类别索引中删除
        for category, frame_ids in self._category_to_frames.items():
            if frame_id in frame_ids:
                frame_ids.remove(frame_id)
    
    def _persist_category_mappings(self) -> None:
        """持久化类别映射到文件"""
        try:
            categories_dir = self.storage_dir / "categories"
            
            for category, frame_ids in self._category_to_frames.items():
                category_file = categories_dir / f"{category}_frames.json"
                
                with open(category_file, 'w', encoding='utf-8') as f:
                    for frame_id in frame_ids:
                        f.write(f"{frame_id}\n")
                        
        except Exception as e:
            logger.warning(f"Failed to persist category mappings: {e}")
    
    def _persist_frame_data(self, memory_frame: MemoryFrame) -> None:
        """持久化单个frame数据到磁盘"""
        try:
            frames_dir = self.storage_dir / "frames"
            
            # 保存图片文件
            if memory_frame.base64_data:
                # 解码base64并保存为JPG
                try:
                    image_data = base64.b64decode(memory_frame.base64_data)
                    image_file = frames_dir / f"{memory_frame.frame_id}.jpg"
                    
                    with open(image_file, 'wb') as f:
                        f.write(image_data)
                    
                    logger.debug(f"Saved image file for {memory_frame.frame_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to save image for {memory_frame.frame_id}: {e}")
            
            # 保存元数据文件
            metadata_file = frames_dir / f"{memory_frame.frame_id}_meta.json"
            frame_data = {
                "frame_id": memory_frame.frame_id,
                "timestamp": memory_frame.timestamp,
                "categories": memory_frame.categories,
                "metadata": memory_frame.metadata,
                "created_at": memory_frame.created_at,
                "has_image": bool(memory_frame.base64_data)
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(frame_data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Persisted frame data and image for {memory_frame.frame_id}")
            
        except Exception as e:
            logger.warning(f"Failed to persist frame data for {memory_frame.frame_id}: {e}")
    
    def clear_all(self) -> None:
        """清空所有记忆数据"""
        try:
            with self._lock:
                self._category_to_frames.clear()
                self._frame_to_data.clear()
                self._frame_completion_time.clear()
                
                # 清理存储目录
                if self.storage_dir.exists():
                    shutil.rmtree(self.storage_dir)
                    self._init_storage()
                
                logger.info("All memory data cleared")
                
        except Exception as e:
            logger.error(f"Failed to clear memory data: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆存储的统计信息"""
        with self._lock:
            total_frames = len(self._frame_to_data)
            total_categories = len(self._category_to_frames)
            
            # 计算各类别的帧数
            category_counts = {
                category: len(frame_ids) 
                for category, frame_ids in self._category_to_frames.items()
            }
            
            return {
                "total_frames": total_frames,
                "total_categories": total_categories,
                "category_counts": category_counts,
                "storage_dir": str(self.storage_dir),
                "max_frames_per_category": self.max_frames_per_category,
                "max_total_frames": self.max_total_frames
            }
