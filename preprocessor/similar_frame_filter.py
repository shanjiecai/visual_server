#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
相似帧过滤器实现
用于过滤相似的视频帧，减少后续处理的计算量
"""

import asyncio
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger
import torch
from PIL import Image

# 添加CLIP相关导入
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("transformers库未安装或CLIP模型不可用，将使用传统方法进行相似度计算")

from .base import BasePreprocessor
from core.interfaces import FrameData, ProcessingResult


class SimilarFrameFilterProcessor(BasePreprocessor):
    """相似帧过滤处理器实现"""
    
    @property
    def processor_name(self) -> str:
        return "similar_frame_filter"
    
    async def _do_initialize(self) -> bool:
        """初始化相似帧过滤器"""
        try:
            # 相似度阈值，超过此值认为是相似帧
            self.similarity_threshold = self.config.get("similarity_threshold", 0.95)
            
            # 比较方法：histogram, mse, ssim, hash, clip
            self.comparison_method = self.config.get("comparison_method", "histogram")
            
            # 历史帧保存数量
            self.history_size = self.config.get("history_size", 5)
            
            # 是否跳过相似帧（True=跳过，False=标记但不跳过）
            self.skip_similar = self.config.get("skip_similar", True)
            
            # 最小时间间隔（秒），避免过于频繁的帧
            self.min_time_interval = self.config.get("min_time_interval", 0.5)
            
            # 是否用最新的相似帧替换历史帧中最相似的帧（True=替换，False=不替换）
            self.replace_similar_frame = self.config.get("replace_similar_frame", True)
            
            # 初始化历史帧存储
            self._frame_history = []
            self._last_processed_time = 0
            
            # 初始化CLIP模型（如果选择了CLIP方法）
            self.clip_model = None
            self.clip_processor = None
            self.clip_device = None
            
            if self.comparison_method == "clip" and CLIP_AVAILABLE:
                # 获取CLIP模型路径
                clip_model_path = self.config.get("clip_model_path", "models/clip-vit-base-patch32")
                
                # 在单独的线程中加载模型，避免阻塞
                await self._load_clip_model(clip_model_path)
                
                if self.clip_model is None:
                    logger.warning("CLIP模型加载失败，将使用直方图方法作为备选")
                    self.comparison_method = "histogram"
            
            logger.info(f"Similar frame filter initialized: method={self.comparison_method}, threshold={self.similarity_threshold}, replace_similar={self.replace_similar_frame}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize similar frame filter: {e}")
            return False
    
    async def _load_clip_model(self, model_path: str) -> None:
        """异步加载CLIP模型"""
        try:
            logger.info(f"Loading CLIP model from {model_path}...")
            
            # 在线程池中加载模型（避免阻塞主事件循环）
            self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 使用线程池执行模型加载
            self.clip_model, self.clip_processor = await self._run_in_thread(
                self._load_clip_model_sync, model_path, self.clip_device
            )
            
            logger.info(f"CLIP model loaded successfully on {self.clip_device}")
            
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def _load_clip_model_sync(self, model_path: str, device: str):
        """同步加载CLIP模型（在线程池中执行）"""
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        return model, processor
    
    async def _do_process(self, frame_data: FrameData) -> ProcessingResult:
        """执行相似帧过滤"""
        try:
            current_time = time.time()
            
            # 时间间隔检查
            if current_time - self._last_processed_time < self.min_time_interval:
                logger.info(f"帧 {frame_data.frame_id} 因时间间隔过短被过滤 (间隔: {current_time - self._last_processed_time:.2f}s < {self.min_time_interval}s)")
                return self._create_filtered_result(frame_data, "time_interval_too_short", should_skip=True)
            
            # 从帧数据中提取图像
            image = self._extract_image_from_frame(frame_data)
            if image is None:
                logger.warning(f"帧 {frame_data.frame_id} 无法提取有效图像，被过滤")
                return self._create_filtered_result(frame_data, "invalid_image", should_skip=True)
            
            # 计算与历史帧的相似度
            logger.info(f"开始计算帧 {frame_data.frame_id} 的相似度 (方法: {self.comparison_method})...")
            similarity_result = await self._calculate_similarity(image, frame_data)
            
            # 判断是否应该跳过这一帧
            should_skip = self.skip_similar and similarity_result["is_similar"]
            
            # 如果相似度过高，处理替换逻辑或过滤
            if similarity_result["is_similar"]:
                if self.replace_similar_frame and similarity_result["most_similar_index"] >= 0:
                    # 用新帧替换历史帧中最相似的那个
                    old_frame_id = self._frame_history[similarity_result["most_similar_index"]]["frame_id"]
                    self._frame_history[similarity_result["most_similar_index"]] = {
                        "frame_id": frame_data.frame_id,
                        "timestamp": frame_data.timestamp,
                        "image": image.copy(),
                        "added_at": time.time(),
                        "replaced_frame": old_frame_id  # 记录被替换的帧ID
                    }
                    self._last_processed_time = current_time
                    logger.info(f"帧 {frame_data.frame_id} 替换了历史帧 {old_frame_id} (相似度: {similarity_result['max_similarity']:.4f})")
                    similarity_result["reason"] = "high_similarity"
                    # 创建替换结果
                    result = self._create_result(
                        frame_data,
                        similarity_result,
                        confidence=1.0 - similarity_result["max_similarity"],
                        additional_metadata={
                            "should_skip": True,
                            "comparison_method": self.comparison_method,
                            "history_size": len(self._frame_history),
                            "processed_time": current_time,
                            "action": "replaced",
                            "replaced_frame_id": old_frame_id,
                            "replaced_at_index": similarity_result["most_similar_index"]
                        }
                    )
                    
                    logger.debug(f"相似帧替换结果摘要 - 帧 {frame_data.frame_id}:")
                    logger.debug(f"  - 替换的历史帧: {old_frame_id}")
                    logger.debug(f"  - 相似度: {similarity_result['max_similarity']:.4f}")
                    logger.debug(f"  - 历史帧数: {len(self._frame_history)}")
                    
                    return result
                
                elif should_skip:
                    # 如果不替换且设置了跳过相似帧，则过滤
                    logger.info(f"帧 {frame_data.frame_id} 因相似度过高被过滤 (相似度: {similarity_result['max_similarity']:.4f} > {self.similarity_threshold})")
                    return self._create_filtered_result(frame_data, "high_similarity", should_skip=True)
            
            # 如果不相似或相似但不跳过，更新历史帧
            if not similarity_result["is_similar"] or not should_skip:
                self._update_frame_history(image, frame_data)
                self._last_processed_time = current_time
                action = "added_new" if not similarity_result["is_similar"] else "added_similar"
                logger.info(f"帧 {frame_data.frame_id} 添加到历史帧 (当前历史帧数: {len(self._frame_history)}, 动作: {action})")
                
                # 创建处理结果
                result = self._create_result(
                    frame_data,
                    similarity_result,
                    confidence=1.0 - similarity_result["max_similarity"],  # 越不相似置信度越高
                    additional_metadata={
                        "should_skip": False,  # 不跳过
                        "comparison_method": self.comparison_method,
                        "history_size": len(self._frame_history),
                        "processed_time": current_time,
                        "action": action
                    }
                )
                
                # 打印总结信息
                logger.debug(f"相似度处理结果摘要 - 帧 {frame_data.frame_id}:")
                logger.debug(f"  - 方法: {self.comparison_method}")
                logger.debug(f"  - 最大相似度: {similarity_result['max_similarity']:.4f} (阈值: {self.similarity_threshold})")
                logger.debug(f"  - 是否相似: {similarity_result['is_similar']}")
                logger.debug(f"  - 处理动作: {action}")
                logger.debug(f"  - 置信度: {1.0 - similarity_result['max_similarity']:.4f}")
                logger.debug(f"  - 历史帧数: {len(self._frame_history)}")
                
                return result
            
        except Exception as e:
            logger.error(f"帧 {frame_data.frame_id} 相似度过滤失败: {e}")
            raise RuntimeError(f"Similar frame filtering failed: {e}")
    
    def _extract_image_from_frame(self, frame_data: FrameData) -> Optional[np.ndarray]:
        """从帧数据中提取图像"""
        try:
            raw_data = frame_data.raw_data
            
            # 如果已经是numpy数组
            if isinstance(raw_data, np.ndarray):
                return raw_data
            
            # 如果是字典格式
            if isinstance(raw_data, dict) and "data" in raw_data:
                data = raw_data["data"]
                if isinstance(data, np.ndarray):
                    return data
                # 模拟数据，创建测试图像
                if isinstance(data, str):
                    width = raw_data.get("width", 640)
                    height = raw_data.get("height", 480)
                    # 创建带有帧ID信息的测试图像
                    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                    # 在图像上添加帧ID文本，增加变化
                    cv2.putText(image, f"Frame_{frame_data.frame_id}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    return image
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting image from frame: {e}")
            return None
    
    async def _calculate_similarity(self, current_image: np.ndarray, frame_data: FrameData) -> Dict[str, Any]:
        """计算帧相似度"""
        if not self._frame_history:
            return {
                "is_similar": False,
                "max_similarity": 0.0,
                "similarities": [],
                "most_similar_index": -1,
                "reason": "first_frame"
            }
        
        # 在线程池中计算相似度（CPU密集型操作）
        similarities = await self._run_in_thread(self._compute_similarities, current_image)
        
        max_similarity = max(similarities) if similarities else 0.0
        most_similar_index = similarities.index(max_similarity) if similarities else -1
        is_similar = max_similarity > self.similarity_threshold
        
        # 添加详细日志打印相似度结果
        logger.debug(f"帧 {frame_data.frame_id} 相似度计算结果:")
        logger.debug(f"  - 最大相似度: {max_similarity:.4f} (阈值: {self.similarity_threshold})")
        logger.debug(f"  - 是否相似: {is_similar}")
        logger.debug(f"  - 最相似的历史帧索引: {most_similar_index}")
        if most_similar_index >= 0:
            logger.debug(f"  - 最相似的历史帧ID: {self._frame_history[most_similar_index]['frame_id']}")
        logger.debug(f"  - 比较方法: {self.comparison_method}")
        logger.debug(f"  - 历史帧数量: {len(self._frame_history)}")
        
        return {
            "is_similar": is_similar,
            "max_similarity": max_similarity,
            "similarities": similarities,
            "most_similar_index": most_similar_index,
            "similarity_threshold": self.similarity_threshold,
            "comparison_method": self.comparison_method,
            "compared_frames": len(self._frame_history)
        }
    
    def _compute_similarities(self, current_image: np.ndarray) -> List[float]:
        """计算当前帧与历史帧的相似度（在线程池中运行）"""
        similarities = []
        
        for i, hist_frame in enumerate(self._frame_history):
            hist_image = hist_frame["image"]
            
            if self.comparison_method == "clip" and self.clip_model is not None:
                similarity = self._clip_similarity(current_image, hist_image)
            elif self.comparison_method == "histogram":
                similarity = self._histogram_similarity(current_image, hist_image)
            elif self.comparison_method == "mse":
                similarity = self._mse_similarity(current_image, hist_image)
            elif self.comparison_method == "ssim":
                similarity = self._ssim_similarity(current_image, hist_image)
            elif self.comparison_method == "hash":
                similarity = self._hash_similarity(current_image, hist_image)
            else:
                similarity = self._histogram_similarity(current_image, hist_image)
            
            # 打印每一帧的相似度详情
            logger.debug(f"与历史帧 {hist_frame['frame_id']} 的相似度: {similarity:.4f} (方法: {self.comparison_method})")
            similarities.append(similarity)
        
        return similarities
    
    def _clip_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """使用CLIP模型计算图像相似度"""
        try:
            if self.clip_model is None or self.clip_processor is None:
                # 如果CLIP模型不可用，回退到直方图方法
                logger.warning("CLIP模型不可用，回退到直方图方法")
                return self._histogram_similarity(img1, img2)
            
            # 转换为PIL图像
            pil_image1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            pil_image2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            
            # 处理图像
            inputs1 = self.clip_processor(images=pil_image1, return_tensors="pt").to(self.clip_device)
            inputs2 = self.clip_processor(images=pil_image2, return_tensors="pt").to(self.clip_device)
            
            # 提取特征
            with torch.no_grad():
                features1 = self.clip_model.get_image_features(**inputs1)
                features2 = self.clip_model.get_image_features(**inputs2)
            
            # 归一化特征
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(features1, features2).item()
            
            logger.debug(f"CLIP相似度计算结果: {similarity:.4f} (设备: {self.clip_device})")
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"CLIP similarity calculation failed: {e}, falling back to histogram method")
            return self._histogram_similarity(img1, img2)
    
    def _histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """基于直方图的相似度计算"""
        try:
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            # 计算直方图
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            # 使用相关系数比较
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            logger.debug(f"直方图相似度计算结果: {similarity:.4f}")
            return max(0.0, similarity)  # 确保非负
            
        except Exception as e:
            logger.error(f"Histogram similarity calculation failed: {e}")
            return 0.0
    
    def _mse_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """基于均方误差的相似度计算"""
        try:
            # 调整图像大小以确保一致性
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(img1, (w, h))
            img2_resized = cv2.resize(img2, (w, h))
            
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY) if len(img1_resized.shape) == 3 else img1_resized
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY) if len(img2_resized.shape) == 3 else img2_resized
            
            # 计算MSE
            mse = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)
            
            # 转换为相似度（MSE越小，相似度越高）
            max_val = 255.0 * 255.0
            similarity = 1.0 - (mse / max_val)
            
            logger.debug(f"MSE相似度计算结果: {similarity:.4f} (原始MSE: {mse:.2f})")
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"MSE similarity calculation failed: {e}")
            return 0.0
    
    def _ssim_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """简化的结构相似性指数计算"""
        try:
            # 调整图像大小
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(img1, (w, h))
            img2_resized = cv2.resize(img2, (w, h))
            
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY) if len(img1_resized.shape) == 3 else img1_resized
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY) if len(img2_resized.shape) == 3 else img2_resized
            
            # 简化的SSIM计算
            mu1 = np.mean(gray1)
            mu2 = np.mean(gray2)
            sigma1 = np.std(gray1)
            sigma2 = np.std(gray2)
            sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
            
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
            
            logger.debug(f"SSIM相似度计算结果: {ssim:.4f} (均值: {mu1:.2f}/{mu2:.2f}, 方差: {sigma1:.2f}/{sigma2:.2f})")
            return max(0.0, ssim)
            
        except Exception as e:
            logger.error(f"SSIM similarity calculation failed: {e}")
            return 0.0
    
    def _hash_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """基于感知哈希的相似度计算"""
        try:
            def dhash(image, hash_size=8):
                # 调整大小
                resized = cv2.resize(image, (hash_size + 1, hash_size))
                # 转换为灰度
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
                # 计算差分哈希
                diff = gray[:, 1:] > gray[:, :-1]
                return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
            
            hash1 = dhash(img1)
            hash2 = dhash(img2)
            
            # 计算汉明距离
            hamming_distance = bin(hash1 ^ hash2).count('1')
            # 转换为相似度
            max_distance = 64  # 8x8的哈希
            similarity = 1.0 - (hamming_distance / max_distance)
            
            logger.debug(f"哈希相似度计算结果: {similarity:.4f} (汉明距离: {hamming_distance})")
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Hash similarity calculation failed: {e}")
            return 0.0
    
    def _update_frame_history(self, image: np.ndarray, frame_data: FrameData) -> None:
        """更新历史帧"""
        # 添加当前帧到历史
        self._frame_history.append({
            "frame_id": frame_data.frame_id,
            "timestamp": frame_data.timestamp,
            "image": image.copy(),  # 保存图像副本
            "added_at": time.time()
        })
        
        # 保持历史大小限制
        if len(self._frame_history) > self.history_size:
            self._frame_history.pop(0)
    
    def _create_filtered_result(self, frame_data: FrameData, reason: str, should_skip: bool = True) -> ProcessingResult:
        """创建过滤结果"""
        return ProcessingResult(
            frame_id=frame_data.frame_id,
            processor_name=self.processor_name,
            result_data={
                "filtered": True,
                "reason": reason,
                "should_skip": should_skip
            },
            confidence=0.0,
            metadata={
                "status": "filtered",
                "reason": reason,
                "should_skip": should_skip
            },
            timestamp=time.time()
        )
    
    async def _do_cleanup(self) -> None:
        """清理历史帧数据"""
        self._frame_history.clear()
        
        # 释放CLIP模型资源
        if self.clip_model is not None and hasattr(self.clip_model, 'cpu'):
            self.clip_model = self.clip_model.cpu()
            self.clip_model = None
            self.clip_processor = None
            # 手动触发垃圾回收
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("Frame history and model resources cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取过滤器统计信息"""
        base_stats = super().get_health_status()
        base_stats.update({
            "similarity_threshold": self.similarity_threshold,
            "comparison_method": self.comparison_method,
            "history_size": len(self._frame_history),
            "max_history_size": self.history_size,
            "skip_similar": self.skip_similar,
            "replace_similar_frame": self.replace_similar_frame,
            "min_time_interval": self.min_time_interval,
            "clip_model_loaded": self.clip_model is not None
        })
        return base_stats
