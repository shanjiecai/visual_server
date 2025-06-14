#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缓存模块
提供各种缓存实现，包括内存缓存、LRU缓存等
"""

import asyncio
import time
from loguru import logger
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import json
import hashlib
from dataclasses import asdict

from core.interfaces import ICache, FrameData, ProcessingResult


class BaseCache(ICache):
    """缓存基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_name = config.get("cache_name", "default")
        self._is_connected = False
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
        }
        
        # 配置参数
        self.max_size = config.get("max_size", 1000)
        self.default_ttl = config.get("default_ttl", 300)  # 5分钟
        self.serialize_values = config.get("serialize_values", True)
    
    async def initialize(self) -> bool:
        """初始化缓存"""
        try:
            logger.info(f"Initializing cache: {self.cache_name}")
            result = await self._do_initialize()
            if result:
                self._is_connected = True
                logger.info(f"Cache {self.cache_name} initialized successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to initialize cache {self.cache_name}: {e}")
            return False
    
    @abstractmethod
    async def _do_initialize(self) -> bool:
        """子类实现具体的初始化逻辑"""
        pass
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self._is_connected:
            raise RuntimeError("Cache not connected")
        
        try:
            value = await self._do_get(key)
            if value is not None:
                self._stats["hits"] += 1
                logger.debug(f"Cache hit for key: {key}")
                # 反序列化
                if self.serialize_values and isinstance(value, str):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                return value
            else:
                self._stats["misses"] += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
                
        except Exception as e:
            self._stats["misses"] += 1
            logger.error(f"Error getting cache value for key {key}: {e}")
            return None
    
    @abstractmethod
    async def _do_get(self, key: str) -> Optional[Any]:
        """子类实现具体的获取逻辑"""
        pass
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        if not self._is_connected:
            raise RuntimeError("Cache not connected")
        
        try:
            result = await self._do_set(key, value, ttl or self.default_ttl)
            if result:
                self._stats["sets"] += 1
                logger.debug(f"Cache set for key: {key}")
            return result
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {e}")
            return False
    
    @abstractmethod
    async def _do_set(self, key: str, value: Any, ttl: int) -> bool:
        """子类实现具体的设置逻辑"""
        pass
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self._is_connected:
            raise RuntimeError("Cache not connected")
        
        try:
            result = await self._do_delete(key)
            if result:
                self._stats["deletes"] += 1
                logger.debug(f"Cache deleted for key: {key}")
            return result
        except Exception as e:
            logger.error(f"Error deleting cache value for key {key}: {e}")
            return False
    
    @abstractmethod
    async def _do_delete(self, key: str) -> bool:
        """子类实现具体的删除逻辑"""
        pass
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            value = await self.get(key)
            return value is not None
        except Exception:
            return False
    
    async def clear(self) -> bool:
        """清空所有缓存"""
        try:
            success = await self._do_clear()
            if success:
                logger.info("Cache cleared successfully")
            return success
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    @abstractmethod
    async def _do_clear(self) -> bool:
        """子类实现具体的清空逻辑"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_name": self.cache_name,
            "is_connected": self._is_connected,
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "deletes": self._stats["deletes"],
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "is_initialized": self._is_initialized,
        }


class InMemoryCache(BaseCache):
    """内存缓存实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._cache: Dict[str, Tuple[Any, float]] = {}  # (value, expire_time)
        self._access_times: Dict[str, float] = {}  # 用于LRU
        self._lock = asyncio.Lock()
        
        # 配置参数
        self.cleanup_interval = config.get("cleanup_interval", 300)  # 5分钟
        self._cleanup_task = None
    
    async def _do_initialize(self) -> bool:
        """初始化内存缓存"""
        try:
            self._cache.clear()
            self._access_times.clear()
            
            # 启动清理任务
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._logger.info(f"In-memory cache initialized with max size: {self.max_size}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize in-memory cache: {e}")
            return False
    
    async def _do_get(self, key: str) -> Optional[Any]:
        """从内存缓存获取值"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, expire_time = self._cache[key]
            
            # 检查是否过期
            if expire_time < time.time():
                del self._cache[key]
                self._access_times.pop(key, None)
                return None
            
            # 更新访问时间
            self._access_times[key] = time.time()
            return value
    
    async def _do_set(self, key: str, value: Any, ttl: int) -> bool:
        """向内存缓存设置值"""
        async with self._lock:
            try:
                # 检查缓存大小限制
                if len(self._cache) >= self.max_size and key not in self._cache:
                    # 移除最少访问的项目
                    await self._evict_lru()
                
                expire_time = time.time() + ttl
                self._cache[key] = (value, expire_time)
                self._access_times[key] = time.time()
                
                return True
                
            except Exception as e:
                self._logger.error(f"Error setting value in memory cache: {e}")
                return False
    
    async def _do_delete(self, key: str) -> bool:
        """从内存缓存删除值"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_times.pop(key, None)
                return True
            return False
    
    async def _do_clear(self) -> bool:
        """清空内存缓存"""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            return True
    
    async def _evict_lru(self) -> None:
        """驱逐最少使用的项目"""
        if not self._access_times:
            return
        
        # 找到最少访问的键
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # 移除该项目
        self._cache.pop(lru_key, None)
        self._access_times.pop(lru_key, None)
        
        self._logger.debug(f"Evicted LRU item with key: {lru_key}")
    
    async def _cleanup_loop(self) -> None:
        """定期清理过期项目"""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            self._logger.info("Cache cleanup task cancelled")
        except Exception as e:
            self._logger.error(f"Error in cache cleanup loop: {e}")
    
    async def _cleanup_expired(self) -> None:
        """清理过期的缓存项目"""
        async with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, expire_time) in self._cache.items():
                if expire_time < current_time:
                    expired_keys.append(key)
            
            # 移除过期项目
            for key in expired_keys:
                del self._cache[key]
                self._access_times.pop(key, None)
            
            if expired_keys:
                self._logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    async def close(self) -> None:
        """关闭缓存"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self._do_clear()
        self._logger.info("In-memory cache closed")


class LRUCache(BaseCache):
    """LRU缓存实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._cache = OrderedDict()
        self._expire_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def _do_initialize(self) -> bool:
        """初始化LRU缓存"""
        try:
            self._cache.clear()
            self._expire_times.clear()
            
            self._logger.info(f"LRU cache initialized with max size: {self.max_size}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize LRU cache: {e}")
            return False
    
    async def _do_get(self, key: str) -> Optional[Any]:
        """从LRU缓存获取值"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            # 检查是否过期
            if key in self._expire_times and self._expire_times[key] < time.time():
                del self._cache[key]
                del self._expire_times[key]
                return None
            
            # 移动到末尾（最近使用）
            value = self._cache.pop(key)
            self._cache[key] = value
            
            return value
    
    async def _do_set(self, key: str, value: Any, ttl: int) -> bool:
        """向LRU缓存设置值"""
        async with self._lock:
            try:
                # 如果键已存在，更新值
                if key in self._cache:
                    self._cache.pop(key)
                
                # 检查大小限制
                while len(self._cache) >= self.max_size:
                    # 移除最老的项目
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._expire_times.pop(oldest_key, None)
                
                # 添加新项目
                self._cache[key] = value
                self._expire_times[key] = time.time() + ttl
                
                return True
                
            except Exception as e:
                self._logger.error(f"Error setting value in LRU cache: {e}")
                return False
    
    async def _do_delete(self, key: str) -> bool:
        """从LRU缓存删除值"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._expire_times.pop(key, None)
                return True
            return False
    
    async def _do_clear(self) -> bool:
        """清空LRU缓存"""
        async with self._lock:
            self._cache.clear()
            self._expire_times.clear()
            return True
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


class FrameCache(BaseCache):
    """视频帧专用缓存"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._frame_cache: Dict[str, Tuple[FrameData, float]] = {}
        self._result_cache: Dict[str, Tuple[List[ProcessingResult], float]] = {}
        self._lock = asyncio.Lock()
        
        # 配置参数
        self.frame_ttl = config.get("frame_ttl", 1800)  # 30分钟
        self.result_ttl = config.get("result_ttl", 3600)  # 1小时
        self.max_frame_size = config.get("max_frame_size", 500)
        self.max_result_size = config.get("max_result_size", 1000)
    
    async def _do_initialize(self) -> bool:
        """初始化帧缓存"""
        try:
            self._frame_cache.clear()
            self._result_cache.clear()
            
            self._logger.info(f"Frame cache initialized (frames: {self.max_frame_size}, results: {self.max_result_size})")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize frame cache: {e}")
            return False
    
    async def cache_frame(self, frame_data: FrameData) -> bool:
        """缓存视频帧"""
        key = self._generate_frame_key(frame_data)
        return await self.set(key, frame_data, self.frame_ttl)
    
    async def get_cached_frame(self, frame_id: str) -> Optional[FrameData]:
        """获取缓存的视频帧"""
        key = f"frame:{frame_id}"
        return await self.get(key)
    
    async def cache_processing_results(self, frame_id: str, results: List[ProcessingResult]) -> bool:
        """缓存处理结果"""
        key = f"results:{frame_id}"
        return await self.set(key, results, self.result_ttl)
    
    async def get_cached_results(self, frame_id: str) -> Optional[List[ProcessingResult]]:
        """获取缓存的处理结果"""
        key = f"results:{frame_id}"
        return await self.get(key)
    
    async def cache_similar_frame_hash(self, frame_data: FrameData, frame_hash: str) -> bool:
        """缓存相似帧的哈希值"""
        key = f"hash:{frame_data.frame_id}"
        return await self.set(key, frame_hash, self.frame_ttl)
    
    async def get_similar_frame_hash(self, frame_id: str) -> Optional[str]:
        """获取相似帧的哈希值"""
        key = f"hash:{frame_id}"
        return await self.get(key)
    
    def _generate_frame_key(self, frame_data: FrameData) -> str:
        """生成帧缓存键"""
        return f"frame:{frame_data.frame_id}"
    
    async def _do_get(self, key: str) -> Optional[Any]:
        """获取缓存值的实现"""
        async with self._lock:
            # 检查帧缓存
            if key.startswith("frame:") and key in self._frame_cache:
                frame_data, expire_time = self._frame_cache[key]
                if expire_time > time.time():
                    return frame_data
                else:
                    del self._frame_cache[key]
            
            # 检查结果缓存
            elif key.startswith("results:") and key in self._result_cache:
                results, expire_time = self._result_cache[key]
                if expire_time > time.time():
                    return results
                else:
                    del self._result_cache[key]
            
            return None
    
    async def _do_set(self, key: str, value: Any, ttl: int) -> bool:
        """设置缓存值的实现"""
        async with self._lock:
            try:
                expire_time = time.time() + ttl
                
                if key.startswith("frame:"):
                    # 检查帧缓存大小
                    while len(self._frame_cache) >= self.max_frame_size:
                        oldest_key = min(self._frame_cache.keys(), 
                                       key=lambda k: self._frame_cache[k][1])
                        del self._frame_cache[oldest_key]
                    
                    self._frame_cache[key] = (value, expire_time)
                
                elif key.startswith("results:"):
                    # 检查结果缓存大小
                    while len(self._result_cache) >= self.max_result_size:
                        oldest_key = min(self._result_cache.keys(),
                                       key=lambda k: self._result_cache[k][1])
                        del self._result_cache[oldest_key]
                    
                    self._result_cache[key] = (value, expire_time)
                
                return True
                
            except Exception as e:
                self._logger.error(f"Error setting frame cache value: {e}")
                return False
    
    async def _do_delete(self, key: str) -> bool:
        """删除缓存值的实现"""
        async with self._lock:
            deleted = False
            
            if key in self._frame_cache:
                del self._frame_cache[key]
                deleted = True
            
            if key in self._result_cache:
                del self._result_cache[key]
                deleted = True
            
            return deleted
    
    async def _do_clear(self) -> bool:
        """清空缓存的实现"""
        async with self._lock:
            self._frame_cache.clear()
            self._result_cache.clear()
            return True
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._frame_cache) + len(self._result_cache)
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """获取详细的缓存统计信息"""
        base_stats = self.get_statistics()
        
        base_stats.update({
            "frame_cache_size": len(self._frame_cache),
            "result_cache_size": len(self._result_cache),
            "max_frame_size": self.max_frame_size,
            "max_result_size": self.max_result_size,
            "frame_ttl": self.frame_ttl,
            "result_ttl": self.result_ttl,
        })
        
        return base_stats 