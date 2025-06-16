#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缓存模块
提供内存缓存实现
"""

import asyncio
import time
import json
from loguru import logger
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from core.interfaces import ICache


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
        }


class InMemoryCache(BaseCache):
    """内存缓存实现"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self._cache: Dict[str, Tuple[Any, float]] = {}  # (value, expire_time)
        self._access_times: Dict[str, float] = {}  # 用于LRU
        self._lock = asyncio.Lock()
        
        # 配置参数
        self.cleanup_interval = self.config.get("cleanup_interval", 300)  # 5分钟
        self._cleanup_task = None
    
    async def _do_initialize(self) -> bool:
        """初始化内存缓存"""
        try:
            self._cache.clear()
            self._access_times.clear()
            
            # 启动清理任务
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"In-memory cache initialized with max size: {self.max_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize in-memory cache: {e}")
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
                logger.error(f"Error setting value in memory cache: {e}")
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
        
        logger.debug(f"Evicted LRU item with key: {lru_key}")
    
    async def _cleanup_loop(self) -> None:
        """定期清理过期项目"""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            logger.info("Cache cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cache cleanup loop: {e}")
    
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
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
    
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
        logger.info("In-memory cache closed") 