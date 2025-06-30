#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内存队列实现

基于asyncio的简单内存队列
"""

import asyncio
import time
from collections import deque
from typing import Dict, Any, Optional
from loguru import logger

from .base import BaseMessageQueue


class InMemoryQueue(BaseMessageQueue):
    """内存队列实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._queue = deque()
        self._condition = asyncio.Condition()
    
    async def _do_initialize(self) -> bool:
        """初始化内存队列"""
        try:
            self._queue.clear()
            logger.info(f"内存队列 {self.queue_name} 初始化完成")
            return True
        except Exception as e:
            logger.error(f"初始化内存队列失败: {e}")
            return False
    
    async def _do_put(self, item: Any, timeout: float) -> bool:
        """向内存队列添加消息"""
        try:
            async with self._condition:
                if len(self._queue) >= self.max_size:
                    logger.warning(f"内存队列已满 (大小: {len(self._queue)})")
                    return False
                
                self._queue.append(item)
                self._condition.notify()
                return True
        except Exception as e:
            logger.error(f"向内存队列添加消息失败: {e}")
            return False
    
    async def _do_get(self, timeout: float) -> Optional[Any]:
        """从内存队列获取消息"""
        try:
            async with self._condition:
                end_time = time.time() + timeout
                
                while not self._queue:
                    remaining_time = end_time - time.time()
                    if remaining_time <= 0:
                        return None
                    
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(), 
                            timeout=remaining_time
                        )
                    except asyncio.TimeoutError:
                        return None
                
                return self._queue.popleft()
        except Exception as e:
            logger.error(f"从内存队列获取消息失败: {e}")
            return None
    
    def size(self) -> int:
        """获取队列大小"""
        return len(self._queue)
    
    async def _do_close(self) -> None:
        """关闭内存队列"""
        try:
            async with self._condition:
                self._queue.clear()
                self._condition.notify_all()
        except Exception as e:
            logger.error(f"关闭内存队列失败: {e}") 