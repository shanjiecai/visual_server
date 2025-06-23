#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内存队列实现
基于asyncio的高性能内存队列
"""

import asyncio
import time
from collections import deque
from typing import Dict, Any, Optional

from message_queue.base import BaseMessageQueue


class InMemoryQueue(BaseMessageQueue):
    """内存队列实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._queue = deque()
        self._priority_queue = []  # 优先级队列
        self._condition = asyncio.Condition()
        
        # 配置参数
        self.use_priority = config.get("use_priority", False)
    
    async def _do_initialize(self) -> bool:
        """初始化内存队列"""
        try:
            self._queue.clear()
            self._priority_queue.clear()
            self._logger.info("In-memory queue initialized")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize in-memory queue: {e}")
            return False
    
    async def _do_put(self, item: Any, priority: int) -> bool:
        """向内存队列中添加项目"""
        try:
            async with self._condition:
                if len(self._queue) + len(self._priority_queue) >= self.max_size:
                    self._logger.warning(f"Queue is full (size: {self.max_size})")
                    return False
                
                if self.use_priority and priority > 0:
                    # 插入优先级队列，保持优先级排序
                    import bisect
                    bisect.insort(self._priority_queue, (-priority, time.time(), item))
                else:
                    self._queue.append(item)
                
                self._condition.notify()
                return True
                
        except Exception as e:
            self._logger.error(f"Error putting item to in-memory queue: {e}")
            return False
    
    async def _do_get(self, timeout: float) -> Any:
        """从内存队列中获取项目"""
        try:
            async with self._condition:
                # 等待直到有数据或超时
                end_time = time.time() + timeout
                while len(self._queue) == 0 and len(self._priority_queue) == 0:
                    remaining_time = end_time - time.time()
                    if remaining_time <= 0:
                        return None
                    
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining_time)
                    except asyncio.TimeoutError:
                        return None
                
                # 优先从优先级队列获取
                if self._priority_queue:
                    _, _, item = self._priority_queue.pop(0)
                    return item
                elif self._queue:
                    return self._queue.popleft()
                else:
                    return None
                    
        except Exception as e:
            self._logger.error(f"Error getting item from in-memory queue: {e}")
            return None
    
    def size(self) -> int:
        """获取队列大小"""
        return len(self._queue) + len(self._priority_queue)
    
    async def close(self) -> None:
        """关闭队列"""
        try:
            async with self._condition:
                self._queue.clear()
                self._priority_queue.clear()
                self._condition.notify_all()
            self._logger.info("In-memory queue closed")
        except Exception as e:
            self._logger.error(f"Error closing in-memory queue: {e}") 