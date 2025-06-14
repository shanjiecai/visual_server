#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
消息队列基类模块
"""

import asyncio
from loguru import logger
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from core.interfaces import IMessageQueue, FrameData

class BaseMessageQueue(IMessageQueue):
    """消息队列基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queue_name = config.get("queue_name", "default")
        self.max_size = config.get("max_size", 1000)
        self.timeout = config.get("timeout", 30.0)
        
        self._is_connected = False
    
    async def initialize(self) -> bool:
        """初始化消息队列"""
        try:
            logger.info(f"Initializing message queue: {self.queue_name}")
            result = await self._do_initialize()
            if result:
                self._is_connected = True
                logger.info(f"Message queue {self.queue_name} initialized successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to initialize message queue {self.queue_name}: {e}")
            return False
    
    @abstractmethod
    async def _do_initialize(self) -> bool:
        """子类实现具体的初始化逻辑"""
        pass
    
    async def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """发送数据到队列"""
        if not self._is_connected:
            raise RuntimeError("Message queue not connected")
        
        try:
            result = await self._do_put(item, timeout or self.timeout)
            if result:
                logger.debug(f"Successfully put data to queue {self.queue_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to put data to queue {self.queue_name}: {e}")
            return False
    
    @abstractmethod
    async def _do_put(self, data: FrameData, timeout: float) -> bool:
        """子类实现具体的数据发送逻辑"""
        pass
    
    async def get(self, timeout: Optional[float] = None) -> Optional[FrameData]:
        """从队列获取数据"""
        if not self._is_connected:
            raise RuntimeError("Message queue not connected")
        
        try:
            data = await self._do_get(timeout or self.timeout)
            if data:
                logger.debug(f"Successfully got data from queue {self.queue_name}")
            return data
        except Exception as e:
            logger.error(f"Failed to get data from queue {self.queue_name}: {e}")
            return None
    
    @abstractmethod
    async def _do_get(self, timeout: float) -> Optional[FrameData]:
        """子类实现具体的数据接收逻辑"""
        pass
    
    async def close(self) -> None:
        """关闭队列连接"""
        try:
            logger.info(f"Closing message queue: {self.queue_name}")
            await self._do_close()
            self._is_connected = False
            logger.info(f"Message queue {self.queue_name} closed")
        except Exception as e:
            logger.error(f"Error closing message queue {self.queue_name}: {e}")
    
    @abstractmethod
    async def _do_close(self) -> None:
        """子类实现具体的关闭逻辑"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "queue_name": self.queue_name,
            "is_connected": self._is_connected,
            "max_size": self.max_size,
            "timeout": self.timeout
        }
