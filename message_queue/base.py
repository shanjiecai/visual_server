#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
消息队列基类

提供统一的异步消息队列接口
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

from core.interfaces import IMessageQueue


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
            result = await self._do_initialize()
            if result:
                self._is_connected = True
                logger.info(f"消息队列 {self.queue_name} 初始化成功")
            return result
        except Exception as e:
            logger.error(f"初始化消息队列失败: {e}")
            return False
    
    @abstractmethod
    async def _do_initialize(self) -> bool:
        """子类实现具体的初始化逻辑"""
        pass
    
    async def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """发送消息到队列"""
        if not self._is_connected:
            raise RuntimeError("消息队列未连接")
        
        try:
            return await self._do_put(item, timeout or self.timeout)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False
    
    @abstractmethod
    async def _do_put(self, item: Any, timeout: float) -> bool:
        """子类实现具体的消息发送逻辑"""
        pass
    
    async def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """从队列接收消息"""
        if not self._is_connected:
            raise RuntimeError("消息队列未连接")
        
        try:
            return await self._do_get(timeout or self.timeout)
        except Exception as e:
            logger.error(f"接收消息失败: {e}")
            return None
    
    @abstractmethod
    async def _do_get(self, timeout: float) -> Optional[Any]:
        """子类实现具体的消息接收逻辑"""
        pass
    
    async def close(self) -> None:
        """关闭队列连接"""
        try:
            await self._do_close()
            self._is_connected = False
            logger.info(f"消息队列 {self.queue_name} 关闭")
        except Exception as e:
            logger.error(f"关闭消息队列失败: {e}")
    
    @abstractmethod
    async def _do_close(self) -> None:
        """子类实现具体的关闭逻辑"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "queue_name": self.queue_name,
            "is_connected": self._is_connected,
            "max_size": self.max_size
        }
