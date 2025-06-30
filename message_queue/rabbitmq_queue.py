#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RabbitMQ队列实现

基于aio-pika的RabbitMQ消息队列，当RabbitMQ不可用时降级为内存队列
"""

import asyncio
import json
import time
from collections import deque
from typing import Dict, Any, Optional
from loguru import logger

# 可选依赖检查
try:
    import aio_pika
    from aio_pika import connect_robust, Message
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

from .base import BaseMessageQueue


class RabbitMQQueue(BaseMessageQueue):
    """RabbitMQ队列实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self._connection = None
        self._channel = None
        self._queue = None
        self._memory_queue = deque()
        
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5672)
        self.username = config.get("username", "guest")
        self.password = config.get("password", "guest")
        
        self.use_rabbitmq = RABBITMQ_AVAILABLE and config.get("use_rabbitmq", True)
    
    async def _do_initialize(self) -> bool:
        """初始化RabbitMQ连接"""
        if not self.use_rabbitmq:
            logger.info(f"队列 {self.queue_name} 使用内存模式")
            return True
        
        try:
            # 建立连接
            connection_url = f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/"
            self._connection = await connect_robust(connection_url)
            
            # 创建通道和队列
            self._channel = await self._connection.channel()
            self._queue = await self._channel.declare_queue(self.queue_name, durable=True)
            
            logger.info(f"RabbitMQ队列 {self.queue_name} 初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ初始化失败: {e}，降级到内存模式")
            self.use_rabbitmq = False
            return True
    
    def _serialize_message(self, item: Any) -> bytes:
        """序列化消息"""
        try:
            message = {"data": item, "timestamp": time.time()}
            return json.dumps(message).encode('utf-8')
        except Exception as e:
            logger.error(f"消息序列化失败: {e}")
            return json.dumps({"error": "serialization_failed"}).encode('utf-8')
    
    def _deserialize_message(self, data: bytes) -> Any:
        """反序列化消息"""
        try:
            message = json.loads(data.decode('utf-8'))
            return message.get("data", message)
        except Exception as e:
            logger.error(f"消息反序列化失败: {e}")
            return None
    
    async def _do_put(self, item: Any, timeout: float) -> bool:
        """发送消息"""
        if not self.use_rabbitmq or not self._channel:
            return self._put_to_memory(item)
        
        try:
            message_body = self._serialize_message(item)
            message = Message(message_body)
            await self._channel.default_exchange.publish(message, routing_key=self.queue_name)
            return True
        except Exception as e:
            logger.error(f"RabbitMQ发送失败: {e}")
            return self._put_to_memory(item)
    
    def _put_to_memory(self, item: Any) -> bool:
        """发送到内存队列"""
        if len(self._memory_queue) >= self.max_size:
            return False
        self._memory_queue.append(item)
        return True
    
    async def _do_get(self, timeout: float) -> Optional[Any]:
        """接收消息"""
        if not self.use_rabbitmq or not self._queue:
            return self._get_from_memory()
        
        try:
            message = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            if message:
                message.ack()
                return self._deserialize_message(message.body)
            return None
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"RabbitMQ接收失败: {e}")
            return self._get_from_memory()
    
    def _get_from_memory(self) -> Optional[Any]:
        """从内存队列接收"""
        return self._memory_queue.popleft() if self._memory_queue else None
    
    def size(self) -> int:
        """获取队列大小"""
        return len(self._memory_queue) if not self.use_rabbitmq else 0
    
    async def _do_close(self) -> None:
        """关闭连接"""
        try:
            if self._channel and not self._channel.is_closed:
                await self._channel.close()
            if self._connection and not self._connection.is_closed:
                await self._connection.close()
            self._memory_queue.clear()
        except Exception as e:
            logger.error(f"关闭RabbitMQ连接失败: {e}")
