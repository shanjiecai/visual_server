#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
消息队列模块

提供统一的异步消息队列接口：
- InMemoryQueue: 内存队列
- KafkaQueue: Kafka分布式队列
- RabbitMQQueue: RabbitMQ消息队列
"""

from typing import Dict, Any
from loguru import logger

from .base import BaseMessageQueue
from .memory_queue import InMemoryQueue

# 可选队列组件
try:
    from .kafka_queue import KafkaQueue
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    from .rabbitmq_queue import RabbitMQQueue
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False


def create_message_queue(queue_type: str, config: Dict[str, Any]) -> BaseMessageQueue:
    """创建消息队列实例
    
    Args:
        queue_type: 队列类型 ("memory", "kafka", "rabbitmq")
        config: 配置字典
        
    Returns:
        对应类型的消息队列实例
    """
    queue_type = queue_type.lower()
    
    if queue_type == "memory":
        return InMemoryQueue(config)
    elif queue_type == "kafka":
        if KAFKA_AVAILABLE:
            return KafkaQueue(config)
        else:
            logger.warning("Kafka不可用，使用内存队列")
            return InMemoryQueue(config)
    elif queue_type == "rabbitmq":
        if RABBITMQ_AVAILABLE:
            return RabbitMQQueue(config)
        else:
            logger.warning("RabbitMQ不可用，使用内存队列")
            return InMemoryQueue(config)
    else:
        logger.warning(f"不支持的队列类型: {queue_type}，使用内存队列")
        return InMemoryQueue(config)


# 导出
__all__ = ["BaseMessageQueue", "InMemoryQueue", "create_message_queue"]

if KAFKA_AVAILABLE:
    __all__.append("KafkaQueue")
if RABBITMQ_AVAILABLE:
    __all__.append("RabbitMQQueue")
