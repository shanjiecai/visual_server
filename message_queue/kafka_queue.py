import asyncio
import time
import uuid
import json
from collections import deque
from typing import Dict, Any, Optional, List
from loguru import logger

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("aiokafka not available, using in-memory simulation")

from message_queue.base import BaseMessageQueue


class KafkaQueue(BaseMessageQueue):
    """Kafka队列实现 - 支持真实的Kafka消息传递"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._producer = None
        self._consumer = None
        self._message_buffer = deque()  # 备用内存队列
        
        # 配置参数
        self.bootstrap_servers = config.get("bootstrap_servers", ["localhost:9092"])
        self.topic_name = config.get("topic_name", "video_processing")
        self.consumer_group = config.get("consumer_group", "video_processors")
        self.partition = config.get("partition", 0)
        
        # 序列化设置
        self.value_serializer = self._serialize_message
        self.value_deserializer = self._deserialize_message
        
        # 是否使用Kafka（如果不可用则使用内存队列）
        self.use_kafka = KAFKA_AVAILABLE and config.get("use_kafka", True)
    
    async def _do_initialize(self) -> bool:
        """初始化Kafka连接"""
        try:
            if not self.use_kafka:
                logger.info("Using in-memory queue instead of Kafka")
                return True
            
            logger.info(f"Connecting to Kafka at {self.bootstrap_servers}")
            
            # 初始化生产者
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=self.value_serializer,
                compression_type="gzip",  # 压缩大图像数据
                max_request_size=10 * 1024 * 1024,  # 10MB max message size
                request_timeout_ms=30000,
                retry_backoff_ms=1000
            )
            await self._producer.start()
            
            # 初始化消费者
            self._consumer = AIOKafkaConsumer(
                self.topic_name,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=self.value_deserializer,
                auto_offset_reset='latest',  # 从最新消息开始消费
                enable_auto_commit=True,
                consumer_timeout_ms=1000
            )
            await self._consumer.start()
            
            logger.info(f"Kafka topic '{self.topic_name}' initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            logger.info("Falling back to in-memory queue")
            self.use_kafka = False
            return True  # 仍然返回True，使用内存队列作为备用
    
    def _serialize_message(self, message: Any) -> bytes:
        """序列化消息为JSON字节"""
        try:
            if isinstance(message, dict):
                # 确保消息格式正确
                return json.dumps(message, ensure_ascii=False).encode('utf-8')
            else:
                # 兼容其他格式
                return json.dumps({"data": message}, ensure_ascii=False).encode('utf-8')
        except Exception as e:
            logger.error(f"Error serializing message: {e}")
            return json.dumps({"error": "serialization_failed"}).encode('utf-8')
    
    def _deserialize_message(self, message: bytes) -> Any:
        """反序列化JSON字节为消息"""
        try:
            return json.loads(message.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error deserializing message: {e}")
            return {"error": "deserialization_failed"}
    
    async def _do_put(self, item: Any, timeout: float) -> bool:
        """向Kafka主题发送消息"""
        try:
            if not self.use_kafka:
                # 使用内存队列
                self._message_buffer.append((0, time.time(), item))
                logger.debug(f"Added message to in-memory buffer (size: {len(self._message_buffer)})")
                return True
            
            if not self._producer:
                logger.error("Kafka producer not initialized")
                return False
            
            # 构建消息格式
            message_data = self._build_message(item, 0)
            
            # 发送消息到Kafka
            await self._producer.send_and_wait(
                self.topic_name,
                value=message_data,
                partition=self.partition
            )
            
            logger.debug(f"Published message to Kafka topic '{self.topic_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {e}")
            # 备用到内存队列
            try:
                self._message_buffer.append((0, time.time(), item))
                logger.debug("Message added to backup in-memory buffer")
                return True
            except Exception as backup_error:
                logger.error(f"Backup queue also failed: {backup_error}")
                return False
    
    def _build_message(self, item: Any, priority: int) -> Dict[str, Any]:
        """构建标准消息格式"""
        message = {
            "message_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "priority": priority,
            "data": item
        }
        
        # 如果item已经是包含metadata的字典，保持结构
        if isinstance(item, dict):
            if "metadata" in item:
                message["metadata"] = item["metadata"]
            if "prompt" in item:
                message["prompt"] = item["prompt"]
            if "system_prompt" in item:
                message["system_prompt"] = item["system_prompt"]
            if "user_prompt" in item:
                message["user_prompt"] = item["user_prompt"]
            if "task_type" in item:
                message["task_type"] = item["task_type"]
            if "image_base64" in item:
                message["image_base64"] = item["image_base64"]
        
        return message
    
    async def _do_get(self, timeout: float) -> Any:
        """从Kafka主题消费消息"""
        try:
            if not self.use_kafka:
                # 使用内存队列
                if self._message_buffer:
                    # 按优先级排序
                    self._message_buffer = deque(sorted(self._message_buffer, key=lambda x: x[0]))
                    priority, timestamp, item = self._message_buffer.popleft()
                    logger.debug(f"Retrieved message from in-memory buffer (remaining: {len(self._message_buffer)})")
                    return item
                else:
                    await asyncio.sleep(min(timeout, 0.1))
                    return None
            
            if not self._consumer:
                logger.error("Kafka consumer not initialized")
                return None
            
            # 消费消息
            try:
                # 使用异步超时
                message = await asyncio.wait_for(
                    self._consumer.getone(),
                    timeout=timeout
                )
                
                if message:
                    # 解析消息
                    message_data = message.value
                    logger.debug(f"Consumed message from Kafka topic '{self.topic_name}'")
                    
                    # 返回原始数据部分
                    if isinstance(message_data, dict) and "data" in message_data:
                        return message_data["data"]
                    else:
                        return message_data
                        
            except asyncio.TimeoutError:
                logger.debug(f"Kafka consumer timeout after {timeout}s")
                return None
            
        except Exception as e:
            logger.error(f"Error consuming from Kafka: {e}")
            return None
    
    def size(self) -> int:
        """获取队列大小"""
        if not self.use_kafka:
            return len(self._message_buffer)
        
        # Kafka主题消息数量查询需要额外的API调用
        # 这里返回0，实际应用中可以通过Kafka Admin API查询
        return 0
    
    async def _do_close(self) -> None:
        """关闭Kafka连接的具体实现"""
        try:
            if self._consumer:
                await self._consumer.stop()
                self._consumer = None
                logger.info("Kafka consumer closed")
            
            if self._producer:
                await self._producer.stop()
                self._producer = None
                logger.info("Kafka producer closed")
            
            logger.info("Kafka connections closed")
        except Exception as e:
            logger.error(f"Error closing Kafka connections: {e}")

    async def put_batch(self, items: List[Any]) -> bool:
        """批量添加项目到队列"""
        try:
            success = True
            for item in items:
                result = await self._do_put(item, self.timeout)
                if not result:
                    success = False
            return success
        except Exception as e:
            logger.error(f"Error in batch put: {e}")
            return False
    
    async def get_batch(self, batch_size: int, timeout: Optional[float] = None) -> List[Any]:
        """批量获取项目"""
        results = []
        timeout = timeout or self.timeout
        
        try:
            # 尝试获取指定批次大小的数据
            start_time = time.time()
            remain_time = timeout
            
            while len(results) < batch_size and remain_time > 0:
                item = await self._do_get(remain_time)
                if item:
                    results.append(item)
                # 更新剩余时间
                remain_time = timeout - (time.time() - start_time)
            
            return results
        except Exception as e:
            logger.error(f"Error in batch get: {e}")
            return results
    
    async def close(self) -> None:
        """关闭Kafka连接"""
        try:
            logger.info(f"Closing message queue: {self.queue_name}")
            await self._do_close()
            self._is_connected = False
            logger.info(f"Message queue {self.queue_name} closed")
        except Exception as e:
            logger.error(f"Error closing message queue {self.queue_name}: {e}")
            
    async def get_message_with_metadata(self, timeout: float) -> Optional[Dict[str, Any]]:
        """获取包含完整metadata的消息"""
        try:
            if not self.use_kafka:
                # 使用内存队列
                if self._message_buffer:
                    self._message_buffer = deque(sorted(self._message_buffer, key=lambda x: x[0]))
                    priority, timestamp, item = self._message_buffer.popleft()
                    return {
                        "data": item,
                        "priority": priority,
                        "timestamp": timestamp,
                        "source": "memory_queue"
                    }
                else:
                    await asyncio.sleep(min(timeout, 0.1))
                    return None
            
            if not self._consumer:
                return None
            
            try:
                message = await asyncio.wait_for(
                    self._consumer.getone(),
                    timeout=timeout
                )
                
                if message:
                    return message.value  # 返回完整的消息数据
                    
            except asyncio.TimeoutError:
                return None
                
        except Exception as e:
            logger.error(f"Error getting message with metadata: {e}")
            return None 