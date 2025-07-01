#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视觉大模型工作进程
从消息队列中消费图像数据，调用视觉大模型处理
"""

import asyncio
import json
import os
import time
import uuid
from typing import Dict, Any, Optional, List
from loguru import logger
import yaml
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from message_queue.kafka_queue import KafkaQueue
from consumer.openai_vlm import OpenAIVLMProcessor
from core.interfaces import ProcessingTask, FrameData, ProcessingStatus


class VLMWorker:
    """视觉大模型工作进程"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化工作进程"""
        self.worker_id = config.get("worker", {}).get("worker_id", f"vlm_worker_{uuid.uuid4().hex[:8]}")
        self.config = config
        self.running = False
        self.queue = None
        self.vlm_processor = None
        self.poll_interval = config.get("worker", {}).get("poll_interval", 1)  # 轮询间隔（秒）
        self.batch_size = config.get("worker", {}).get("batch_size", 1)  # 批处理大小
        self.max_retries = config.get("worker", {}).get("max_retries", 3)  # 最大重试次数
    
    async def initialize(self) -> bool:
        """初始化队列和处理器"""
        try:
            # 初始化队列
            queue_config = self.config.get("queue_config", {})
            self.queue = KafkaQueue(queue_config)
            if not await self.queue.initialize():
                logger.error("Failed to initialize message queue")
                return False
            
            # 初始化视觉大模型处理器
            vlm_config = self.config.get("vlm_config", {})
            vlm_config["processor_name"] = "vlm_processor"
            self.vlm_processor = OpenAIVLMProcessor(vlm_config)
            if not await self.vlm_processor.initialize():
                logger.error("Failed to initialize VLM processor")
                return False
            
            logger.info(f"VLM Worker {self.worker_id} initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing VLM Worker: {e}")
            return False
    
    async def start(self) -> None:
        """启动工作进程"""
        if self.running:
            return
        
        self.running = True
        logger.info(f"Starting VLM Worker {self.worker_id}")
        
        try:
            while self.running:
                try:
                    # 从队列中获取消息
                    message = await self.queue.get_message_with_metadata(timeout=self.poll_interval)
                    
                    if message:
                        await self._process_message(message)
                    else:
                        # 队列为空，等待一小段时间
                        await asyncio.sleep(self.poll_interval)
                        
                except asyncio.CancelledError:
                    logger.info(f"VLM Worker {self.worker_id} was cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await asyncio.sleep(1)  # 出错后短暂等待
        
        except Exception as e:
            logger.error(f"VLM Worker {self.worker_id} encountered an error: {e}")
        finally:
            self.running = False
            logger.info(f"VLM Worker {self.worker_id} stopped")
    
    async def stop(self) -> None:
        """停止工作进程"""
        self.running = False
        if self.queue:
            await self.queue.close()
        
        logger.info(f"VLM Worker {self.worker_id} resources released")
    
    async def _process_message(self, message: Dict[str, Any]) -> None:
        """处理队列消息"""
        try:
            # 记录接收到的消息
            frame_id = message.get("frame_id") or message.get("data", {}).get("frame_id")
            logger.info(f"Received message with frame_id: {frame_id}")
            formatted_msg = self._format_message_for_log(message)
            logger.info(f"Message content: {formatted_msg}")
            
            # 检查必要的字段
            if not self._validate_message(message):
                logger.warning(f"Invalid message format: {message.keys()}")
                return
            
            # 检查图像数据
            data = message.get("data", message)
            image_check_result = self._check_image_data(data)
            logger.info(f"Image data check for frame {frame_id}: {image_check_result}")
            
            # 创建处理任务
            task = self._create_processing_task(message)
            
            # 调用视觉大模型处理
            start_time = time.time()
            result = await self.vlm_processor.process_task(task)
            processing_time = time.time() - start_time
            
            # 处理结果
            if result:
                logger.info(f"Processed frame {frame_id} in {processing_time:.2f}s")
                
                # 打印大模型输出
                content = result.result_data.get("content", "")
                logger.info(f"VLM Output for frame {frame_id}:")
                logger.info(f"Content: {content}")
                
                # 将结果添加到任务中
                task.processing_results.append(result)
                
                # 从消息中获取后处理器配置并执行
                postprocessor_config = message.get("data", {}).get("postprocessor_config", {})
                if postprocessor_config:
                    await self._execute_dynamic_postprocessors(task, postprocessor_config)
                else:
                    logger.info("No postprocessor configuration found in message")
            else:
                logger.warning(f"Failed to process frame {frame_id}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _execute_dynamic_postprocessors(self, task: ProcessingTask, postprocessor_config: Dict[str, Any]) -> None:
        """动态执行后处理器"""
        try:
            logger.info(f"Executing dynamic postprocessors: {list(postprocessor_config.keys())}")
            
            for name, config in postprocessor_config.items():
                if config.get("enabled", False):
                    try:
                        # 动态创建并执行后处理器
                        postprocessor = self._create_postprocessor(name, config)
                        if postprocessor:
                            # 初始化后处理器
                            if await postprocessor.initialize():
                                logger.info(f"Executing postprocessor: {name}")
                                result = await postprocessor.execute(task)
                                logger.info(f"Postprocessor {name} result: {result.get('status', 'unknown')}")
                                
                                # 清理后处理器资源
                                await postprocessor.cleanup()
                            else:
                                logger.warning(f"Failed to initialize postprocessor: {name}")
                        else:
                            logger.warning(f"Failed to create postprocessor: {name}")
                    except Exception as e:
                        logger.error(f"Error executing postprocessor {name}: {e}")
                else:
                    logger.debug(f"Postprocessor {name} is disabled")
                    
        except Exception as e:
            logger.error(f"Error in dynamic postprocessor execution: {e}")
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """验证消息格式"""
        # 检查是否有图像数据
        data = message.get("data", message)
        if "image_base64" not in data:
            logger.warning(f"Missing required field: image_base64")
            return False
        
        # 如果没有frame_id，生成一个随机ID
        if "frame_id" not in data and "frame_id" not in message:
            generated_id = f"generated_{uuid.uuid4().hex[:8]}"
            data["frame_id"] = generated_id
            message["frame_id"] = generated_id
            logger.info(f"Generated frame_id: {generated_id}")
        
        return True
    
    def _create_processing_task(self, message: Dict[str, Any]) -> ProcessingTask:
        """创建处理任务"""
        # 从消息中提取数据
        data = message.get("data", message)  # 兼容直接传递数据的情况
        
        # 创建FrameData对象
        frame_data = FrameData(
            frame_id=data.get("frame_id", message.get("frame_id", str(uuid.uuid4()))),
            timestamp=data.get("timestamp", message.get("timestamp", time.time())),
            raw_data=data,  # 将整个数据作为raw_data传递
            metadata=data.get("metadata", {})
        )
        
        # 创建ProcessingTask对象
        task = ProcessingTask(
            task_id=message.get("message_id", str(uuid.uuid4())),
            frame_data=frame_data,
            processing_results=[],
            status=ProcessingStatus.PENDING,
            created_at=message.get("timestamp", time.time()),
            updated_at=time.time()
        )
        
        return task

    def _create_postprocessor(self, name: str, config: Dict[str, Any]):
        """创建后处理器实例"""
        try:
            if name == "dialogue_initiator":
                from postprocessor.dialogue_initiator import DialogueInitiatorPostprocessor
                return DialogueInitiatorPostprocessor(config)
            elif name == "notification_sender":
                from postprocessor.notification_sender import NotificationSenderPostprocessor
                return NotificationSenderPostprocessor(config)
            elif name == "greeting_printer":
                from postprocessor.greeting_printer import GreetingPrinterPostprocessor
                return GreetingPrinterPostprocessor(config)
            elif name == "memory_storage":
                from postprocessor.memory_storage import MemoryStoragePostprocessor
                return MemoryStoragePostprocessor(config)
            else:
                logger.warning(f"Unknown postprocessor type: {name}")
                return None
        except ImportError as e:
            logger.error(f"Failed to import postprocessor {name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create postprocessor {name}: {e}")
            return None
        
    def _format_message_for_log(self, message, max_len=60):
        """格式化消息用于日志输出，递归处理嵌套结构"""
        
        # 需要简化的字段名称（支持嵌套路径）
        sensitive_fields = {
            "image_base64", "images_base64", "image_data", "base64_image",
            "raw_data", "encoded_image", "photo_data"
        }
        
        def shorten_value(key, value, path=""):
            """递归简化值，支持嵌套路径检查"""
            current_path = f"{path}.{key}" if path else key
            
            # 检查当前字段是否是敏感字段
            is_sensitive = key in sensitive_fields or any(
                field in current_path for field in sensitive_fields
            )
            
            if isinstance(value, str):
                # 字符串类型：如果是敏感字段或长度超限，则简化
                if is_sensitive or len(value) > max_len:
                    return f"<{type(value).__name__}:{len(value)} chars>"
                return value
                
            elif isinstance(value, dict):
                # 字典类型：递归处理每个键值对
                return {
                    k: shorten_value(k, v, current_path) 
                    for k, v in value.items()
                }
                
            elif isinstance(value, list):
                # 列表类型：递归处理每个元素
                if is_sensitive:
                    # 敏感字段的列表，显示概要信息
                    return f"<list:{len(value)} items>"
                return [
                    shorten_value(f"[{i}]", item, current_path) 
                    for i, item in enumerate(value)
                ]
                
            elif isinstance(value, bytes):
                # 字节类型：显示长度信息
                return f"<bytes:{len(value)} bytes>"
                
            elif hasattr(value, '__dict__'):
                # 对象类型：显示类型和主要属性
                return f"<{type(value).__name__} object>"
                
            else:
                # 其他类型：直接返回
                return value
        
        # 处理输入消息
        if isinstance(message, dict):
            return {k: shorten_value(k, v) for k, v in message.items()}
        elif isinstance(message, list):
            return [shorten_value(f"[{i}]", item) for i, item in enumerate(message)]
        else:
            return shorten_value("value", message)

    def _check_image_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查图像数据的有效性"""
        try:
            from utils.image_utils import validate_image_data, extract_image_from_data
            
            check_result = {
                "has_image_base64": False,
                "has_raw_image": False,
                "data_type": str(type(data)),
                "data_keys": [],
                "image_validation": None
            }
            
            if isinstance(data, dict):
                check_result["data_keys"] = list(data.keys())
                
                # 检查是否有base64字段
                base64_fields = ["image_base64", "images_base64", "image_data", "base64_image"]
                for field in base64_fields:
                    if field in data and data[field]:
                        check_result["has_image_base64"] = True
                        break
                
                # 尝试提取图像数据
                image = extract_image_from_data(data)
                if image is not None:
                    check_result["has_raw_image"] = True
                    check_result["image_shape"] = str(image.shape)
                
                # 使用验证函数
                validation_result = validate_image_data(data)
                check_result["image_validation"] = validation_result
            
            return check_result
            
        except Exception as e:
            return {
                "error": str(e),
                "data_type": str(type(data)),
                "has_image_base64": False,
                "has_raw_image": False
            }


async def run_worker(config_path: str = None):
    """运行工作进程"""
    # 加载VLM worker配置
    config = {}
    
    # 默认配置路径
    if config_path is None:
        config_path = "worker/vlm_worker_config.yaml"
    
    # 从YAML文件加载配置
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"从配置文件加载配置: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    else:
        logger.warning(f"配置文件不存在: {config_path}，将使用默认配置和环境变量")
    
    # 设置日志级别
    log_config = config.get("logging", {})
    log_level = os.environ.get("LOG_LEVEL", log_config.get("level", "INFO"))
    setup_logging(log_level, log_config)
    
    logger.info(f"工作进程配置: {config['worker']['worker_id']}")
    logger.info(f"队列主题: {config['queue_config']['topic_name']}")
    logger.info(f"VLM模型: {config['vlm_config']['model_name']}")
    logger.info(f"Kafka Bootstrap Servers: {config['queue_config']['bootstrap_servers']}")

    logger.info("后处理器将从消息中动态加载")
    
    # 创建并启动工作进程
    worker = VLMWorker(config)
    if await worker.initialize():
        try:
            await worker.start()
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
        finally:
            await worker.stop()
    else:
        logger.error("Failed to initialize worker")


def setup_logging(log_level: str, log_config: dict = None):
    """设置日志配置"""
    if log_config is None:
        log_config = {}
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    if log_config.get("console_output", True):
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}:{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
    
    # 添加文件处理器
    log_file = log_config.get("file_path", "logs/vlm_worker.log")
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=log_config.get("max_file_size", "10 MB"),
            retention=log_config.get("backup_count", 5)
        )
        
        logger.info(f"日志将保存到: {log_file}")


if __name__ == "__main__":
    logger.info("Starting VLM Worker")
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="视觉大模型工作进程")
    parser.add_argument("--config", type=str, help="VLM worker配置文件路径", default="worker/vlm_worker_config.yaml")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_worker(args.config))
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        import traceback
        logger.error(traceback.format_exc()) 