#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo公共工具类
提取配置加载、组件初始化等公共逻辑，简化demo编写
"""

import os
import sys
import yaml
import signal
import asyncio
import argparse
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from loguru import logger

from producer.camera_source import CameraVideoSource
from producer.file_source import FileVideoSource
from message_queue.kafka_queue import KafkaQueue
from core.pipeline import PipelineManager, ProcessingPipeline


class DemoConfig:
    """Demo配置管理器"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"已加载配置文件: {config_path}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
        
        # 应用环境变量覆盖
        DemoConfig._apply_env_overrides(config)
        return config
    
    @staticmethod
    def _apply_env_overrides(config: Dict[str, Any]) -> None:
        """应用环境变量覆盖"""
        if os.getenv("KAFKA_HOST") and os.getenv("KAFKA_PORT") and "queue" in config:
            config["queue"]["config"]["bootstrap_servers"] = [f"{os.getenv('KAFKA_HOST')}:{os.getenv('KAFKA_PORT')}"]
        if os.getenv("KAFKA_TOPIC") and "queue" in config:
            config["queue"]["config"]["topic_name"] = os.getenv("KAFKA_TOPIC")
        
        # 视频源环境变量覆盖
        if os.getenv("CAMERA_INDEX"):
            config.setdefault("video_source", {})["type"] = "camera"
            config["video_source"]["camera_index"] = int(os.getenv("CAMERA_INDEX"))
        if os.getenv("VIDEO_FPS"):
            config.setdefault("video_source", {})["fps"] = float(os.getenv("VIDEO_FPS"))
        if os.getenv("VIDEO_FILE"):
            config.setdefault("video_source", {})["type"] = "file"
            config["video_source"]["file_path"] = os.getenv("VIDEO_FILE")


class DemoInitializer:
    """Demo组件初始化器"""
    
    @staticmethod
    async def init_video_source(config: Dict[str, Any]):
        """
        初始化视频源（摄像头或文件）
        
        Args:
            config: 配置字典，包含video_source配置
            
        Returns:
            视频源对象（CameraVideoSource或FileVideoSource）
        """
        video_source_config = config.get("video_source", {})
        source_type = video_source_config.get("type", "camera")
        
        if source_type == "camera":
            camera_index = video_source_config.get("camera_index", 0)
            fps = video_source_config.get("fps", 1.0)
            
            video_source = CameraVideoSource(
                camera_index=camera_index,
                fps=fps
            )
            await video_source.initialize()
            logger.info(f"摄像头视频源初始化完成: index={camera_index}, fps={fps}")
            return video_source
            
        elif source_type == "file":
            file_path = video_source_config.get("file_path")
            if not file_path:
                raise ValueError("file_path is required for file video source")
            
            fps = video_source_config.get("fps", 1.0)
            loop = video_source_config.get("loop", False)
            
            # 检查文件可用性
            is_available, message = FileVideoSource.check_file_availability(file_path)
            if not is_available:
                raise ValueError(f"Video file check failed: {message}")
            
            video_source = FileVideoSource(
                file_path=file_path,
                fps=fps,
                loop=loop
            )
            await video_source.initialize()
            logger.info(f"文件视频源初始化完成: {message}")
            return video_source
            
        else:
            raise ValueError(f"Unsupported video source type: {source_type}. Supported: ['camera', 'file']")
    
    @staticmethod
    async def init_kafka_queue(config: Dict[str, Any]) -> KafkaQueue:
        """初始化Kafka队列"""
        queue_config = config.get("queue", {})
        if queue_config.get("type") != "kafka":
            raise ValueError(f"不支持的队列类型: {queue_config.get('type')}")
        
        kafka_queue = KafkaQueue(config=queue_config.get("config", {}))
        await kafka_queue.initialize()
        topic_name = queue_config.get("config", {}).get("topic_name", "demo")
        logger.info(f"Kafka队列初始化完成: topic={topic_name}")
        return kafka_queue
    
    @staticmethod
    async def init_processor(processor_name: str, processor_class, config: Dict[str, Any]):
        """初始化单个处理器"""
        processor = processor_class(config=config)
        await processor.initialize()
        logger.info(f"处理器 {processor_name} 初始化完成")
        return processor
    
    @staticmethod
    def create_pipeline(pipeline_id: str) -> tuple[PipelineManager, ProcessingPipeline]:
        """创建流水线管理器和流水线"""
        pipeline_manager = PipelineManager()
        pipeline = pipeline_manager.create_pipeline(pipeline_id)
        return pipeline_manager, pipeline


class DemoLogger:
    """Demo日志配置器"""
    
    @staticmethod
    def setup_logging(demo_name: str = "demo") -> None:
        """设置日志"""
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # 添加文件日志
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / f"{demo_name}.log",
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )


class DemoRunner:
    """Demo运行器"""
    
    def __init__(self, demo_name: str, description: str):
        self.demo_name = demo_name
        self.description = description
        self.components = {}
        self._signal_handlers_set = False
    
    def setup_signal_handlers(self, cleanup_func: Callable):
        """设置信号处理器"""
        if self._signal_handlers_set:
            return
            
        loop = asyncio.get_event_loop()
        
        def signal_handler():
            logger.info("收到关闭信号，正在停止服务...")
            loop.create_task(cleanup_func())
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
        
        self._signal_handlers_set = True
    
    async def run_with_stats(self, main_task_coro, config: Dict[str, Any]):
        """运行主任务并定期打印统计信息"""
        stats_interval = config.get("demo", {}).get("stats_interval", 30)
        
        # 创建主任务
        main_task = asyncio.create_task(main_task_coro)
        
        # 创建统计任务
        stats_task = asyncio.create_task(self._stats_loop(stats_interval))
        
        try:
            # 等待任务完成
            done, pending = await asyncio.wait(
                [main_task, stats_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 取消未完成的任务
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        except Exception as e:
            logger.error(f"运行过程中发生错误: {e}")
            raise
    
    async def _stats_loop(self, interval: int):
        """统计信息循环"""
        import time
        last_stats_time = time.time()
        
        while True:
            current_time = time.time()
            
            if current_time - last_stats_time >= interval:
                try:
                    await self.print_stats()
                    last_stats_time = current_time
                except Exception as e:
                    logger.error(f"获取统计信息失败: {e}")
            
            await asyncio.sleep(1)
    
    async def print_stats(self):
        """打印统计信息，子类可覆盖"""
        # 获取流水线统计
        pipeline_manager = self.components.get("pipeline_manager")
        if pipeline_manager:
            pipeline_status = pipeline_manager.get_pipeline_status()
            if pipeline_status:
                logger.info(f"📊 流水线状态: {pipeline_status}")
        
        # 获取队列统计
        kafka_queue = self.components.get("kafka_queue")
        if kafka_queue and hasattr(kafka_queue, "get_stats"):
            queue_stats = kafka_queue.get_stats()
            logger.info(f"📨 队列统计: {queue_stats}")


def demo_main(demo_func: Callable):
    """Demo主函数装饰器，简化demo的main函数编写"""
    def wrapper():
        try:
            asyncio.run(demo_func())
        except KeyboardInterrupt:
            logger.info("👋 用户取消，退出程序")
        except Exception as e:
            logger.error(f"❌ 运行demo失败: {e}")
            import traceback
            traceback.print_exc()
    
    return wrapper


def parse_demo_args(description: str, default_config: str) -> argparse.Namespace:
    """解析demo命令行参数的通用函数"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, help="配置文件路径", default=default_config)
    parser.add_argument("--video-file", type=str, help="视频文件路径")
    parser.add_argument("--video-fps", type=float, help="视频读取FPS")
    parser.add_argument("--loop", action="store_true", help="循环播放视频文件")
    parser.add_argument("--camera-index", type=int, help="摄像头索引")
    return parser.parse_args()


def apply_args_to_config(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """将命令行参数应用到配置中"""
    # 视频源参数
    if args.video_file is not None:
        config.setdefault("video_source", {})["type"] = "file"
        config["video_source"]["file_path"] = args.video_file
    if args.video_fps is not None:
        config.setdefault("video_source", {})["fps"] = args.video_fps
    if args.loop:
        config.setdefault("video_source", {})["loop"] = True
    if args.camera_index is not None:
        config.setdefault("video_source", {})["type"] = "camera"
        config["video_source"]["camera_index"] = args.camera_index


async def cleanup_components(components: Dict[str, Any], pipeline_id: str = None) -> None:
    """清理组件资源的通用函数"""
    logger.info("正在清理资源...")
    
    # 停止流水线
    pipeline_manager = components.get("pipeline_manager")
    if pipeline_manager and pipeline_id:
        await pipeline_manager.stop_pipeline(pipeline_id)
    
    # 关闭视频源
    video_source = components.get("video_source")
    if video_source:
        await video_source.close()
    
    # 关闭Kafka队列
    kafka_queue = components.get("kafka_queue")
    if kafka_queue:
        await kafka_queue.close()
    
    logger.info("资源清理完成") 