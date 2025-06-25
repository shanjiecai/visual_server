#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视觉记忆提取演示程序

该程序实现了以下功能：
1. 捕获摄像头的视频帧
2. 通过相似帧过滤器去除重复帧
3. 将过滤后的帧发送到Kafka队列，并携带记忆提取任务配置
4. VLM Worker消费队列进行记忆提取和存储
5. 提供记忆查询API服务

架构流程：
摄像头 -> 相似帧过滤 -> 队列(携带任务配置) -> VLM Worker(记忆提取) -> 记忆存储 -> API查询

用法:
    python demo_memory.py [--config CONFIG_PATH]
    python demo_memory.py --camera-index 0 --camera-fps 1.0

作者：Visual Processing Team
"""

import asyncio
import argparse
import os
import signal
import sys
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from producer.camera_source import CameraVideoSource
from preprocessor.similar_frame_filter import SimilarFrameFilterProcessor
from message_queue.kafka_queue import KafkaQueue
from core.pipeline import PipelineManager
from app.memory_router import create_memory_api_service


async def initialize_memory_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """初始化记忆提取相关组件"""
    # 初始化摄像头源
    camera_config = config.get("camera", {})
    camera_source = CameraVideoSource(
        camera_index=camera_config.get("camera_index", 0),
        fps=camera_config.get("fps", 1.0),
    )
    await camera_source.initialize()
    logger.info(f"摄像头初始化完成: index={camera_config.get('camera_index', 0)}, fps={camera_config.get('fps', 1.0)}")

    # 初始化相似帧过滤器
    similar_frame_config = config.get("preprocessors", {}).get("similar_frame_filter", {})
    similar_frame_filter = SimilarFrameFilterProcessor(config=similar_frame_config)
    await similar_frame_filter.initialize()
    logger.info("相似帧过滤器初始化完成")

    # 初始化Kafka队列
    queue = config.get("queue", {})
    if queue.get("type") == "kafka":
        kafka_queue = KafkaQueue(config=queue.get("config", {}))
        await kafka_queue.initialize()
        logger.info(f"Kafka队列初始化完成: topic={queue.get('config', {}).get('topic_name', 'visual_memory')}")
    else:
        raise ValueError(f"不支持的队列类型: {queue.get('type')}")

    # 创建流水线
    pipeline_manager = PipelineManager()
    pipeline = pipeline_manager.create_pipeline("memory_pipeline")
    
    # 添加相似帧过滤器
    pipeline.add_processor(similar_frame_filter)
    
    # 设置输出队列
    pipeline.set_output_queue(kafka_queue)
    
    # 设置任务配置（传递给队列消息）
    task_config = _create_memory_task_config(config)
    pipeline.set_postprocessor_config(task_config)
    
    logger.info("记忆提取流水线创建完成")

    # 初始化记忆API服务（仅用于配置）
    memory_api_service = None
    if config.get("memory_api", {}).get("enabled", False):
        api_config = config.get("memory_api", {})
        memory_api_service = create_memory_api_service(api_config)
        await memory_api_service.initialize()
        logger.info("记忆查询API服务配置完成")

    return {
        "camera_source": camera_source,
        "pipeline_manager": pipeline_manager,
        "kafka_queue": kafka_queue,
        "memory_api_service": memory_api_service
    }


def _create_memory_task_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """创建记忆提取任务配置"""
    # 从配置文件中获取VLM任务配置
    vlm_tasks = config.get("vlm_tasks", {})
    memory_config = config.get("memory_extraction", {})
    
    # 如果配置文件中有vlm_tasks，直接使用
    if vlm_tasks:
        # 选择记忆检测任务配置
        task_config = vlm_tasks.get("memory_detection", {})
        
        # 如果没有memory_detection任务，使用第一个可用的任务
        if not task_config and vlm_tasks:
            first_task_key = list(vlm_tasks.keys())[0]
            task_config = vlm_tasks[first_task_key]
            logger.warning(f"未找到memory_detection任务配置，使用 {first_task_key}")
        
        vlm_task_config = {
            "memory_detection": task_config
        }
    else:
        # 兜底：使用原有的配置方式
        vlm_task_config = {
            "memory_detection": {
                "task_type": "memory_detection",
                "system_prompt": memory_config.get("detection_system_prompt", _get_default_detection_prompt(memory_config)),
                "user_prompt": "请识别图像中的所有物体类别，只返回类别名称列表，用中文逗号分隔。",
                "vlm_config": {
                    "model": "Qwen2.5-VL-72B-Instruct-AWQ",
                    "max_tokens": memory_config.get("vlm_max_tokens", 64),
                    "temperature": 0.1,
                    "base_url": "http://cc.komect.com/llm/vlgroup/",
                    "api_key": "EMPTY"
                }
            }
        }
    
    # 后处理器配置，包含记忆存储
    postprocessor_config = {
        "memory_storage": {
            "enabled": True,
            "processor_type": "memory_storage",
            "memory_storage": memory_config.get("memory_storage", {}),
            "target_objects": memory_config.get("target_objects", [
                "手机", "桌子", "电脑", "笔", "水杯", "地板", "椅子", "花", "人"
            ])
        }
    }
    
    return {
        "vlm_task_config": vlm_task_config,
        "postprocessor_config": postprocessor_config
    }


def _get_default_detection_prompt(config: Dict[str, Any]) -> str:
    """获取默认的物体检测提示词"""
    target_objects = config.get("target_objects", [
        "手机", "桌子", "电脑", "笔", "水杯", "地板", "椅子", "花", "人"
    ])
    target_objects_str = ", ".join(target_objects)
    
    return f"""你是一位专业的计算机视觉专家，擅长目标检测和物体识别。请对提供的图像进行全面的目标检测，识别出图像中的所有物体。

输出类别仅限于以下物体类别中的一种或多种：{target_objects_str}

要求：
1. 只返回在图像中真实存在的物体类别
2. 用中文逗号分隔多个类别
3. 不要添加任何解释或描述
4. 不允许出现不在指定类别列表中的物体名称

示例输出格式：水杯,桌子,手机"""


async def run_memory_demo(config: Dict[str, Any]) -> None:
    """运行记忆提取主流程"""
    components = await initialize_memory_components(config)
    camera_source = components["camera_source"]
    pipeline_manager: PipelineManager = components["pipeline_manager"]
    memory_api_service = components.get("memory_api_service")
    
    # 设置信号处理，优雅关闭
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("收到关闭信号，正在停止服务...")
        loop.create_task(cleanup(components))
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # 提示用户启动API服务
        if config.get("memory_api", {}).get("enabled", False):
            logger.info("="*80)
            logger.info("🌐 记忆查询API服务说明:")
            logger.info("   记忆API已集成到主FastAPI应用中")
            logger.info("   请单独启动API服务:")
            logger.info("   python server.py")
            logger.info("   然后访问: http://localhost:8000/memory/stats")
            logger.info("   API文档: http://localhost:8000/docs")
            logger.info("="*80)
        
        # 启动流水线
        await pipeline_manager.start_pipeline("memory_pipeline", camera_source)
        logger.info("开始处理摄像头视频流 -> 相似帧过滤 -> 队列(携带任务配置) -> VLM Worker")
        
        # 定期打印统计信息
        stats_interval = config.get("demo", {}).get("stats_interval", 30)
        last_stats_time = time.time()
        
        # 保持程序运行
        while True:
            current_time = time.time()
            
            # 定期打印统计信息
            if current_time - last_stats_time >= stats_interval:
                try:
                    # 获取流水线统计
                    pipeline_status = pipeline_manager.get_pipeline_status()
                    if pipeline_status:
                        logger.info(f"流水线状态: {pipeline_status}")
                    
                    # 获取队列统计
                    if hasattr(components["kafka_queue"], "get_stats"):
                        queue_stats = components["kafka_queue"].get_stats()
                        logger.info(f"队列统计: {queue_stats}")
                    
                    # 获取记忆统计（如果API服务可用）
                    if memory_api_service and hasattr(memory_api_service, "get_memory_storage"):
                        try:
                            memory_storage = memory_api_service.get_memory_storage()
                            if memory_storage and hasattr(memory_storage, "get_memory_stats"):
                                memory_stats = memory_storage.get_memory_stats()
                                logger.info(f"记忆统计: {memory_stats}")
                        except Exception as e:
                            logger.debug(f"获取记忆统计失败: {e}")
                    
                    last_stats_time = current_time
                    
                except Exception as e:
                    logger.error(f"获取统计信息失败: {e}")
            
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        await cleanup(components)


async def cleanup(components: Dict[str, Any]) -> None:
    """清理所有组件资源"""
    logger.info("正在清理资源...")
    
    # 停止流水线
    pipeline_manager = components.get("pipeline_manager")
    if pipeline_manager:
        await pipeline_manager.stop_pipeline("memory_pipeline")
    
    # 关闭摄像头
    camera_source = components.get("camera_source")
    if camera_source:
        await camera_source.close()
    
    # 关闭Kafka队列
    kafka_queue = components.get("kafka_queue")
    if kafka_queue:
        await kafka_queue.close()
    
    logger.info("资源清理完成")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件"""
    if not config_path:
        raise ValueError("必须指定配置文件路径")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置文件: {config_path}")
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")
    
    # 环境变量覆盖
    if os.getenv("CAMERA_INDEX") and "camera" in config:
        config["camera"]["camera_index"] = int(os.getenv("CAMERA_INDEX"))
    if os.getenv("CAMERA_FPS") and "camera" in config:
        config["camera"]["fps"] = float(os.getenv("CAMERA_FPS"))
    if os.getenv("KAFKA_HOST") and os.getenv("KAFKA_PORT") and "queue" in config:
        config["queue"]["config"]["bootstrap_servers"] = [f"{os.getenv('KAFKA_HOST')}:{os.getenv('KAFKA_PORT')}"]
    if os.getenv("KAFKA_TOPIC") and "queue" in config:
        config["queue"]["config"]["topic_name"] = os.getenv("KAFKA_TOPIC")
    
    return config


def setup_logging() -> None:
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
        log_dir / "memory_demo.log",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视觉记忆提取演示程序")
    parser.add_argument("--config", type=str, help="配置文件路径", default="./configs/memory_demo_config.yaml")
    parser.add_argument("--camera-index", type=int, help="摄像头索引")
    parser.add_argument("--camera-fps", type=float, help="摄像头FPS")
    return parser.parse_args()


async def main() -> None:
    """主函数"""
    setup_logging()
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.camera_index is not None:
        config["camera"]["camera_index"] = args.camera_index
    if args.camera_fps is not None:
        config["camera"]["fps"] = args.camera_fps
    
    logger.info("启动视觉记忆提取演示程序")
    logger.info("架构: 摄像头 -> 相似帧过滤 -> Kafka队列(携带任务配置) -> VLM Worker(记忆提取)")
    logger.info(f"配置: {config}")
    
    # 运行主程序
    await run_memory_demo(config)


if __name__ == "__main__":
    
    asyncio.run(main())
