# -*- coding: utf-8 -*-

"""
视频流处理演示程序

该程序实现了以下功能：
1. 捕获摄像头的视频帧
2. 通过相似帧过滤器去除重复帧
3. 使用YOLO检测人员
4. 将检测到人员的帧发送到Kafka队列供视觉大模型处理，大模型处理后发出打招呼消息

用法:
    python demo.py [--config CONFIG_PATH]
    python demo.py --camera-index 0 --camera-fps 1.0

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
from preprocessor.yolo_detector import YOLODetectorProcessor
from message_queue.kafka_queue import KafkaQueue
from core.pipeline import ProcessingPipeline, PipelineManager
from core.filters import create_person_detection_filter


async def initialize_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """初始化所有组件"""
    # 初始化摄像头源
    camera_config = config.get("camera", {})
    camera_source = CameraVideoSource(
        camera_index=camera_config.get("camera_index", 0),
        fps=camera_config.get("fps", 1.0),
        resolution=tuple(camera_config.get("resolution", [640, 480]))
    )
    await camera_source.initialize()
    logger.info(f"摄像头初始化完成: index={camera_config.get('camera_index', 0)}, fps={camera_config.get('fps', 1.0)}")

    # 初始化相似帧过滤器
    similar_frame_filter = SimilarFrameFilterProcessor(config=config.get("similar_frame_filter", {}))
    await similar_frame_filter.initialize()
    logger.info("相似帧过滤器初始化完成")

    # 初始化YOLO检测器
    yolo_config = config.get("preprocessors", {}).get("yolo_detector", {})
    yolo_detector = YOLODetectorProcessor(config=yolo_config)
    await yolo_detector.initialize()
    logger.info(f"YOLO检测器初始化完成: model={yolo_config.get('model_path', 'models/yolo-v8l-seg.pt')}")

    # 初始化Kafka队列
    queue_config = config.get("queue", {})
    kafka_queue = KafkaQueue(config=queue_config)
    await kafka_queue.initialize()
    logger.info(f"Kafka队列初始化完成: topic={queue_config.get('topic_name', 'video_processing')}")

    # 创建流水线
    pipeline_manager = PipelineManager()
    pipeline = pipeline_manager.create_pipeline("camera_detection_pipeline")
    
    # 添加处理器
    pipeline.add_processor(similar_frame_filter)
    pipeline.add_processor(yolo_detector)
    
    # 添加人员检测过滤器
    pipeline.add_filter(create_person_detection_filter(yolo_detector.processor_name))
    
    # 设置后处理器配置（从config.yaml中读取）
    postprocessor_config = config.get("postprocessors", {})
    if postprocessor_config:
        pipeline.set_postprocessor_config(postprocessor_config)
        logger.info(f"设置后处理器配置: {list(postprocessor_config.keys())}")
    
    # 设置输出队列
    pipeline.set_output_queue(kafka_queue)
    
    logger.info("处理流水线创建完成")

    return {
        "camera_source": camera_source,
        "pipeline_manager": pipeline_manager,
        "kafka_queue": kafka_queue
    }


async def run_camera_processor(config: Dict[str, Any]) -> None:
    """运行摄像头处理主流程"""
    components = await initialize_components(config)
    camera_source = components["camera_source"]
    pipeline_manager: PipelineManager = components["pipeline_manager"]
    
    # 设置信号处理，优雅关闭
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("收到关闭信号，正在停止服务...")
        loop.create_task(cleanup(components))
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # 启动流水线
        await pipeline_manager.start_pipeline("camera_detection_pipeline", camera_source)
        logger.info("开始处理摄像头视频流")
        
        # 保持程序运行
        while True:
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
        await pipeline_manager.stop_pipeline("camera_detection_pipeline")
    
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
    # 默认配置
    default_config = {
        "camera": {
            "camera_index": 0,
            "fps": 1.0,
            "resolution": [640, 480]
        },
        "similar_frame_filter": {
            "similarity_threshold": 0.8,
            "comparison_method": "clip",
            "clip_model_path": "models/clip-vit-base-patch32",
            "history_size": 5,
            "min_time_interval": 0.5
        },
        "preprocessors": {
            "yolo_detector": {
                "model_path": "models/yolo-v8l-seg.pt",
                "device": "cpu",
                "confidence_threshold": 0.5,
                "target_classes": ["person"],
                "enable_downstream": True,
                "task_configs": {
                    "person_detection": {
                        "system_prompt": """你是一个友好的智能助手，当检测到有人出现时，需要用温暖友好的语气向他们打招呼。

请根据图像中人员的情况（人数、位置等），生成合适的问候语。
保持语气自然、友好、热情。""",
                        "user_prompt": "你好！我看到有人出现在画面中，请向他们打个招呼吧。",
                        "task_type": "person_detection"
                    },
                    "general_analysis": {
                        "system_prompt": """你是一个专业的视觉分析专家。请对图像进行全面的分析和描述。

请重点关注：
1. 场景的整体描述
2. 重要物体和人员
3. 环境和氛围
4. 值得注意的细节""",
                        "user_prompt": "请对这张图像进行全面分析，描述你观察到的内容。",
                        "task_type": "general_analysis"
                    }
                }
            }
        },
        "queue": {
            "bootstrap_servers": ["localhost:9092"],
            "topic_name": "demo",
            "consumer_group": "video_processors",
            "use_kafka": True,
            "max_request_size": 10485760,
            "timeout_default": 30.0,
            "serialize_messages": True
        }
    }
    
    # 如果指定了配置文件，则加载并合并
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                # 简单的字典合并
                for key, value in file_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            logger.info(f"已加载配置文件: {config_path}")
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {e}")
    
    # 环境变量覆盖
    if os.getenv("CAMERA_INDEX"):
        default_config["camera"]["camera_index"] = int(os.getenv("CAMERA_INDEX"))
    if os.getenv("CAMERA_FPS"):
        default_config["camera"]["fps"] = float(os.getenv("CAMERA_FPS"))
    if os.getenv("KAFKA_HOST") and os.getenv("KAFKA_PORT"):
        default_config["queue"]["bootstrap_servers"] = [f"{os.getenv('KAFKA_HOST')}:{os.getenv('KAFKA_PORT')}"]
    if os.getenv("KAFKA_TOPIC"):
        default_config["queue"]["topic_name"] = os.getenv("KAFKA_TOPIC")
    
    return default_config


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
        log_dir / "demo.log",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频流处理演示程序")
    parser.add_argument("--config", type=str, help="配置文件路径")
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
    
    logger.info("启动视频流处理演示程序")
    logger.info(f"配置: {config}")
    
    # 运行主程序
    await run_camera_processor(config)


if __name__ == "__main__":
    asyncio.run(main()) 