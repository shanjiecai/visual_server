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
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from producer.camera_source import CameraVideoSource
from preprocessor.similar_frame_filter import SimilarFrameFilterProcessor
from preprocessor.yolo_detector import YOLODetectorProcessor
from message_queue.kafka_queue import KafkaQueue
from core.pipeline import PipelineManager
from core.filters import create_person_detection_filter


async def initialize_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """初始化所有组件"""
    # 初始化摄像头源
    camera_config = config.get("camera", {})
    camera_source = CameraVideoSource(
        camera_index=camera_config.get("camera_index", 0),
        fps=camera_config.get("fps", 1.0),
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
    queue = config.get("queue", {})
    if queue.get("type") == "kafka":
        kafka_queue = KafkaQueue(config=queue.get("config", {}))
        await kafka_queue.initialize()
        logger.info(f"Kafka队列初始化完成: topic={queue.get('config', {}).get('topic_name', 'video_processing')}")
    else:
        raise ValueError(f"不支持的队列类型: {queue.get('type')}")

    # 创建流水线
    pipeline_manager = PipelineManager()
    pipeline = pipeline_manager.create_pipeline("camera_detection_pipeline")
    
    # 添加处理器
    pipeline.add_processor(similar_frame_filter)
    pipeline.add_processor(yolo_detector)
    
    # 添加人员检测过滤器
    pipeline.add_filter(create_person_detection_filter(yolo_detector.processor_name))
    
    # 设置独立的VLM任务配置和后处理器配置
    vlm_and_postprocessor_config = {
        "vlm_task_config": config.get("vlm_task_config", {}),
        "postprocessor_config": config.get("postprocessors", {})
    }
    pipeline.set_postprocessor_config(vlm_and_postprocessor_config)
    logger.info("设置独立的VLM任务配置和后处理器配置")
    
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
        log_dir / "demo.log",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频流处理演示程序")
    parser.add_argument("--config", type=str, help="配置文件路径", default="./configs/demo_config.yaml")
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
