#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拓扑并行处理演示程序

该程序展示了如何配置和使用拓扑并行处理策略：
1. 第一阶段：相似帧过滤（串行）
2. 第二阶段：YOLO、OMDet、SAM2并行检测
3. 第三阶段：记忆提取（串行）

架构流程：
摄像头 -> 相似帧过滤 -> [YOLO + OMDet + SAM2 并行] -> 记忆提取 -> 队列

用法:
    python demo_topology.py [--config CONFIG_PATH]
    python demo_topology.py --camera-index 0 --camera-fps 1.0

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
# from preprocessor.yolo_detector import YOLODetectorProcessor
# from preprocessor.omdet_detector import OMDetDetectorProcessor  
# from preprocessor.sam2_segmentor import SAM2SegmentorProcessor
# from preprocessor.memory_extractor import MemoryExtractorProcessor
from message_queue.kafka_queue import KafkaQueue
from core.pipeline import PipelineManager


class MockProcessor:
    """模拟处理器，用于演示拓扑配置"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self._name = name
        self._config = config or {}
    
    @property
    def processor_name(self) -> str:
        return self._name
    
    async def initialize(self) -> bool:
        """初始化处理器"""
        logger.info(f"初始化模拟处理器: {self._name}")
        return True
    
    async def process(self, frame_data) -> Any:
        """模拟处理逻辑"""
        # 模拟处理时间
        processing_time = self._config.get("processing_time", 0.1)
        await asyncio.sleep(processing_time)
        
        # 创建模拟结果
        from core.interfaces import ProcessingResult
        result = ProcessingResult(
            frame_id=frame_data.frame_id,
            timestamp=frame_data.timestamp,
            processor_name=self._name,
            confidence=0.8,
            result_data={
                "detections": [f"{self._name}_detection_1", f"{self._name}_detection_2"],
                "processing_time": processing_time,
                "status": "success"
            },
            metadata={
                "processor_type": self._config.get("processor_type", "mock"),
                "model_path": self._config.get("model_path", f"models/{self._name}.pt")
            },
            frame_data=frame_data
        )
        
        logger.info(f"{self._name} 处理完成 - 帧ID: {frame_data.frame_id}")
        return result


async def initialize_topology_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """初始化拓扑处理相关组件"""
    # 初始化摄像头源
    camera_config = config.get("camera", {})
    camera_source = CameraVideoSource(
        camera_index=camera_config.get("camera_index", 0),
        fps=camera_config.get("fps", 1.0),
    )
    await camera_source.initialize()
    logger.info(f"摄像头初始化完成: index={camera_config.get('camera_index', 0)}, fps={camera_config.get('fps', 1.0)}")

    # 初始化所有处理器
    processors = {}
    preprocessor_configs = config.get("preprocessors", {})
    
    for processor_name, processor_config in preprocessor_configs.items():
        if not processor_config.get("enabled", True):
            continue
            
        if processor_name == "similar_frame_filter":
            # 使用真实的相似帧过滤器
            processor = SimilarFrameFilterProcessor(config=processor_config)
        else:
            # 对于其他处理器，使用模拟处理器
            processor = MockProcessor(processor_name, processor_config)
        
        await processor.initialize()
        processors[processor_name] = processor
        logger.info(f"处理器 {processor_name} 初始化完成")

    # 初始化Kafka队列
    queue_config = config.get("queue", {})
    if queue_config.get("type") == "kafka":
        kafka_queue = KafkaQueue(config=queue_config.get("config", {}))
        await kafka_queue.initialize()
        logger.info(f"Kafka队列初始化完成: topic={queue_config.get('config', {}).get('topic_name', 'topology_demo')}")
    else:
        raise ValueError(f"不支持的队列类型: {queue_config.get('type')}")

    # 创建流水线
    pipeline_manager = PipelineManager()
    pipeline = pipeline_manager.create_pipeline("topology_pipeline")
    
    # 添加所有处理器到流水线
    for processor in processors.values():
        pipeline.add_processor(processor)
    
    # 设置拓扑配置 - 这是核心功能
    topology_config = config.get("topology", {})
    if topology_config:
        pipeline.set_topology(topology_config)
        logger.info("✅ 拓扑配置设置完成")
    else:
        logger.warning("未找到拓扑配置，将使用默认串行处理")
    
    # 设置输出队列
    pipeline.set_output_queue(kafka_queue)
    
    # 设置VLM任务配置
    vlm_config = {
        "vlm_task_config": config.get("vlm_task_config", {}),
        "postprocessor_config": {}
    }
    pipeline.set_postprocessor_config(vlm_config)
    
    logger.info("拓扑并行处理流水线创建完成")

    return {
        "camera_source": camera_source,
        "pipeline_manager": pipeline_manager,
        "kafka_queue": kafka_queue,
        "processors": processors
    }


async def run_topology_demo(config: Dict[str, Any]) -> None:
    """运行拓扑处理主流程"""
    components = await initialize_topology_components(config)
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
        # 打印拓扑信息
        pipeline_status = pipeline_manager.get_pipeline_status()
        if "topology_pipeline" in pipeline_status:
            status = pipeline_status["topology_pipeline"]
            if "topology" in status:
                logger.info("🏗️ 拓扑处理架构:")
                logger.info(f"   总阶段数: {status['topology']['stages_count']}")
                logger.info(f"   执行顺序: {status['topology']['execution_order']}")
                
                for stage in status['topology']['stages']:
                    mode_emoji = "⚡" if stage['execution_mode'] == "parallel" else "➡️"
                    logger.info(f"   {mode_emoji} 阶段 '{stage['stage_id']}' ({stage['execution_mode']}): {stage['processors']}")
                    if stage['dependencies']:
                        logger.info(f"     依赖: {stage['dependencies']}")
        
        # 启动流水线
        await pipeline_manager.start_pipeline("topology_pipeline", camera_source)
        logger.info("🚀 开始拓扑并行处理: 摄像头 -> 相似帧过滤 -> [YOLO+OMDet+SAM2并行] -> 记忆提取")
        
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
                        logger.info(f"📊 流水线状态: {pipeline_status}")
                    
                    # 获取队列统计
                    if hasattr(components["kafka_queue"], "get_stats"):
                        queue_stats = components["kafka_queue"].get_stats()
                        logger.info(f"📨 队列统计: {queue_stats}")
                    
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
        await pipeline_manager.stop_pipeline("topology_pipeline")
    
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
        config_path = "./configs/topology_demo_config.yaml"
    
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
        log_dir / "topology_demo.log",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="拓扑并行处理演示程序")
    parser.add_argument("--config", type=str, help="配置文件路径", default="./configs/topology_demo_config.yaml")
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
    
    logger.info("🚀 启动拓扑并行处理演示程序")
    logger.info("架构: 摄像头 -> 相似帧过滤 -> [YOLO+OMDet+SAM2并行] -> 记忆提取 -> 队列")
    logger.info(f"配置: {config}")
    
    # 运行主程序
    await run_topology_demo(config)


if __name__ == "__main__":
    asyncio.run(main()) 