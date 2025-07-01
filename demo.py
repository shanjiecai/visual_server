#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人员检测演示程序

用法:
    python demo.py [--config CONFIG_PATH]
    python demo.py --camera-index 0

作者：Visual Processing Team
"""

import asyncio
from loguru import logger

from core.pipeline import PipelineManager
from preprocessor.similar_frame_filter import SimilarFrameFilterProcessor
from preprocessor.yolo_detector import YOLODetectorProcessor
from core.filters import create_person_detection_filter
from core.demo_utils import (
    DemoConfig, DemoInitializer, DemoLogger, DemoRunner,
    demo_main, parse_demo_args, apply_args_to_config, cleanup_components
)


class PersonDetectionDemo(DemoRunner):
    """人员检测Demo"""
    
    def __init__(self):
        super().__init__("person_detection", "人员检测演示程序")
        self.pipeline_id = "person_detection_pipeline"
    
    async def initialize_components(self, config):
        """初始化所有组件"""
        # 使用公共工具初始化基础组件
        self.components["video_source"] = await DemoInitializer.init_video_source(config)
        self.components["kafka_queue"] = await DemoInitializer.init_kafka_queue(config)
        
        # 初始化处理器
        similar_frame_filter = await DemoInitializer.init_processor(
            "similar_frame_filter",
            SimilarFrameFilterProcessor,
            config.get("similar_frame_filter", {})
        )
        
        yolo_detector = await DemoInitializer.init_processor(
            "yolo_detector", 
            YOLODetectorProcessor,
            config.get("preprocessors", {}).get("yolo_detector", {})
        )
        
        # 创建流水线
        pipeline_manager, pipeline = DemoInitializer.create_pipeline(self.pipeline_id)
        self.components["pipeline_manager"] = pipeline_manager
        
        # 配置流水线
        pipeline.add_processor(similar_frame_filter)
        pipeline.add_processor(yolo_detector)
        pipeline.add_filter(create_person_detection_filter(yolo_detector.processor_name))
        
        # 设置VLM任务配置和后处理器配置
        vlm_and_postprocessor_config = {
            "vlm_task_config": config.get("vlm_task_config", {}),
            "postprocessor_config": config.get("postprocessors", {})
        }
        pipeline.set_postprocessor_config(vlm_and_postprocessor_config)
        pipeline.set_output_queue(self.components["kafka_queue"])
        
        return self.components
    
    async def run_main_loop(self):
        """运行主循环"""
        video_source = self.components["video_source"]
        pipeline_manager: PipelineManager = self.components["pipeline_manager"]
        
        # 启动流水线
        await pipeline_manager.start_pipeline(self.pipeline_id, video_source)
        
        # 根据视频源类型显示不同的信息
        source_info = ""
        if hasattr(video_source, 'get_camera_info'):
            # 摄像头源
            camera_info = video_source.get_camera_info()
            source_info = f"摄像头{camera_info.get('index', 'N/A')}"
        elif hasattr(video_source, 'get_video_info'):
            # 文件源
            video_info = video_source.get_video_info()
            source_info = f"文件 {video_info.get('filename', 'N/A')}"
        else:
            source_info = "视频源"
        
        logger.info(f"🚀 开始处理{source_info} -> 相似帧过滤 -> YOLO人员检测 -> VLM打招呼")
        
        # 保持程序运行
        while True:
            await asyncio.sleep(1)
    
    async def cleanup(self):
        """清理资源"""
        await cleanup_components(self.components, self.pipeline_id)


@demo_main
async def main() -> None:
    """主函数"""
    # 设置日志
    DemoLogger.setup_logging("person_detection")
    
    # 解析参数
    args = parse_demo_args("人员检测演示程序", "./configs/demo_config.yaml")
    
    # 加载配置
    config = DemoConfig.load_config(args.config)
    apply_args_to_config(config, args)
    
    logger.info("启动人员检测演示程序")
    logger.info("架构: 视频源 -> 相似帧过滤 -> YOLO人员检测 -> Kafka队列 -> VLM打招呼")
    logger.info(f"配置: {config}")
    
    # 创建并运行demo
    demo = PersonDetectionDemo()
    
    # 初始化组件
    await demo.initialize_components(config)
    
    # 设置信号处理
    demo.setup_signal_handlers(demo.cleanup)
    
    # 运行主循环（带统计信息）
    await demo.run_with_stats(demo.run_main_loop(), config)


if __name__ == "__main__":
    main()
