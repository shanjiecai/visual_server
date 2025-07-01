#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拓扑并行处理演示程序

该程序展示了如何配置和使用拓扑并行处理策略：
1. 第一阶段：相似帧过滤（串行）
2. 第二阶段：YOLO、OMDet、SAM2并行检测
3. 第三阶段：记忆提取（串行）

用法:
    python demo_topology.py [--config CONFIG_PATH]
    python demo_topology.py --camera-index 0

作者：Visual Processing Team
"""

import asyncio
from loguru import logger

from preprocessor.similar_frame_filter import SimilarFrameFilterProcessor
from core.demo_utils import (
    DemoConfig, DemoInitializer, DemoLogger, DemoRunner,
    demo_main, parse_demo_args, apply_args_to_config, cleanup_components
)


class MockProcessor:
    """模拟处理器，用于演示拓扑配置"""
    
    def __init__(self, name: str, config: dict = None):
        self._name = name
        self._config = config or {}
    
    @property
    def processor_name(self) -> str:
        return self._name
    
    async def initialize(self) -> bool:
        """初始化处理器"""
        logger.info(f"初始化模拟处理器: {self._name}")
        return True
    
    async def process(self, frame_data):
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


class TopologyDemo(DemoRunner):
    """拓扑并行处理Demo"""
    
    def __init__(self):
        super().__init__("topology", "拓扑并行处理演示程序")
        self.pipeline_id = "topology_pipeline"
    
    async def initialize_components(self, config):
        """初始化拓扑处理相关组件"""
        # 使用公共工具初始化基础组件
        self.components["video_source"] = await DemoInitializer.init_video_source(config)
        self.components["kafka_queue"] = await DemoInitializer.init_kafka_queue(config)
        
        # 初始化所有处理器
        processors = {}
        preprocessor_configs = config.get("preprocessors", {})
        
        for processor_name, processor_config in preprocessor_configs.items():
            if not processor_config.get("enabled", True):
                continue
                
            if processor_name == "similar_frame_filter":
                # 使用真实的相似帧过滤器
                processor = await DemoInitializer.init_processor(
                    processor_name,
                    SimilarFrameFilterProcessor,
                    processor_config
                )
            else:
                # 对于其他处理器，使用模拟处理器
                processor = MockProcessor(processor_name, processor_config)
                await processor.initialize()
            
            processors[processor_name] = processor
            logger.info(f"处理器 {processor_name} 初始化完成")
        
        self.components["processors"] = processors
        
        # 创建流水线
        pipeline_manager, pipeline = DemoInitializer.create_pipeline(self.pipeline_id)
        self.components["pipeline_manager"] = pipeline_manager
        
        # 配置流水线
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
        pipeline.set_output_queue(self.components["kafka_queue"])
        
        # 设置VLM任务配置
        vlm_config = {
            "vlm_task_config": config.get("vlm_task_config", {}),
            "postprocessor_config": {}
        }
        pipeline.set_postprocessor_config(vlm_config)
        
        return self.components
    
    async def run_main_loop(self):
        """运行主循环"""
        video_source = self.components["video_source"]
        pipeline_manager = self.components["pipeline_manager"]
        
        # 打印拓扑信息
        await self._print_topology_info()
        
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
        
        logger.info(f"🚀 开始拓扑并行处理: {source_info} -> 相似帧过滤 -> [YOLO+OMDet+SAM2并行] -> 记忆提取")
        
        # 保持程序运行
        while True:
            await asyncio.sleep(1)
    
    async def _print_topology_info(self):
        """打印拓扑架构信息"""
        pipeline_manager = self.components.get("pipeline_manager")
        if not pipeline_manager:
            return
            
        pipeline_status = pipeline_manager.get_pipeline_status()
        if self.pipeline_id in pipeline_status:
            status = pipeline_status[self.pipeline_id]
            if "topology" in status and status["topology"]:
                logger.info("🏗️ 拓扑处理架构:")
                logger.info(f"   总阶段数: {status['topology']['stages_count']}")
                logger.info(f"   执行顺序: {status['topology']['execution_order']}")
                
                for stage in status['topology']['stages']:
                    mode_emoji = "⚡" if stage['execution_mode'] == "parallel" else "➡️"
                    logger.info(f"   {mode_emoji} 阶段 '{stage['stage_id']}' ({stage['execution_mode']}): {stage['processors']}")
                    if stage['dependencies']:
                        logger.info(f"     依赖: {stage['dependencies']}")
    
    async def print_stats(self):
        """打印统计信息，包含拓扑相关统计"""
        # 先调用父类的统计
        await super().print_stats()
        
        # 首次打印时显示拓扑架构信息（如果还没显示过）
        # 这里可以添加更详细的拓扑统计信息
        pass
    
    async def cleanup(self):
        """清理资源"""
        await cleanup_components(self.components, self.pipeline_id)


@demo_main
async def main() -> None:
    """主函数"""
    # 设置日志
    DemoLogger.setup_logging("topology")
    
    # 解析参数
    args = parse_demo_args("拓扑并行处理演示程序", "./configs/topology_demo_config.yaml")
    
    # 加载配置
    config = DemoConfig.load_config(args.config)
    apply_args_to_config(config, args)
    
    logger.info("🚀 启动拓扑并行处理演示程序")
    logger.info("架构: 摄像头 -> 相似帧过滤 -> [YOLO+OMDet+SAM2并行] -> 记忆提取 -> 队列")
    logger.info(f"配置: {config}")
    
    # 创建并运行demo
    demo = TopologyDemo()
    
    # 初始化组件
    await demo.initialize_components(config)
    
    # 设置信号处理
    demo.setup_signal_handlers(demo.cleanup)
    
    # 运行主循环（带统计信息）
    await demo.run_with_stats(demo.run_main_loop(), config)


if __name__ == "__main__":
    main() 