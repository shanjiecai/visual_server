#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视觉记忆提取演示程序

该程序实现了以下功能：
1. 从视频源（摄像头或MP4文件）捕获视频帧
2. 通过相似帧过滤器去除重复帧
3. 将过滤后的帧发送到Kafka队列，并携带记忆提取任务配置
4. VLM Worker消费队列进行记忆提取和存储
5. 提供记忆查询API服务

支持的视频源：
- 摄像头：实时捕获摄像头画面
- MP4文件：从本地MP4视频文件读取帧，支持循环播放

用法:
    python demo_memory.py [--config CONFIG_PATH]
    python demo_memory.py --camera-index 0
    python demo_memory.py --video-file ./test_video.mp4 --video-fps 1.0 --loop

配置文件说明：
    在configs/memory_demo_config.yaml中配置video_source:
    - type: "camera" 使用摄像头
    - type: "file" 使用MP4文件，需设置file_path

作者：Visual Processing Team
"""

import asyncio
from loguru import logger

from preprocessor.similar_frame_filter import SimilarFrameFilterProcessor
from app.memory_router import create_memory_api_service
from core.demo_utils import (
    DemoConfig, DemoInitializer, DemoLogger, DemoRunner,
    demo_main, parse_demo_args, apply_args_to_config, cleanup_components
)


class MemoryExtractionDemo(DemoRunner):
    """记忆提取Demo"""
    
    def __init__(self):
        super().__init__("memory_extraction", "视觉记忆提取演示程序")
        self.pipeline_id = "memory_pipeline"
    
    async def initialize_components(self, config):
        """初始化记忆提取相关组件"""
        # 使用公共工具初始化基础组件 - 支持摄像头或文件源
        self.components["video_source"] = await DemoInitializer.init_video_source(config)
        self.components["kafka_queue"] = await DemoInitializer.init_kafka_queue(config)
        
        # 初始化相似帧过滤器
        similar_frame_filter = await DemoInitializer.init_processor(
            "similar_frame_filter",
            SimilarFrameFilterProcessor,
            config.get("preprocessors", {}).get("similar_frame_filter", {})
        )
        
        # 创建流水线
        pipeline_manager, pipeline = DemoInitializer.create_pipeline(self.pipeline_id)
        self.components["pipeline_manager"] = pipeline_manager
        
        # 配置流水线
        pipeline.add_processor(similar_frame_filter)
        pipeline.set_output_queue(self.components["kafka_queue"])
        
        # 设置任务配置（传递给队列消息）
        task_config = self._create_memory_task_config(config)
        pipeline.set_postprocessor_config(task_config)
        
        # 初始化记忆API服务（如果启用）
        if config.get("memory_api", {}).get("enabled", False):
            api_config = config.get("memory_api", {})
            memory_api_service = create_memory_api_service(api_config)
            await memory_api_service.initialize()
            self.components["memory_api_service"] = memory_api_service
            logger.info("记忆查询API服务配置完成")
        
        return self.components
    
    def _create_memory_task_config(self, config):
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
                    "system_prompt": memory_config.get("detection_system_prompt", self._get_default_detection_prompt(memory_config)),
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
    
    def _get_default_detection_prompt(self, config):
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
    
    async def run_main_loop(self):
        """运行主循环"""
        video_source = self.components["video_source"]
        pipeline_manager = self.components["pipeline_manager"]
        
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
        
        logger.info(f"🚀 开始处理{source_info} -> 相似帧过滤 -> 队列(携带任务配置) -> VLM Worker")
        
        # 保持程序运行
        while True:
            await asyncio.sleep(1)
    
    async def print_stats(self):
        """打印统计信息，包含记忆相关统计"""
        # 先调用父类的统计
        await super().print_stats()
        
        # 获取记忆统计（如果API服务可用）
        memory_api_service = self.components.get("memory_api_service")
        if memory_api_service and hasattr(memory_api_service, "get_memory_storage"):
            try:
                memory_storage = memory_api_service.get_memory_storage()
                if memory_storage and hasattr(memory_storage, "get_memory_stats"):
                    memory_stats = memory_storage.get_memory_stats()
                    logger.info(f"📚 记忆统计: {memory_stats}")
            except Exception as e:
                logger.debug(f"获取记忆统计失败: {e}")
    
    async def cleanup(self):
        """清理资源"""
        await cleanup_components(self.components, self.pipeline_id)


@demo_main
async def main() -> None:
    """主函数"""
    # 设置日志
    DemoLogger.setup_logging("memory_extraction")
    
    # 解析参数
    args = parse_demo_args("视觉记忆提取演示程序", "./configs/memory_demo_config.yaml")
    
    # 加载配置
    config = DemoConfig.load_config(args.config)
    apply_args_to_config(config, args)
    
    logger.info("启动视觉记忆提取演示程序")
    logger.info("架构: 摄像头 -> 相似帧过滤 -> Kafka队列(携带任务配置) -> VLM Worker(记忆提取)")
    logger.info(f"配置: {config}")
    
    # 提示用户API服务信息
    if config.get("memory_api", {}).get("enabled", False):
        logger.info("API文档: http://localhost:8000/docs")
    
    # 创建并运行demo
    demo = MemoryExtractionDemo()
    
    # 初始化组件
    await demo.initialize_components(config)
    
    # 设置信号处理
    demo.setup_signal_handlers(demo.cleanup)
    
    # 运行主循环（带统计信息）
    await demo.run_with_stats(demo.run_main_loop(), config)


if __name__ == "__main__":
    main()
