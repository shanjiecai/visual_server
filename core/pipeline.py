#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
处理流水线管理器
支持按顺序执行多个预处理器，并根据条件过滤结果
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from loguru import logger

from core.interfaces import FrameData, ProcessingResult, IPreprocessor, IMessageQueue, IVideoSource
from utils.image_utils import image_to_base64


class ProcessingPipeline:
    """处理流水线"""
    
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self._processors: List[IPreprocessor] = []
        self._filters: List[Callable[[ProcessingResult], bool]] = []
        self._output_queue: Optional[IMessageQueue] = None
        self._postprocessor_config: Dict[str, Any] = {}
    
    def add_processor(self, processor: IPreprocessor) -> None:
        """添加处理器到流水线"""
        self._processors.append(processor)
        logger.info(f"添加处理器 {processor.processor_name} 到流水线 {self.pipeline_id}")
    
    def add_filter(self, filter_func: Callable[[ProcessingResult], bool], description: str = "") -> None:
        """添加过滤条件"""
        self._filters.append(filter_func)
        logger.info(f"添加过滤器到流水线 {self.pipeline_id}: {description}")
    
    def set_output_queue(self, queue: IMessageQueue) -> None:
        """设置输出队列"""
        self._output_queue = queue
        logger.info(f"设置流水线 {self.pipeline_id} 的输出队列")
    
    def set_postprocessor_config(self, config: Dict[str, Any]) -> None:
        """设置后处理器配置"""
        self._postprocessor_config = config
        logger.info(f"设置流水线 {self.pipeline_id} 的后处理器配置: {list(config.keys())}")
    
    async def process_frame(self, frame_data: FrameData) -> Optional[ProcessingResult]:
        """处理单帧数据"""
        try:
            current_result = None
            
            # 按顺序执行所有处理器
            for processor in self._processors:
                try:
                    logger.debug(f"使用 {processor.processor_name} 处理帧 {frame_data.frame_id}")
                    # 处理器之间不传递结果
                    current_result = await processor.process(frame_data)

                    if current_result is None:
                        logger.warning(f"处理器 {processor.processor_name} 返回空结果，跳过后续处理")
                        return None
                    
                    # 检查是否应该跳过后续处理
                    if current_result.metadata.get("should_skip", False) or current_result.result_data.get("should_skip", False):
                        logger.info(f"帧 {frame_data.frame_id} 被处理器 {processor.processor_name} 标记为应跳过，停止后续处理")
                        reason = current_result.result_data.get("reason", "unknown")
                        logger.info(f"跳过原因: {reason}")
                        return current_result  # 返回当前结果，但不发送到队列
                    
                except Exception as e:
                    logger.error(f"处理器 {processor.processor_name} 处理错误: {e}")
                    return None
            
            # 应用过滤条件
            if current_result and self._filters:
                for i, filter_func in enumerate(self._filters):
                    try:
                        if not filter_func(current_result):
                            logger.debug(f"帧 {frame_data.frame_id} 被过滤器 {i} 过滤掉")
                            return None
                    except Exception as e:
                        logger.error(f"过滤器 {i} 错误: {e}")
                        return None
            
            # 如果通过所有过滤器，发送到输出队列
            if current_result and self._output_queue:
                await self._send_to_queue(current_result)
            
            return current_result
            
        except Exception as e:
            logger.error(f"处理帧 {frame_data.frame_id} 时出错: {e}")
            return None
    
    async def _send_to_queue(self, result: ProcessingResult) -> None:
        """发送结果到队列"""
        try:
            # 构建队列消息
            queue_message = self._build_queue_message(result)
            
            # 发送到队列
            success = await self._output_queue.put(queue_message)
            
            if success:
                logger.info(f"已将帧 {result.frame_id} 发送到队列")
            else:
                logger.error(f"将帧 {result.frame_id} 发送到队列失败")
                
        except Exception as e:
            logger.error(f"发送结果到队列时出错: {e}")
    
    def _build_queue_message(self, result: ProcessingResult) -> Dict[str, Any]:
        """构建队列消息格式"""
        message = {
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
            "processor_name": result.processor_name,
            "confidence": result.confidence,
            "metadata": result.metadata,
            "pipeline_id": self.pipeline_id,
            "image_base64": image_to_base64(result.frame_data.raw_data["data"]),
            **result.result_data,  # 包含所有结果数据
        }
        
        # 添加独立的VLM任务配置和后处理器配置
        if self._postprocessor_config:
            # 如果有vlm_task_config，直接传递
            if "vlm_task_config" in self._postprocessor_config:
                message["vlm_task_config"] = self._postprocessor_config["vlm_task_config"]
            
            # 添加后处理器配置
            if "postprocessor_config" in self._postprocessor_config:
                message["postprocessor_config"] = self._postprocessor_config["postprocessor_config"]
            
        return message


class PipelineManager:
    """流水线管理器"""
    
    def __init__(self):
        self._pipelines: Dict[str, ProcessingPipeline] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    def create_pipeline(self, pipeline_id: str) -> ProcessingPipeline:
        """创建新的流水线"""
        pipeline = ProcessingPipeline(pipeline_id)
        self._pipelines[pipeline_id] = pipeline
        logger.info(f"创建流水线: {pipeline_id}")
        return pipeline
    
    def get_pipeline(self, pipeline_id: str) -> Optional[ProcessingPipeline]:
        """获取流水线"""
        return self._pipelines.get(pipeline_id)
    
    async def start_pipeline(self, pipeline_id: str, frame_source: IVideoSource) -> bool:
        """启动流水线处理"""
        try:
            pipeline = self.get_pipeline(pipeline_id)
            if not pipeline:
                logger.error(f"流水线 {pipeline_id} 不存在")
                return False
            
            # 创建处理任务
            task = asyncio.create_task(self._run_pipeline(pipeline, frame_source))
            self._running_tasks[pipeline_id] = task
            
            logger.info(f"已启动流水线: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"启动流水线 {pipeline_id} 出错: {e}")
            return False
    
    async def stop_pipeline(self, pipeline_id: str) -> bool:
        """停止流水线处理"""
        try:
            if pipeline_id in self._running_tasks:
                task = self._running_tasks[pipeline_id]
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                del self._running_tasks[pipeline_id]
                logger.info(f"已停止流水线: {pipeline_id}")
                return True
            else:
                logger.warning(f"流水线 {pipeline_id} 未运行")
                return False
                
        except Exception as e:
            logger.error(f"停止流水线 {pipeline_id} 出错: {e}")
            return False
    
    async def _run_pipeline(self, pipeline: ProcessingPipeline, frame_source: IVideoSource) -> None:
        """运行流水线处理"""
        logger.info(f"流水线 {pipeline.pipeline_id} 处理开始")
        
        try:
            async for frame_data in frame_source.get_frame_stream():
                if frame_data:
                    result = await pipeline.process_frame(frame_data)
                    if result:
                        logger.debug(f"流水线 {pipeline.pipeline_id} 成功处理帧 {frame_data.frame_id}")
                else:
                    # 没有新帧，短暂等待
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info(f"流水线 {pipeline.pipeline_id} 处理被取消")
            raise
        except Exception as e:
            logger.error(f"流水线 {pipeline.pipeline_id} 处理错误: {e}")
        finally:
            logger.info(f"流水线 {pipeline.pipeline_id} 处理结束")
    
    def get_pipeline_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有流水线状态"""
        status = {}
        for pipeline_id, pipeline in self._pipelines.items():
            status[pipeline_id] = {
                "processors": [p.processor_name for p in pipeline._processors],
                "filters_count": len(pipeline._filters),
                "has_output_queue": pipeline._output_queue is not None,
                "running": pipeline_id in self._running_tasks
            }
        return status
