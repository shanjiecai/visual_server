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


class ProcessorStage:
    """处理器阶段配置"""
    
    def __init__(self, stage_id: str, execution_mode: str = "sequential", 
                 processors: List[str] = None, dependencies: List[str] = None):
        self.stage_id = stage_id
        self.execution_mode = execution_mode  # "parallel" 或 "sequential"
        self.processors = processors or []  # 处理器名称列表
        self.dependencies = dependencies or []  # 依赖的阶段ID列表


class ProcessingPipeline:
    """处理流水线"""
    
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self._processors: List[IPreprocessor] = []
        self._processor_map: Dict[str, IPreprocessor] = {}  # 处理器名称到对象的映射
        self._topology: List[ProcessorStage] = []
        self._filters: List[Callable[[ProcessingResult], bool]] = []
        self._output_queue: Optional[IMessageQueue] = None
        self._postprocessor_config: Dict[str, Any] = {}
    
    def add_processor(self, processor: IPreprocessor) -> None:
        """添加处理器到流水线"""
        self._processors.append(processor)
        self._processor_map[processor.processor_name] = processor
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

    def set_topology(self, topology_config: Dict[str, Any]) -> None:
        """设置拓扑配置"""
        stages_config = topology_config.get("stages", [])
        self._topology = []
        
        for stage_config in stages_config:
            stage = ProcessorStage(
                stage_id=stage_config.get("stage_id"),
                execution_mode=stage_config.get("execution_mode", "sequential"),
                processors=stage_config.get("processors", []),
                dependencies=stage_config.get("dependencies", [])
            )
            self._topology.append(stage)
        
        logger.info(f"设置流水线 {self.pipeline_id} 的拓扑: {len(self._topology)} 个阶段")
        for stage in self._topology:
            mode_emoji = "⚡" if stage.execution_mode == "parallel" else "➡️"
            logger.info(f"  {mode_emoji} 阶段 '{stage.stage_id}' ({stage.execution_mode}): {stage.processors}")
    
    async def process_frame(self, frame_data: FrameData) -> Optional[ProcessingResult]:
        """处理单帧数据 - 支持拓扑并行执行"""
        try:
            # 如果没有配置拓扑，使用传统的串行处理
            if not self._topology:
                return await self._process_frame_sequential(frame_data)
            
            # 使用拓扑配置进行处理
            return await self._process_frame_with_topology(frame_data)
            
        except Exception as e:
            logger.error(f"处理帧 {frame_data.frame_id} 时出错: {e}")
            return None
    
    async def _process_frame_sequential(self, frame_data: FrameData) -> Optional[ProcessingResult]:
        """传统的串行处理方式（兼容性保证）"""
        current_result = None
        
        # 按顺序执行所有处理器
        for processor in self._processors:
            try:
                logger.debug(f"使用 {processor.processor_name} 处理帧 {frame_data.frame_id}")
                current_result = await processor.process(frame_data)

                if current_result is None:
                    logger.warning(f"处理器 {processor.processor_name} 返回空结果，跳过后续处理")
                    return None
                
                # 检查是否应该跳过后续处理
                if current_result.metadata.get("should_skip", False) or current_result.result_data.get("should_skip", False):
                    logger.info(f"帧 {frame_data.frame_id} 被处理器 {processor.processor_name} 标记为应跳过，停止后续处理")
                    reason = current_result.result_data.get("reason", "unknown")
                    logger.info(f"跳过原因: {reason}")
                    return current_result
                
            except Exception as e:
                logger.error(f"处理器 {processor.processor_name} 处理错误: {e}")
                return None
        
        # 应用过滤器和发送到队列
        if current_result:
            current_result = await self._apply_filters_and_queue(current_result)
        
        return current_result
    
    async def _process_frame_with_topology(self, frame_data: FrameData) -> Optional[ProcessingResult]:
        """基于拓扑配置的处理方式"""
        stage_results = {}  # 保存每个阶段的结果
        final_result = None
        
        # 按依赖关系排序阶段
        sorted_stages = self._sort_stages_by_dependencies()
        
        for stage in sorted_stages:
            logger.debug(f"执行阶段 '{stage.stage_id}' - 模式: {stage.execution_mode}")
            
            # 获取该阶段的处理器
            stage_processors = []
            for processor_name in stage.processors:
                if processor_name in self._processor_map:
                    stage_processors.append(self._processor_map[processor_name])
                else:
                    logger.warning(f"未找到处理器: {processor_name}")
            
            if not stage_processors:
                logger.warning(f"阶段 '{stage.stage_id}' 没有有效的处理器")
                continue
            
            # 根据执行模式处理
            if stage.execution_mode == "parallel":
                stage_result = await self._execute_stage_parallel(stage_processors, frame_data, stage.stage_id)
            else:
                stage_result = await self._execute_stage_sequential(stage_processors, frame_data, stage.stage_id)
            
            if stage_result is None:
                logger.warning(f"阶段 '{stage.stage_id}' 处理失败")
                return None
            
            # 检查是否应该跳过后续处理
            if isinstance(stage_result, ProcessingResult):
                if stage_result.metadata.get("should_skip", False) or stage_result.result_data.get("should_skip", False):
                    reason = stage_result.result_data.get("reason", "unknown")
                    logger.info(f"帧 {frame_data.frame_id} 在阶段 '{stage.stage_id}' 被标记为应跳过，原因: {reason}")
                    return stage_result
            
            stage_results[stage.stage_id] = stage_result
            final_result = stage_result  # 更新最终结果
        
        # 应用过滤器和发送到队列
        if final_result:
            final_result = await self._apply_filters_and_queue(final_result)
        
        return final_result
    
    async def _execute_stage_parallel(self, processors: List[IPreprocessor], 
                                     frame_data: FrameData, stage_id: str) -> Optional[ProcessingResult]:
        """并行执行阶段中的处理器"""
        try:
            # 并行执行所有处理器
            tasks = [processor.process(frame_data) for processor in processors]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"处理器 {processors[i].processor_name} 在阶段 '{stage_id}' 执行失败: {result}")
                elif result is not None:
                    valid_results.append(result)
                    logger.debug(f"处理器 {processors[i].processor_name} 在阶段 '{stage_id}' 执行成功")
            
            if not valid_results:
                logger.warning(f"阶段 '{stage_id}' 并行执行后没有有效结果")
                return None
            
            # 合并多个处理器的结果
            merged_result = self._merge_parallel_results(valid_results, stage_id)
            logger.info(f"阶段 '{stage_id}' 并行执行完成，合并了 {len(valid_results)} 个结果")
            
            return merged_result
            
        except Exception as e:
            logger.error(f"阶段 '{stage_id}' 并行执行出错: {e}")
            return None
    
    async def _execute_stage_sequential(self, processors: List[IPreprocessor], 
                                       frame_data: FrameData, stage_id: str) -> Optional[ProcessingResult]:
        """串行执行阶段中的处理器"""
        current_result = None
        
        for processor in processors:
            try:
                logger.debug(f"串行执行处理器 {processor.processor_name} 在阶段 '{stage_id}'")
                current_result = await processor.process(frame_data)
                
                if current_result is None:
                    logger.warning(f"处理器 {processor.processor_name} 在阶段 '{stage_id}' 返回空结果")
                    return None
                    
                # 检查是否应该跳过
                if current_result.metadata.get("should_skip", False) or current_result.result_data.get("should_skip", False):
                    logger.info(f"处理器 {processor.processor_name} 在阶段 '{stage_id}' 标记跳过")
                    return current_result
                    
            except Exception as e:
                logger.error(f"处理器 {processor.processor_name} 在阶段 '{stage_id}' 执行错误: {e}")
                return None
        
        logger.info(f"阶段 '{stage_id}' 串行执行完成")
        return current_result
    
    def _merge_parallel_results(self, results: List[ProcessingResult], stage_id: str) -> ProcessingResult:
        """合并并行处理的结果"""
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # 取第一个结果作为基础
        base_result = results[0]
        
        # 合并所有结果的数据
        merged_result_data = {}
        merged_metadata = {}
        all_processor_names = []
        total_confidence = 0
        
        for result in results:
            # 合并result_data
            for key, value in result.result_data.items():
                if key in merged_result_data:
                    # 如果键已存在，尝试合并
                    if isinstance(value, list) and isinstance(merged_result_data[key], list):
                        merged_result_data[key].extend(value)
                    elif isinstance(value, dict) and isinstance(merged_result_data[key], dict):
                        merged_result_data[key].update(value)
                    else:
                        merged_result_data[f"{result.processor_name}_{key}"] = value
                else:
                    merged_result_data[key] = value
            
            # 合并metadata
            merged_metadata.update(result.metadata)
            all_processor_names.append(result.processor_name)
            total_confidence += result.confidence
        
        # 创建合并后的结果
        merged_result = ProcessingResult(
            frame_id=base_result.frame_id,
            timestamp=base_result.timestamp,
            processor_name=f"merged_{stage_id}",
            confidence=total_confidence / len(results),  # 平均置信度
            result_data=merged_result_data,
            metadata={
                **merged_metadata,
                "stage_id": stage_id,
                "merged_processors": all_processor_names,
                "parallel_execution": True
            },
            frame_data=base_result.frame_data
        )
        
        return merged_result
    
    def _sort_stages_by_dependencies(self) -> List[ProcessorStage]:
        """根据依赖关系对阶段进行拓扑排序"""
        stages = self._topology.copy()
        sorted_stages = []
        processed_stages = set()
        
        while stages:
            # 找到没有未处理依赖的阶段
            available_stages = []
            for stage in stages:
                dependencies_satisfied = all(dep in processed_stages for dep in stage.dependencies)
                if dependencies_satisfied:
                    available_stages.append(stage)
            
            if not available_stages:
                # 检测到循环依赖
                logger.error("检测到循环依赖，无法排序阶段")
                remaining_stages = [s.stage_id for s in stages]
                logger.error(f"剩余阶段: {remaining_stages}")
                # 返回剩余阶段，忽略依赖关系
                return sorted_stages + stages
            
            # 处理可用的阶段
            for stage in available_stages:
                sorted_stages.append(stage)
                processed_stages.add(stage.stage_id)
                stages.remove(stage)
        
        return sorted_stages
    
    async def _apply_filters_and_queue(self, result: ProcessingResult) -> Optional[ProcessingResult]:
        """应用过滤器并发送到队列"""
        # 应用过滤条件
        if self._filters:
            for i, filter_func in enumerate(self._filters):
                try:
                    if not filter_func(result):
                        logger.debug(f"帧 {result.frame_id} 被过滤器 {i} 过滤掉")
                        return None
                except Exception as e:
                    logger.error(f"过滤器 {i} 错误: {e}")
                    return None
        
        # 如果通过所有过滤器，发送到输出队列
        if self._output_queue:
            await self._send_to_queue(result)
        
        return result

    async def _send_to_queue(self, result: ProcessingResult) -> None:
        """发送结果到队列"""
        try:
            # 构建队列消息
            queue_message = self._build_queue_message(result)
            
            # 打印大小，多少字节
            print(f"队列消息大小: {len(queue_message['image_base64'])} 字节")

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
            pipeline_status = {
                "processors": [p.processor_name for p in pipeline._processors],
                "filters_count": len(pipeline._filters),
                "has_output_queue": pipeline._output_queue is not None,
                "running": pipeline_id in self._running_tasks
            }
            
            # 添加拓扑信息
            if pipeline._topology:
                topology_info = {
                    "stages_count": len(pipeline._topology),
                    "execution_order": [stage.stage_id for stage in pipeline._sort_stages_by_dependencies()],
                    "stages": []
                }
                
                for stage in pipeline._topology:
                    stage_info = {
                        "stage_id": stage.stage_id,
                        "execution_mode": stage.execution_mode,
                        "processors": stage.processors,
                        "dependencies": stage.dependencies
                    }
                    topology_info["stages"].append(stage_info)
                
                pipeline_status["topology"] = topology_info
            else:
                pipeline_status["topology"] = None
            
            status[pipeline_id] = pipeline_status
        return status
