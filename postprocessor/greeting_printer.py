#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
打招呼打印后处理器
用于处理VLM生成的问候语并打印输出
"""

import time
from abc import ABC
from typing import Dict, Any
from loguru import logger

from .base import BasePostProcessor
from core.interfaces import ProcessingTask


class GreetingPrinterPostprocessor(BasePostProcessor):
    """打招呼打印后处理器"""
    
    @property
    def processor_name(self) -> str:
        return "greeting_printer"
    
    async def _do_initialize(self) -> bool:
        """初始化后处理器"""
        try:
            # 获取配置
            self.trigger_conditions = self.config.get("trigger_conditions", [])
            self.greeting_config = self.config.get("greeting_config", {})
            
            # 打印格式配置
            self.print_format = self.greeting_config.get("print_format", "🤖 AI打招呼: {content}")
            self.include_metadata = self.greeting_config.get("include_metadata", True)
            self.show_person_count = self.greeting_config.get("show_person_count", True)
            
            logger.info("Greeting printer postprocessor initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize greeting printer: {e}")
            return False
    
    async def _do_execute(self, task: ProcessingTask) -> Dict[str, Any]:
        """执行打招呼打印"""
        try:
            # 检查是否满足触发条件
            if not self._should_trigger(task):
                return {"status": "skipped", "reason": "trigger conditions not met"}
            
            # 获取VLM处理结果
            vlm_results = [r for r in task.processing_results if r.processor_name == "vlm_processor"]
            
            if not vlm_results:
                return {"status": "skipped", "reason": "no VLM results found"}
            
            # 处理每个VLM结果
            for result in vlm_results:
                await self._print_greeting(result.result_data, task)
            
            return {
                "status": "success",
                "processed_count": len(vlm_results),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in greeting printer: {e}")
            return {"status": "error", "error": str(e)}
    
    def _should_trigger(self, task: ProcessingTask) -> bool:
        """检查是否应该触发打招呼"""
        try:
            for condition in self.trigger_conditions:
                condition_type = condition.get("type")
                
                if condition_type == "detection_threshold":
                    # 检查检测数量阈值
                    threshold = condition.get("threshold", 1)
                    detection_count = 0
                    
                    # 从处理结果中获取检测数量
                    for result in task.processing_results:
                        if result.processor_name == "yolo_detector":
                            detection_count = result.result_data.get("detection_count", 0)
                            break
                    
                    if detection_count < threshold:
                        return False
                
                elif condition_type == "task_type":
                    # 检查任务类型
                    required_types = condition.get("task_types", [])
                    task_type = None
                    
                    # 从处理结果中获取任务类型
                    for result in task.processing_results:
                        if result.processor_name == "yolo_detector":
                            task_type = result.result_data.get("task_type")
                            break
                    
                    if task_type not in required_types:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trigger conditions: {e}")
            return False
    
    async def _print_greeting(self, vlm_result: Dict[str, Any], task: ProcessingTask) -> None:
        """打印问候语"""
        try:
            # 获取AI生成的内容
            content = vlm_result.get("content", "")
            
            if not content:
                logger.warning("No content found in VLM result")
                return
            
            # 格式化输出
            print_text = self.print_format.format(content=content)
            
            # 打印主要内容
            logger.info("=" * 60)
            logger.info(print_text)
            
            # 如果需要显示元数据
            if self.include_metadata:
                frame_id = task.frame_data.frame_id
                processing_time = vlm_result.get("processing_time", 0)
                
                logger.info(f"📋 Frame ID: {frame_id}")
                logger.info(f"⏱️  处理时间: {processing_time:.2f}s")
                
                # 显示人员数量
                if self.show_person_count:
                    person_count = self._get_person_count(task)
                    if person_count > 0:
                        logger.info(f"👥 检测到人数: {person_count}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error printing greeting: {e}")
    
    def _get_person_count(self, task: ProcessingTask) -> int:
        """获取检测到的人员数量"""
        try:
            for result in task.processing_results:
                if result.processor_name == "yolo_detector":
                    detections = result.result_data.get("detections", [])
                    return len([d for d in detections if d.get("class_name") == "person"])
            return 0
        except Exception as e:
            logger.error(f"Error getting person count: {e}")
            return 0
    
    async def _do_cleanup(self) -> None:
        """清理资源"""
        logger.info("Greeting printer postprocessor cleaned up") 