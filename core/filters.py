#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预先定义好的预处理结果过滤器
"""

from typing import Callable
from loguru import logger
from core.interfaces import ProcessingResult


def create_person_detection_filter(processor_name: str = "yolo_detector") -> Callable[[ProcessingResult], bool]:
    """创建人员检测过滤器 - 只有检测到人的帧才通过
    
    Args:
        processor_name: YOLO检测器的处理器名称
        
    Returns:
        过滤函数
    """
    def filter_func(result: ProcessingResult) -> bool:
        try:
            # 检查是否是YOLO检测结果
            if result.processor_name != processor_name:
                return True  # 非YOLO结果直接通过
            
            # 检查是否检测到人
            categories = result.result_data.get("categories", [])
            detections = result.result_data.get("detections", [])
            
            # 检查类别中是否有"person"
            has_person = "person" in categories
            
            # 检查检测结果中是否有高置信度的人
            person_detections = [
                det for det in detections 
                if det.get("class_name") == "person" and det.get("confidence", 0) > 0.8
            ]
            
            if has_person and person_detections:
                logger.info(f"帧 {result.frame_id} 包含 {len(person_detections)} 个人，放行到队列")
                return True
            else:
                logger.debug(f"帧 {result.frame_id} 没有检测到人，被过滤掉")
                return False
                
        except Exception as e:
            logger.error(f"人员检测过滤器错误: {e}")
            return False  # 出错时默认不通过
    
    return filter_func
