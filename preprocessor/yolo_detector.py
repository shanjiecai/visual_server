#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO检测器实现
用于人员及物体检测和定位
支持配置prompt和metadata传递到下游处理
"""

import asyncio
import time
import cv2
import base64
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger

from .base import BasePreprocessor
from core.interfaces import FrameData, ProcessingResult


# YOLO模型配置
DEFAULT_YOLO_MODEL_PATH = "models/yoloe-v8l-seg.pt"
DEFAULT_YOLO_DEVICE = "cpu"  # 如果没有GPU，会自动回退到CPU
# DEFAULT_DETECTION_CLASSES = [
#     'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
#     'traffic light', 'fire hydrant', 'stop sign', 'bench', 'bird', 'cat',
#     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#     'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
#     'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#     'toothbrush'
# ]
DEFAULT_DETECTION_CLASSES = ["person", "chair", "table", "cup", "phone", "laptop"]


class YOLODetectorProcessor(BasePreprocessor):
    """YOLO检测处理器实现"""
    
    @property
    def processor_name(self) -> str:
        return "yolo_detector"
    
    async def _do_initialize(self) -> bool:
        """初始化YOLO检测器"""
        try:
            self.model_path = self.config.get("model_path", DEFAULT_YOLO_MODEL_PATH)
            self.device = self.config.get("device", DEFAULT_YOLO_DEVICE)
            self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
            self.nms_threshold = self.config.get("nms_threshold", 0.4)
            self.target_classes = self.config.get("target_classes", DEFAULT_DETECTION_CLASSES)
            
            # 任务配置 - 支持不同类型的视觉任务
            self.task_configs = self.config.get("task_configs", self._get_default_task_configs())
            
            # 是否需要发送到下游处理
            self.enable_downstream = self.config.get("enable_downstream", True)
            
            # 加载YOLO模型（简化实现）
            await self._load_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            return False
    
    def _get_default_task_configs(self) -> Dict[str, Dict[str, str]]:
        """获取默认任务配置"""
        return {
            "person_detection": {
                "system_prompt": """你是一个专业的人员检测和行为分析专家。请分析图像中的人员位置、动作和行为。
                
请重点关注：
1. 人员的位置和姿态
2. 人员的动作和行为
3. 人员与环境的交互
4. 安全相关的观察""",
                "user_prompt": "请分析图像中的人员情况，包括位置、动作和可能的行为意图。",
                "task_type": "person_detection"
            },
            "object_detection": {
                "system_prompt": """你是一个专业的物体检测和场景分析专家。请分析图像中的物体和场景信息。
                
请重点关注：
1. 物体的类型和位置
2. 物体的状态和条件
3. 场景的整体布局
4. 物体之间的关系""",
                "user_prompt": "请分析图像中的物体和场景，描述物体的类型、位置和相互关系。",
                "task_type": "object_detection"
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
    
    async def _load_model(self) -> None:
        """加载YOLO模型（简化实现）"""
        try:
            # 模拟模型加载
            logger.info(f"Loading YOLO model: {self.model_path} on device {self.device}")
            await asyncio.sleep(0.1)  # 模拟加载时间
            self.yolo_available = True
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"YOLO model loading failed: {e}")
            self.yolo_available = False
    
    async def _do_process(self, frame_data: FrameData) -> ProcessingResult:
        """执行目标检测"""
        try:
            # 从帧数据中提取图像
            raw_data = frame_data.raw_data
            image = self._get_image_from_frame(raw_data)
            
            if image is None:
                raise ValueError("无法从帧数据中提取图像")
            
            # 执行检测（简化实现）
            detections = await self._detect_objects(image)
            
            # 提取检测到的类别
            categories = self._extract_categories(detections)
            
            # 转换图像为base64（用于下游处理）
            image_base64 = None
            if self.enable_downstream:
                image_base64 = self._image_to_base64(image)
            
            # 确定任务类型
            task_type = self._determine_task_type(categories)
            
            # 获取任务配置
            task_config = self.task_configs.get(task_type, self.task_configs["general_analysis"])
            
            # 构建结果数据
            result_data = {
                "detections": detections,
                "categories": categories,
                "task_type": task_type,
                "image_base64": image_base64,
                "model_info": {
                    "model_path": self.model_path,
                    "confidence_threshold": self.confidence_threshold,
                    "nms_threshold": self.nms_threshold,
                    "device": self.device
                },
                "frame_id": frame_data.frame_id,
                "timestamp": frame_data.timestamp,
                # 添加下游处理需要的prompt信息
                "system_prompt": task_config["system_prompt"],
                "user_prompt": task_config["user_prompt"],
                "prompt": task_config["user_prompt"],  # 兼容字段
                "metadata": {
                    "source": "yolo_detector",
                    "detection_count": len(detections),
                    "primary_objects": categories[:3] if categories else [],
                    "task_type": task_type,
                    "processing_timestamp": time.time()
                }
            }
            
            # 计算置信度
            confidence = self._calculate_confidence(detections, categories)
            
            return self._create_result(
                frame_data,
                result_data,
                confidence=confidence,
                additional_metadata={
                    "model_path": self.model_path,
                    "detection_count": len(detections),
                    "categories": categories,
                    "task_type": task_type,
                    "enable_downstream": self.enable_downstream
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"YOLO detection failed: {e}")
    
    def _get_image_from_frame(self, raw_data: Any) -> Optional[np.ndarray]:
        """从帧数据中提取图像"""
        try:
            # 检查是否已经是numpy数组
            if isinstance(raw_data, np.ndarray):
                return raw_data
            
            # 检查是否是字典格式（包含数据和元信息）
            if isinstance(raw_data, dict) and "data" in raw_data:
                data = raw_data["data"]
                
                # 如果data是numpy数组
                if isinstance(data, np.ndarray):
                    return data
                
                # 如果data是字符串（模拟数据），创建空白图像
                if isinstance(data, str) and data.startswith("mock_frame_data_"):
                    width = raw_data.get("width", 640)
                    height = raw_data.get("height", 480)
                    return np.zeros((height, width, 3), dtype=np.uint8)
                    
            # 未能识别的格式
            logger.warning(f"Unrecognized frame data format: {type(raw_data)}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting image from frame: {e}")
            return None
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """将图像转换为base64编码"""
        try:
            # 编码为JPEG格式
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""
    
    async def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用YOLO模型检测图像中的物体（简化实现）"""
        try:
            # 模拟检测结果
            await asyncio.sleep(0.1)  # 模拟处理时间
            
            # 模拟一些检测结果
            mock_detections = [
                {
                    "class_name": "person",
                    "confidence": 0.85,
                    "bbox": [100, 50, 200, 300],  # x1, y1, x2, y2
                    "center": [150, 175]
                },
                {
                    "class_name": "chair", 
                    "confidence": 0.72,
                    "bbox": [300, 200, 450, 400],
                    "center": [375, 300]
                }
            ]
            
            # 过滤置信度
            filtered_detections = [
                det for det in mock_detections 
                if det["confidence"] >= self.confidence_threshold
            ]
            
            logger.info(f"Detected {len(filtered_detections)} objects")
            return filtered_detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _extract_categories(self, detection_results: List[Dict]) -> List[str]:
        """从检测结果中提取类别"""
        try:
            categories = []
            seen_classes = set()
            
            # 按置信度排序
            sorted_detections = sorted(
                detection_results, 
                key=lambda x: x.get("confidence", 0), 
                reverse=True
            )
            
            for detection in sorted_detections:
                class_name = detection.get("class_name", "unknown")
                if class_name not in seen_classes:
                    categories.append(class_name)
                    seen_classes.add(class_name)
            
            return categories
        except Exception as e:
            logger.error(f"Error extracting categories: {e}")
            return []
    
    def _determine_task_type(self, categories: List[str]) -> str:
        """根据检测到的类别确定任务类型"""
        if "person" in categories:
            return "person_detection"
        elif any(obj in categories for obj in ["chair", "table", "cup", "phone", "laptop"]):
            return "object_detection"
        else:
            return "general_analysis"
    
    def _calculate_confidence(self, detections: List[Dict], categories: List[str]) -> float:
        """计算整体检测置信度"""
        if not detections:
            return 0.0
        
        # 计算平均置信度
        total_confidence = sum(det.get("confidence", 0) for det in detections)
        avg_confidence = total_confidence / len(detections)
        
        # 根据检测数量调整
        detection_bonus = min(0.1, len(detections) * 0.02)
        
        return min(1.0, avg_confidence + detection_bonus)
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """设置置信度阈值"""
        self.confidence_threshold = threshold
        logger.info(f"Set confidence threshold to {threshold}")
    
    def set_task_config(self, task_type: str, config: Dict[str, str]) -> None:
        """设置任务配置"""
        self.task_configs[task_type] = config
        logger.info(f"Updated task config for {task_type}")
    
    async def _do_cleanup(self) -> None:
        """清理资源"""
        # 释放YOLO模型资源
        if hasattr(self, 'yolo_model'):
            self.yolo_model = None
        self.yolo_available = False
        logger.info("YOLO detector resources released")
