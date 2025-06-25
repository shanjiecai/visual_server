#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO检测器实现
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
from utils.image_utils import image_to_base64, extract_image_from_data


# YOLO模型配置
DEFAULT_YOLO_MODEL_PATH = "models/yoloe-v8l-seg.pt"
DEFAULT_YOLO_DEVICE = "cpu"  # 如果没有GPU，会自动回退到CPU
DEFAULT_DETECTION_CLASSES = ["person", "chair", "table", "cup", "phone", "laptop"]


class YOLODetectorProcessor(BasePreprocessor):
    """YOLO检测处理器实现 - 纯预处理功能"""
    
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
            
            # 是否需要发送到下游处理
            self.enable_downstream = self.config.get("enable_downstream", True)
            
            # 加载YOLO模型（简化实现）
            await self._load_model()
            
            logger.info("YOLO检测器初始化完成 - 纯预处理模式")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            return False
    
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
        """执行目标检测 - 只负责检测，不包含下游配置"""
        try:
            # 从帧数据中提取图像
            raw_data = frame_data.raw_data
            image = extract_image_from_data(raw_data)
            
            if image is None:
                raise ValueError("无法从帧数据中提取图像")
            
            # 执行检测（简化实现）
            detections = await self._detect_objects(image)
            
            # 提取检测到的类别
            categories = self._extract_categories(detections)
            
            # 转换图像为base64（用于下游处理）
            image_base64 = None
            if self.enable_downstream:
                image_base64 = image_to_base64(image)
            
            # 构建结果数据 - 只包含检测结果，不包含任务配置
            result_data = {
                "detections": detections,
                "categories": categories,
                "image_base64": image_base64,
                "detection_info": {
                    "model_path": self.model_path,
                    "confidence_threshold": self.confidence_threshold,
                    "nms_threshold": self.nms_threshold,
                    "device": self.device,
                    "detected_objects": categories
                },
                "frame_id": frame_data.frame_id,
                "timestamp": frame_data.timestamp,
                "metadata": {
                    "source": "yolo_detector",
                    "detection_count": len(detections),
                    "primary_objects": categories[:3] if categories else [],
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
                    "enable_downstream": self.enable_downstream,
                    "processor_type": "detection"  # 标识这是检测类型的预处理器
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"YOLO detection failed: {e}")
    
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
    
    async def _do_cleanup(self) -> None:
        """清理资源"""
        # 释放YOLO模型资源
        if hasattr(self, 'yolo_model'):
            self.yolo_model = None
        self.yolo_available = False
        logger.info("YOLO detector resources released")
