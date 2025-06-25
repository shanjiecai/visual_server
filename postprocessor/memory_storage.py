#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记忆存储后处理器
将VLM检测的物体结果存储到视觉记忆中
"""

import base64
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger

from .base import BasePostProcessor
from core.interfaces import ProcessingTask
from utils.memory_storage import VisualMemoryStorage
from utils.image_utils import ensure_image_base64, extract_image_from_data, image_to_base64, validate_image_data


class MemoryStoragePostprocessor(BasePostProcessor):
    """记忆存储后处理器"""
    
    async def _do_initialize(self) -> bool:
        """初始化记忆存储"""
        try:
            # 记忆存储配置
            memory_config = self.config.get("memory_storage", {})
            self.memory_storage = VisualMemoryStorage(memory_config)
            
            # 目标物体列表
            self.target_objects = self.config.get("target_objects", [
                "手机", "桌子", "电脑", "笔", "水杯", "地板", "椅子", "花", "人"
            ])
            
            logger.info(f"Memory storage postprocessor initialized with {len(self.target_objects)} target objects")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize memory storage postprocessor: {e}")
            return False

    async def _do_execute(self, task: ProcessingTask) -> Dict[str, Any]:
        """执行记忆存储"""
        try:
            # 获取VLM处理结果
            if not task.processing_results:
                return {"status": "skipped", "reason": "no_vlm_results"}
            
            vlm_result = task.processing_results[-1]  # 获取最新的VLM结果
            vlm_content = vlm_result.result_data.get("content", "")
            
            if not vlm_content:
                return {"status": "skipped", "reason": "empty_vlm_content"}
            
            # 解析检测到的物体
            detected_objects = self._parse_detected_objects(vlm_content)
            
            if not detected_objects:
                logger.debug(f"No target objects detected in frame {task.frame_data.frame_id}")
                return {"status": "skipped", "reason": "no_target_objects"}
            
            # 提取图像数据
            image_base64 = self._extract_image_base64(task)
            if not image_base64:
                return {"status": "error", "reason": "failed_to_extract_image"}
            
            # 验证base64数据
            base64_info = self._validate_base64_data(image_base64)
            logger.info(f"Base64 validation for frame {task.frame_data.frame_id}: {base64_info}")
            
            if not base64_info["is_valid"]:
                logger.error(f"Invalid base64 data for frame {task.frame_data.frame_id}: {base64_info['error']}")
                return {"status": "error", "reason": f"invalid_base64: {base64_info['error']}"}
            
            # 存储到记忆中
            logger.info(f"Attempting to store frame {task.frame_data.frame_id} with categories {detected_objects}")
            logger.debug(f"Base64 length: {len(image_base64)} chars")
            
            success = self.memory_storage.store_frame(
                frame_id=task.frame_data.frame_id,
                timestamp=task.frame_data.timestamp,
                base64_data=image_base64,
                categories=detected_objects,
                metadata=task.frame_data.metadata
            )
            
            if success:
                logger.info(f"✅ Successfully stored frame {task.frame_data.frame_id} in memory: {detected_objects}")
                # 获取存储后的统计信息
                memory_stats = self.memory_storage.get_memory_stats()
                logger.info(f"Memory stats after storage: {memory_stats}")
                return {
                    "status": "success",
                    "detected_objects": detected_objects,
                    "memory_stats": memory_stats,
                    "base64_info": base64_info
                }
            else:
                logger.error(f"❌ Failed to store frame {task.frame_data.frame_id} to memory")
                return {"status": "error", "reason": "storage_failed"}
                
        except Exception as e:
            logger.error(f"Error in memory storage postprocessor: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "reason": str(e)}

    async def _do_cleanup(self) -> None:
        """清理资源"""
        logger.info("Memory storage postprocessor cleaned up")
    
    def _parse_detected_objects(self, vlm_content: str) -> List[str]:
        """解析VLM输出中的检测物体"""
        try:
            # 清理响应文本
            content = vlm_content.strip().strip('`').strip('json').strip()
            
            # 按逗号分割
            objects = [obj.strip() for obj in content.split(',') if obj.strip()]
            
            # 过滤只保留目标物体
            detected_objects = []
            for obj in objects:
                if obj in self.target_objects:
                    detected_objects.append(obj)
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error parsing detected objects: {e}")
            return []
    
    def _extract_image_base64(self, task: ProcessingTask) -> Optional[str]:
        """从任务中提取图像的base64编码"""
        try:
            # 首先尝试从raw_data中直接获取，使用更健壮的方法
            raw_data = task.frame_data.raw_data
            
            # 如果raw_data是字典，先尝试确保有image_base64字段
            if isinstance(raw_data, dict):
                # 创建一个副本来避免修改原始数据
                data_copy = raw_data.copy()
                
                # 使用utils中的函数确保有base64字段
                if ensure_image_base64(data_copy):
                    # 验证base64数据
                    validation_result = validate_image_data(data_copy)
                    if validation_result["has_image_base64"] and validation_result["has_image_data"]:
                        logger.info(f"Successfully extracted and validated image base64 for frame {task.frame_data.frame_id}")
                        logger.debug(f"Image size: {validation_result['image_size']}")
                        return data_copy["image_base64"]
                    else:
                        logger.warning(f"Image validation failed: {validation_result['errors']}")
            
            # 如果上面的方法失败，尝试直接从数据中提取图像
            image = extract_image_from_data(raw_data)
            if image is not None:
                # 转换为base64
                image_base64 = image_to_base64(image)
                if image_base64:
                    logger.info(f"Successfully converted image to base64 for frame {task.frame_data.frame_id}")
                    logger.debug(f"Image shape: {image.shape}")
                    return image_base64
                else:
                    logger.error(f"Failed to convert image to base64 for frame {task.frame_data.frame_id}")
            
            # 最后尝试从任务的其他可能位置获取图像
            for result in task.processing_results:
                if hasattr(result, 'frame_data') and result.frame_data:
                    image = extract_image_from_data(result.frame_data)
                    if image is not None:
                        image_base64 = image_to_base64(image)
                        if image_base64:
                            logger.info(f"Extracted image from processing result for frame {task.frame_data.frame_id}")
                            return image_base64
            
            logger.warning(f"Could not extract valid image from frame {task.frame_data.frame_id}")
            logger.debug(f"Raw data type: {type(raw_data)}")
            if isinstance(raw_data, dict):
                logger.debug(f"Raw data keys: {list(raw_data.keys())}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting image base64 for frame {task.frame_data.frame_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_memory_storage(self) -> VisualMemoryStorage:
        """获取记忆存储实例"""
        return self.memory_storage
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if hasattr(self, 'memory_storage'):
            return self.memory_storage.get_memory_stats()
        return {}
    
    def _validate_base64_data(self, base64_data: str) -> Dict[str, Any]:
        """验证base64数据的有效性"""
        try:
            if not base64_data:
                return {"is_valid": False, "error": "empty_base64"}
            
            # 检查基本格式
            if len(base64_data) < 100:  # 太短，不可能是图像
                return {"is_valid": False, "error": f"base64_too_short: {len(base64_data)} chars"}
            
            # 尝试解码base64
            try:
                decoded_data = base64.b64decode(base64_data)
            except Exception as e:
                return {"is_valid": False, "error": f"base64_decode_failed: {str(e)}"}
            
            # 尝试解码为图像
            try:
                np_array = np.frombuffer(decoded_data, np.uint8)
                image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                if image is None:
                    return {"is_valid": False, "error": "image_decode_failed"}
                
                height, width = image.shape[:2]
                if height < 10 or width < 10:  # 图像太小
                    return {"is_valid": False, "error": f"image_too_small: {width}x{height}"}
                
                return {
                    "is_valid": True,
                    "size_bytes": len(decoded_data),
                    "size_chars": len(base64_data),
                    "image_dimensions": f"{width}x{height}",
                    "image_channels": image.shape[2] if len(image.shape) == 3 else 1
                }
                
            except Exception as e:
                return {"is_valid": False, "error": f"image_validation_failed: {str(e)}"}
                
        except Exception as e:
            return {"is_valid": False, "error": f"validation_error: {str(e)}"} 