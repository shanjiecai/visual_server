#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像处理工具函数
"""

import base64
import cv2
import numpy as np
from typing import Any, Dict, Optional
from loguru import logger


def image_to_base64(image: np.ndarray, format: str = '.jpg', quality: int = 100) -> str:
    """
    将numpy图像数组转换为base64编码字符串
    
    Args:
        image: numpy图像数组
        format: 图像格式，默认为'.jpg'
        quality: JPEG质量(1-100)，默认100（最高质量不压缩）
        
    Returns:
        base64编码的图像字符串
    """
    try:
        if image is None or image.size == 0:
            logger.warning("Input image is None or empty")
            return ""
        
        # 设置编码参数
        encode_params = []
        if format.lower() in ['.jpg', '.jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format.lower() == '.png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 0 = 不压缩
        
        # 编码图像
        success, buffer = cv2.imencode(format, image, encode_params)
        if not success:
            logger.error(f"Failed to encode image to {format}")
            return ""
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
        
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return ""


def base64_to_image(base64_str: str) -> Optional[np.ndarray]:
    """
    将base64编码字符串转换为numpy图像数组
    
    Args:
        base64_str: base64编码的图像字符串
        
    Returns:
        numpy图像数组，失败时返回None
    """
    try:
        if not base64_str:
            logger.warning("Input base64 string is empty")
            return None
        
        # 解码base64
        image_data = base64.b64decode(base64_str)
        
        # 转换为numpy数组
        np_array = np.frombuffer(image_data, np.uint8)
        
        # 解码图像
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode base64 to image")
            return None
        
        return image
        
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None


def extract_image_from_data(data: Any) -> Optional[np.ndarray]:
    """
    从各种数据格式中提取图像
    
    Args:
        data: 包含图像的数据（可能是numpy数组、字典等）
        
    Returns:
        numpy图像数组，失败时返回None
    """
    try:
        # 检查是否已经是numpy数组
        if isinstance(data, np.ndarray):
            return data
        
        # 检查是否是字典格式（包含数据和元信息）
        if isinstance(data, dict):
            if "image_base64" in data:
                return base64_to_image(data["image_base64"])
            
            # 尝试从'data'字段获取
            if "data" in data:
                image_data = data["data"]
                if isinstance(image_data, np.ndarray):
                    return image_data
                
                # 如果data是字符串（模拟数据），创建空白图像
                if isinstance(image_data, str) and image_data.startswith("mock_frame_data_"):
                    width = data.get("width", 640)
                    height = data.get("height", 480)
                    return np.zeros((height, width, 3), dtype=np.uint8)
            
            # 尝试从'image'字段获取
            if "image" in data and isinstance(data["image"], np.ndarray):
                return data["image"]
            
            # 尝试从'raw_data'字段获取
            if "raw_data" in data:
                return extract_image_from_data(data["raw_data"])
                
        # 未能识别的格式
        logger.warning(f"Unrecognized image data format: {type(data)}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting image from data: {e}")
        return None


def ensure_image_base64(data: Dict[str, Any]) -> bool:
    """
    确保数据中包含image_base64字段，如果没有则尝试生成
    
    Args:
        data: 需要检查的数据字典
        
    Returns:
        bool: 是否成功确保image_base64存在
    """
    try:
        # 检查是否已经有image_base64字段
        image_base64_fields = [
            "image_base64", "images_base64", "image_data", "base64_image"
        ]
        
        for field in image_base64_fields:
            if field in data and data[field]:
                logger.debug(f"Found existing {field} field")
                return True
        
        # 如果没有base64字段，尝试从图像数据生成
        image = extract_image_from_data(data)
        if image is not None:
            image_base64 = image_to_base64(image)
            if image_base64:
                data["image_base64"] = image_base64
                logger.info("Generated image_base64 from image data")
                return True
        
        # 尝试从raw_data中生成
        raw_data = data.get("raw_data")
        if raw_data:
            image = extract_image_from_data(raw_data)
            if image is not None:
                image_base64 = image_to_base64(image)
                if image_base64:
                    data["image_base64"] = image_base64
                    logger.info("Generated image_base64 from raw_data")
                    return True
        
        logger.warning("Could not generate image_base64: no valid image data found")
        return False
        
    except Exception as e:
        logger.error(f"Error ensuring image_base64: {e}")
        return False


def validate_image_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证并增强图像数据，确保包含必要的字段
    
    Args:
        data: 输入数据字典
        
    Returns:
        Dict[str, Any]: 验证并增强后的数据字典
    """
    try:
        result = {
            "has_image_base64": False,
            "has_image_data": False,
            "image_size": None,
            "errors": []
        }
        
        # 检查base64字段
        image_base64_fields = [
            "image_base64", "images_base64", "image_data", "base64_image"
        ]
        
        for field in image_base64_fields:
            if field in data and data[field]:
                result["has_image_base64"] = True
                # 尝试验证base64格式
                try:
                    image = base64_to_image(data[field])
                    if image is not None:
                        result["image_size"] = image.shape
                        result["has_image_data"] = True
                    else:
                        result["errors"].append(f"Invalid base64 in field {field}")
                except Exception as e:
                    result["errors"].append(f"Error validating {field}: {str(e)}")
                break
        
        # 如果没有base64字段，检查原始图像数据
        if not result["has_image_base64"]:
            image = extract_image_from_data(data)
            if image is not None:
                result["has_image_data"] = True
                result["image_size"] = image.shape
            else:
                result["errors"].append("No valid image data found")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating image data: {e}")
        return {
            "has_image_base64": False,
            "has_image_data": False,
            "image_size": None,
            "errors": [f"Validation error: {str(e)}"]
        }


def resize_image(image: np.ndarray, max_width: int = 1024, max_height: int = 1024) -> np.ndarray:
    """
    调整图像大小，保持宽高比
    
    Args:
        image: 输入图像
        max_width: 最大宽度
        max_height: 最大高度
        
    Returns:
        调整大小后的图像
    """
    try:
        height, width = image.shape[:2]
        
        # 计算缩放比例
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return resized_image
        else:
            return image
            
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image


def compress_image_base64(base64_str: str, max_size_kb: Optional[int] = None, quality_step: int = 5) -> str:
    """
    压缩base64编码的图像到指定大小
    
    Args:
        base64_str: 原始base64字符串
        max_size_kb: 最大大小(KB)，None表示不限制大小
        quality_step: 质量递减步长
        
    Returns:
        压缩后的base64字符串
    """
    try:
        # 如果不限制大小，直接返回原始字符串
        if max_size_kb is None:
            return base64_str
            
        # 检查当前大小
        current_size_kb = len(base64_str.encode('utf-8')) / 1024
        if current_size_kb <= max_size_kb:
            return base64_str
        
        # 转换为图像
        image = base64_to_image(base64_str)
        if image is None:
            return base64_str
        
        # 优先通过调整图像尺寸来压缩，保持最高质量
        max_dimension = 1200
        while max_dimension > 200:
            resized_image = resize_image(image, max_dimension, max_dimension)
            compressed_base64 = image_to_base64(resized_image, quality=100)  # 保持最高质量
            if not compressed_base64:
                break
                
            size_kb = len(compressed_base64.encode('utf-8')) / 1024
            if size_kb <= max_size_kb:
                logger.info(f"Compressed image from {current_size_kb:.1f}KB to {size_kb:.1f}KB (size={max_dimension}x{max_dimension}, quality=100)")
                return compressed_base64
            
            max_dimension -= 100
        
        # 如果调整尺寸还不够，则适度降低质量（从95开始，不会太低）
        quality = 95
        resized_image = resize_image(image, 400, 400)  # 使用较小尺寸
        while quality > 70:  # 质量不会低于70，保证画质
            compressed_base64 = image_to_base64(resized_image, quality=quality)
            if not compressed_base64:
                break
                
            size_kb = len(compressed_base64.encode('utf-8')) / 1024
            if size_kb <= max_size_kb:
                logger.info(f"Compressed image from {current_size_kb:.1f}KB to {size_kb:.1f}KB (size=400x400, quality={quality})")
                return compressed_base64
            
            quality -= quality_step
        
        logger.warning(f"Could not compress image below {max_size_kb}KB while maintaining quality >= 70, returning original")
        return base64_str
        
    except Exception as e:
        logger.error(f"Error compressing image base64: {e}")
        return base64_str 