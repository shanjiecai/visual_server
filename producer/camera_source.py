#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
摄像头视频源实现
支持从本地摄像头捕获视频帧，按秒取帧
"""

import asyncio
import cv2
import time
from typing import Dict, Any, Optional, AsyncGenerator
from loguru import logger

from .base import BaseVideoSource
from core.interfaces import FrameData


class CameraVideoSource(BaseVideoSource):
    """摄像头视频源实现"""
    
    def __init__(self, camera_index: int = 0, fps: float = 1.0, resolution: tuple = (640, 480)):
        config = {
            "camera_index": camera_index,
            "resolution": resolution,
            "fps": fps
        }
        super().__init__(f"camera_{camera_index}", config)
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps  # 每秒取帧数，默认1帧/秒
        self._camera = None
        self._last_frame_time = 0
        self._frame_interval = 1.0 / self.fps  # 帧间间隔（秒）
        self._closing = False
    
    @property
    def is_initialized(self):
        """检查视频源是否初始化完成"""
        return self._is_active
    
    async def _do_initialize(self) -> bool:
        """初始化摄像头"""
        try:
            logger.info(f"Opening camera {self.camera_index} with resolution {self.resolution}")
            
            # 初始化OpenCV摄像头
            self._camera = cv2.VideoCapture(self.camera_index)
            
            if not self._camera.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # 设置摄像头参数
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._camera.set(cv2.CAP_PROP_FPS, 30)  # 摄像头原始帧率
            
            # 测试读取一帧
            ret, frame = self._camera.read()
            if not ret:
                logger.error("Failed to read test frame from camera")
                return False
            
            logger.info(f"Camera {self.camera_index} opened successfully")
            logger.info(f"Actual resolution: {frame.shape[1]}x{frame.shape[0]}")
            logger.info(f"Frame capture rate: {self.fps} FPS")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera {self.camera_index}: {e}")
            return False
    
    async def _get_next_frame(self) -> Optional[FrameData]:
        """从摄像头捕获帧（按设定FPS）"""
        try:
            if not self._camera or not self._camera.isOpened():
                return None
            
            current_time = time.time()
            
            # 检查是否到了下一帧的时间
            if current_time - self._last_frame_time < self._frame_interval:
                # 还没到取帧时间，等待一小段时间
                await asyncio.sleep(0.1)
                return None
            
            # 读取帧
            ret, frame = self._camera.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                return None
            
            # 更新最后取帧时间
            self._last_frame_time = current_time
            
            # 检查帧是否有效
            if frame is None or frame.size == 0:
                logger.warning("Received empty frame from camera")
                return None
            
            # 构建帧数据
            additional_metadata = {
                "camera_source": True,
                "camera_index": self.camera_index,
                "resolution": frame.shape[:2][::-1],  # (width, height)
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                "capture_time": current_time,
                "fps_setting": self.fps
            }
            
            return self._create_frame_data(frame, additional_metadata)
            
        except Exception as e:
            logger.error(f"Error capturing frame from camera: {e}")
            return None
    
    async def _do_close(self) -> None:
        """关闭摄像头"""
        if self._camera:
            self._camera.release()
            self._camera = None
            logger.info(f"Released camera {self.camera_index}")
    
    async def get_frame_stream(self) -> AsyncGenerator[FrameData, None]:
        """获取帧流"""
        while True:
            if not self.is_initialized or self._closing:
                break
                
            frame = await self._get_next_frame()
            if frame is not None:
                yield frame
            else:
                await asyncio.sleep(0.1)
    
    def set_fps(self, fps: float) -> None:
        """动态设置帧率"""
        self.fps = fps
        self._frame_interval = 1.0 / fps
        logger.info(f"Camera FPS updated to {fps}")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """获取摄像头信息"""
        if not self._camera or not self._camera.isOpened():
            return {}
        
        return {
            "index": self.camera_index,
            "width": int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._camera.get(cv2.CAP_PROP_FPS),
            "backend": self._camera.getBackendName()
        }
    
    def _create_frame_data(self, frame, additional_metadata: Dict[str, Any] = None) -> FrameData:
        """创建帧数据对象
        
        Args:
            frame: OpenCV格式的图像帧
            additional_metadata: 额外元数据
            
        Returns:
            FrameData: 帧数据对象
        """
        # 生成帧ID - 使用时间戳和摄像头ID的组合
        frame_id = f"cam{self.camera_index}_{time.time():.6f}"
        timestamp = time.time()
        
        # 基础元数据
        metadata = {
            "source_id": self.source_id,
            "camera_index": self.camera_index,
            "frame_count": self._frame_count,
            "resolution": self.resolution
        }
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
            
        # 创建并返回帧数据对象
        return FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            raw_data=frame,
            metadata=metadata
        )
