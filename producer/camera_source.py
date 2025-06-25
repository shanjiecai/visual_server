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
    
    def __init__(self, camera_index: int = 0, fps: float = 1.0):
        config = {
            "camera_index": camera_index,
            "fps": fps
        }
        super().__init__(f"camera_{camera_index}", config)
        self.camera_index = camera_index
        self.fps = fps  # 每秒取帧数，默认1帧/秒
        self._camera = None
        self._last_frame_time = 0
        self._frame_interval = 1.0 / self.fps  # 帧间间隔（秒）
        self._closing = False
    
    @staticmethod
    def check_camera_availability(camera_index: int = 0) -> tuple[bool, str]:
        """检查摄像头是否可用"""
        try:
            # 尝试快速打开和关闭摄像头
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False, f"Cannot open camera {camera_index}"
            
            # 尝试读取一帧
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False, f"Camera {camera_index} opened but cannot read frame"
            
            if frame is None or frame.size == 0:
                return False, f"Camera {camera_index} returns empty frame"
            
            return True, f"Camera {camera_index} is available"
            
        except Exception as e:
            return False, f"Error checking camera {camera_index}: {e}"
    
    @property
    def is_initialized(self):
        """检查视频源是否初始化完成"""
        return self._is_active
    
    async def _do_initialize(self) -> bool:
        """初始化摄像头"""
        try:
            # 预先检查摄像头可用性
            is_available, message = self.check_camera_availability(self.camera_index)
            if not is_available:
                logger.error(f"Camera availability check failed: {message}")
                return False
            
            logger.info(f"Camera pre-check passed: {message}")
            logger.info(f"Opening camera {self.camera_index}")
            
            # 初始化OpenCV摄像头
            self._camera = cv2.VideoCapture(self.camera_index)
            
            if not self._camera.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # 设置摄像头参数
            self._camera.set(cv2.CAP_PROP_FPS, 30)  # 摄像头原始帧率
            
            # 设置缓冲区大小为1，减少延迟
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 给摄像头一些时间预热
            logger.info("Warming up camera...")
            await asyncio.sleep(0.5)
            
            # 尝试读取几帧来清空缓冲区和预热摄像头
            for i in range(3):
                ret, frame = self._camera.read()
                if ret:
                    break
                logger.warning(f"Warmup frame {i+1} failed, retrying...")
                await asyncio.sleep(0.2)
            
            # 最终测试读取一帧，带重试机制
            max_retries = 5
            for attempt in range(max_retries):
                ret, frame = self._camera.read()
                if ret and frame is not None and frame.size > 0:
                    logger.info(f"Camera {self.camera_index} opened successfully")
                    logger.info(f"Actual resolution: {frame.shape[1]}x{frame.shape[0]}")
                    logger.info(f"Frame capture rate: {self.fps} FPS")
                    logger.info(f"Camera backend: {self._camera.getBackendName()}")
                    return True
                else:
                    logger.warning(f"Test frame attempt {attempt + 1}/{max_retries} failed")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.3)
            
            # 如果所有重试都失败，记录详细错误信息
            logger.error("Failed to read test frame from camera after all retries")
            logger.error(f"Camera info: isOpened={self._camera.isOpened()}")
            logger.error(f"Camera backend: {self._camera.getBackendName()}")
            logger.error("Possible causes:")
            logger.error("  1. Camera is being used by another application (Zoom, Skype, etc.)")
            logger.error("  2. Insufficient camera permissions")
            logger.error("  3. Camera hardware issue")
            logger.error("  4. Camera driver problem")
            
            return False
            
        except Exception as e:
            logger.error(f"Error opening camera {self.camera_index}: {e}")
            return False
    
    async def _get_next_frame(self) -> Optional[FrameData]:
        """从摄像头捕获帧（按设定FPS）"""
        try:
            if not self._camera or not self._camera.isOpened():
                logger.warning("Camera is not opened or available")
                return None
            
            current_time = time.time()
            
            # 检查是否到了下一帧的时间
            if current_time - self._last_frame_time < self._frame_interval:
                # 还没到取帧时间，等待一小段时间
                await asyncio.sleep(0.1)
                return None
            
            # 读取帧，带简单重试
            ret, frame = None, None
            for attempt in range(3):  # 最多重试3次
                ret, frame = self._camera.read()
                if ret and frame is not None and frame.size > 0:
                    break
                if attempt < 2:  # 前两次失败时稍等一下
                    await asyncio.sleep(0.05)
            
            if not ret or frame is None or frame.size == 0:
                logger.warning("Failed to read valid frame from camera after retries")
                return None
            
            # 更新最后取帧时间
            self._last_frame_time = current_time
            
            # 构建额外元数据
            additional_metadata = {
                "camera_source": True,
                "camera_index": self.camera_index,
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                "capture_time": current_time,
                "fps_setting": self.fps
            }
            
            # 使用基类的通用方法创建FrameData
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
