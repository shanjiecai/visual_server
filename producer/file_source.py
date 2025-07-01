#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件视频源实现
支持从视频文件读取帧数据，使用OpenCV真实读取mp4等视频文件
"""

import os
import cv2
import time
import asyncio
from typing import Dict, Any, Optional
from loguru import logger

from .base import BaseVideoSource
from core.interfaces import FrameData


class FileVideoSource(BaseVideoSource):
    """文件视频源实现"""
    
    def __init__(self, file_path: str, fps: float = 1.0, loop: bool = False):
        """
        初始化文件视频源
        
        Args:
            file_path: 视频文件路径
            fps: 读取帧率，默认1.0 FPS（每秒读取1帧）
            loop: 是否循环播放，默认False
        """
        config = {
            "file_path": file_path,
            "fps": fps,
            "loop": loop
        }
        super().__init__(f"file_{os.path.basename(file_path)}", config)
        self.file_path = file_path
        self.loop = loop
        self._video_capture = None
        self._current_frame_index = 0
        self._total_frames = 0
        self._original_fps = 0
        self._last_frame_time = 0
        self._frame_interval = 1.0 / fps
        
        if not self.file_path:
            raise ValueError("file_path is required for FileVideoSource")
    
    @staticmethod
    def check_file_availability(file_path: str) -> tuple[bool, str]:
        """检查视频文件是否可用"""
        try:
            if not os.path.exists(file_path):
                return False, f"Video file not found: {file_path}"
            
            if not os.path.isfile(file_path):
                return False, f"Path is not a file: {file_path}"
            
            # 检查文件扩展名
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in valid_extensions:
                return False, f"Unsupported file format: {file_ext}. Supported: {valid_extensions}"
            
            # 尝试快速打开和关闭视频文件
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False, f"Cannot open video file: {file_path}"
            
            # 尝试读取一帧验证文件完整性
            ret, frame = cap.read()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if not ret or frame is None:
                return False, f"Video file corrupted or cannot read frames: {file_path}"
            
            return True, f"Video file is valid: {os.path.basename(file_path)} ({width}x{height}, {total_frames} frames, {fps:.2f} FPS)"
            
        except Exception as e:
            return False, f"Error checking video file {file_path}: {e}"
    
    async def _do_initialize(self) -> bool:
        """初始化文件视频源"""
        try:
            # 预先检查文件可用性
            is_available, message = self.check_file_availability(self.file_path)
            if not is_available:
                logger.error(f"Video file check failed: {message}")
                return False
            
            logger.info(f"Video file pre-check passed: {message}")
            logger.info(f"Opening video file: {self.file_path}")
            
            # 使用OpenCV打开视频文件
            self._video_capture = cv2.VideoCapture(self.file_path)
            
            if not self._video_capture.isOpened():
                logger.error(f"Failed to open video file: {self.file_path}")
                return False
            
            # 获取视频信息
            self._total_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._original_fps = self._video_capture.get(cv2.CAP_PROP_FPS)
            width = int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 尝试读取第一帧验证
            ret, frame = self._video_capture.read()
            if not ret or frame is None:
                logger.error(f"Cannot read first frame from video file: {self.file_path}")
                return False
            
            # 重置到第一帧
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame_index = 0
            
            logger.info(f"Video file opened successfully:")
            logger.info(f"  文件: {os.path.basename(self.file_path)}")
            logger.info(f"  分辨率: {width}x{height}")
            logger.info(f"  总帧数: {self._total_frames}")
            logger.info(f"  原始帧率: {self._original_fps:.2f} FPS")
            logger.info(f"  读取帧率: {self.fps} FPS")
            logger.info(f"  循环播放: {self.loop}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening video file {self.file_path}: {e}")
            return False
    
    async def _get_next_frame(self) -> Optional[FrameData]:
        """从文件读取下一帧"""
        try:
            if not self._video_capture or not self._video_capture.isOpened():
                logger.warning("Video capture is not opened")
                return None
            
            current_time = time.time()
            
            # 检查是否到了下一帧的时间
            if current_time - self._last_frame_time < self._frame_interval:
                # 还没到取帧时间，等待一小段时间
                await asyncio.sleep(0.01)
                return None
            
            # 检查是否到达文件末尾
            if self._current_frame_index >= self._total_frames:
                if self.loop:
                    # 重置到文件开头
                    self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._current_frame_index = 0
                    logger.info(f"Looping video file: {os.path.basename(self.file_path)}")
                else:
                    logger.info(f"Reached end of video file: {os.path.basename(self.file_path)}")
                    return None
            
            # 读取帧
            ret, frame = self._video_capture.read()
            
            if not ret or frame is None:
                if self.loop and self._current_frame_index < self._total_frames:
                    # 可能是读取错误，尝试重置
                    logger.warning("Frame read failed, attempting to reset for loop")
                    self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._current_frame_index = 0
                    ret, frame = self._video_capture.read()
                
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame {self._current_frame_index} from video file")
                    return None
            
            # 更新时间和索引
            self._last_frame_time = current_time
            frame_index = self._current_frame_index
            self._current_frame_index += 1
            
            # 构建额外元数据
            additional_metadata = {
                "file_source": True,
                "file_path": self.file_path,
                "file_frame_index": frame_index,
                "total_frames": self._total_frames,
                "original_fps": self._original_fps,
                "loop_enabled": self.loop,
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                "progress": frame_index / self._total_frames if self._total_frames > 0 else 0
            }
            
            # 使用基类的通用方法创建FrameData
            return self._create_frame_data(frame, additional_metadata)
            
        except Exception as e:
            logger.error(f"Error reading frame from video file: {e}")
            return None
    
    async def _do_close(self) -> None:
        """关闭文件视频源"""
        if self._video_capture:
            self._video_capture.release()
            self._video_capture = None
            logger.info(f"Released video file: {os.path.basename(self.file_path)}")
    
    def get_video_info(self) -> Dict[str, Any]:
        """获取视频文件信息"""
        if not self._video_capture or not self._video_capture.isOpened():
            return {}
        
        return {
            "file_path": self.file_path,
            "filename": os.path.basename(self.file_path),
            "width": int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": self._total_frames,
            "current_frame": self._current_frame_index,
            "original_fps": self._original_fps,
            "read_fps": self.fps,
            "loop": self.loop,
            "progress": self._current_frame_index / self._total_frames if self._total_frames > 0 else 0,
            "duration_seconds": self._total_frames / self._original_fps if self._original_fps > 0 else 0
        }
    
    def set_fps(self, fps: float) -> None:
        """动态设置读取帧率"""
        self.fps = fps
        self._frame_interval = 1.0 / fps
        logger.info(f"Video file FPS updated to {fps}")
    
    def seek_to_frame(self, frame_index: int) -> bool:
        """跳转到指定帧"""
        try:
            if not self._video_capture or not self._video_capture.isOpened():
                return False
            
            frame_index = max(0, min(frame_index, self._total_frames - 1))
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._current_frame_index = frame_index
            logger.info(f"Seeked to frame {frame_index}")
            return True
        except Exception as e:
            logger.error(f"Error seeking to frame {frame_index}: {e}")
            return False
    
    def seek_to_time(self, seconds: float) -> bool:
        """跳转到指定时间（秒）"""
        if self._original_fps > 0:
            frame_index = int(seconds * self._original_fps)
            return self.seek_to_frame(frame_index)
        return False 