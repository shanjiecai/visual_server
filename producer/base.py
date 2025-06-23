#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频源基类模块
定义视频源的通用接口和基础实现
"""

import asyncio
import time
from loguru import logger
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Optional, Union
import numpy as np

from core.interfaces import IVideoSource, FrameData


class BaseVideoSource(IVideoSource):
    """视频源基类"""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        self.source_id = source_id
        self.config = config
        self.fps = config.get("fps", 30.0)
        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 0.0
        self.max_frames = config.get("max_frames", None)
        
        self._is_active = False
        self._frame_count = 0
        self._start_time = None
    
    async def initialize(self) -> bool:
        """初始化视频源"""
        try:
            logger.info(f"Initializing video source: {self.source_id}")
            result = await self._do_initialize()
            if result:
                self._is_active = True
                logger.info(f"Video source {self.source_id} initialized successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to initialize video source {self.source_id}: {e}")
            return False
    
    @abstractmethod
    async def _do_initialize(self) -> bool:
        """子类实现具体的初始化逻辑"""
        pass
    
    async def get_frame_stream(self) -> AsyncGenerator[FrameData, None]:
        """获取视频帧流"""
        if not self._is_active:
            raise RuntimeError("Video source not initialized")
        
        self._start_time = time.time()
        last_frame_time = 0.0
        
        try:
            while self._is_active:
                if self.max_frames and self._frame_count >= self.max_frames:
                    break
                
                # 控制帧率
                current_time = time.time()
                elapsed = current_time - self._start_time - last_frame_time
                
                if elapsed < self.frame_interval:
                    await asyncio.sleep(self.frame_interval - elapsed)
                
                frame_data = await self._get_next_frame()
                if frame_data is None:
                    break
                
                self._frame_count += 1
                last_frame_time = time.time() - self._start_time
                
                yield frame_data
                
        except Exception as e:
            logger.error(f"Error in frame stream: {e}")
            raise
        finally:
            await self.close()
    
    @abstractmethod
    async def _get_next_frame(self) -> Optional[FrameData]:
        """获取下一帧"""
        pass
    
    def _create_frame_data(self, 
                          raw_data: Union[np.ndarray, Dict[str, Any]], 
                          additional_metadata: Optional[Dict[str, Any]] = None) -> FrameData:
        """创建帧数据对象
        
        Args:
            raw_data: 原始帧数据，可以是numpy数组或包含帧信息的字典
            additional_metadata: 额外的元数据
            
        Returns:
            FrameData: 帧数据对象
        """
        # 生成帧ID
        frame_id = f"{self.source_id}_{self._frame_count}_{time.time():.6f}"
        timestamp = time.time()
        
        # 基础元数据
        metadata = {
            "source_id": self.source_id,
            "frame_count": self._frame_count,
            "source_type": self.__class__.__name__,
            "fps": self.fps,
            "capture_timestamp": timestamp,
        }
        
        # 如果raw_data是numpy数组，转换为标准字典格式
        if isinstance(raw_data, np.ndarray):
            height, width = raw_data.shape[:2]
            channels = raw_data.shape[2] if len(raw_data.shape) > 2 else 1
            
            frame_dict = {
                "width": width,
                "height": height,
                "channels": channels,
                "data": raw_data,
                "format": "numpy_array",
                "dtype": str(raw_data.dtype)
            }
        else:
            # raw_data已经是字典格式
            frame_dict = raw_data
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # 创建并返回帧数据对象
        return FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            raw_data=frame_dict,
            metadata=metadata
        )
    
    async def close(self) -> None:
        """关闭视频源"""
        self._is_active = False
        await self._do_close()
        logger.info(f"Video source {self.source_id} closed")
    
    @abstractmethod
    async def _do_close(self) -> None:
        """子类实现具体的关闭逻辑"""
        pass
    
    def is_active(self) -> bool:
        """检查视频源是否活跃"""
        return self._is_active
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        elapsed_time = current_time - self._start_time if self._start_time else 0
        actual_fps = self._frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "source_id": self.source_id,
            "frame_count": self._frame_count,
            "elapsed_time": elapsed_time,
            "actual_fps": actual_fps,
            "target_fps": self.fps,
            "is_active": self._is_active
        }

