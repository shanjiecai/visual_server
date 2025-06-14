#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTMP视频源实现
支持从RTMP流获取视频帧
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional
from loguru import logger

from .base import BaseVideoSource
from core.interfaces import FrameData


class RTMPVideoSource(BaseVideoSource):
    """RTMP视频流源实现"""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        super().__init__(source_id, config)
        self.rtmp_url = config.get("rtmp_url")
        self.connection_timeout = config.get("connection_timeout", 10)
        self.reconnect_attempts = config.get("reconnect_attempts", 3)
        self._stream = None
        
        if not self.rtmp_url:
            raise ValueError("rtmp_url is required for RTMPVideoSource")
    
    async def _do_initialize(self) -> bool:
        """初始化RTMP流连接"""
        for attempt in range(self.reconnect_attempts):
            try:
                logger.info(f"Connecting to RTMP stream: {self.rtmp_url} (attempt {attempt + 1})")
                
                # 模拟RTMP连接
                await asyncio.sleep(0.5)  # 模拟连接延迟
                
                self._stream = {
                    "url": self.rtmp_url,
                    "connected": True,
                    "attempt": attempt + 1
                }
                
                logger.info(f"Successfully connected to RTMP stream: {self.rtmp_url}")
                return True
                
            except Exception as e:
                logger.warning(f"RTMP connection attempt {attempt + 1} failed: {e}")
                if attempt < self.reconnect_attempts - 1:
                    await asyncio.sleep(1)  # 重连延迟
        
        logger.error(f"Failed to connect to RTMP stream after {self.reconnect_attempts} attempts")
        return False
    
    async def _get_next_frame(self) -> Optional[FrameData]:
        """从RTMP流读取帧"""
        try:
            if not self._stream or not self._stream.get("connected"):
                return None
            
            # 模拟从RTMP流读取帧
            frame_data = {
                "width": 1920,
                "height": 1080,
                "channels": 3,
                "data": f"rtmp_frame_{self._frame_count}",
                "rtmp_url": self.rtmp_url
            }
            
            additional_metadata = {
                "rtmp_source": True,
                "stream_url": self.rtmp_url,
                "connection_attempt": self._stream.get("attempt", 1),
            }
            
            return self._create_frame_data(frame_data, additional_metadata)
            
        except Exception as e:
            logger.error(f"Error reading frame from RTMP stream: {e}")
            # 尝试重连
            if self._is_active:
                logger.info("Attempting to reconnect to RTMP stream...")
                await self._do_initialize()
            return None
    
    async def _do_close(self) -> None:
        """关闭RTMP流连接"""
        if self._stream:
            self._stream["connected"] = False
            self._stream = None
            logger.info(f"Disconnected from RTMP stream: {self.rtmp_url}")
    
    def _create_frame_data(self, frame_data: Dict[str, Any], additional_metadata: Dict[str, Any] = None) -> FrameData:
        """创建帧数据对象
        
        Args:
            frame_data: 帧数据字典
            additional_metadata: 额外元数据
            
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
            "rtmp_url": self.rtmp_url
        }
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
            
        # 创建并返回帧数据对象
        return FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            raw_data=frame_data,
            metadata=metadata
        ) 