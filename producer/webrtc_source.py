#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebRTC视频源实现
支持从WebRTC流获取视频帧，基于aiortc库
"""

import asyncio
import threading
import time
import queue
from typing import Dict, Any, Optional
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCBundlePolicy
from loguru import logger

from .base import BaseVideoSource
from core.interfaces import FrameData


class WebRTCVideoSource(BaseVideoSource):
    """WebRTC视频源实现"""

    def __init__(self, source_id: str, config: Dict[str, Any]):
        super().__init__(source_id, config)
        self.url = config.get("url")
        self.max_frames_buffer = config.get("max_frames_buffer", 1)
        self.connection_timeout = config.get("connection_timeout", 10)

        if not self.url:
            raise ValueError("url is required for WebRTCVideoSource")

        # WebRTC相关属性
        self.pc = None
        self.session = None
        self.frame_queue = None
        self.webrtc_loop = None
        self.client_thread = None
        self.stop_event = None

    async def _do_initialize(self) -> bool:
        """初始化WebRTC连接"""
        try:
            logger.info(f"Initializing WebRTC connection to: {self.url}")

            # 创建线程安全的帧队列
            self.frame_queue = queue.Queue(maxsize=self.max_frames_buffer)
            self.stop_event = threading.Event()

            # 启动WebRTC客户端线程
            self.client_thread = threading.Thread(target=self._run_webrtc_client, daemon=True)
            self.client_thread.start()

            # 等待连接建立
            await asyncio.sleep(2.0)  # 给WebRTC连接一些时间建立

            if self._is_webrtc_connected():
                logger.info("WebRTC connection established successfully")
                return True
            else:
                logger.error("Failed to establish WebRTC connection")
                return False

        except Exception as e:
            logger.error(f"Error initializing WebRTC source: {e}")
            return False

    def _run_webrtc_client(self):
        """在后台线程中运行WebRTC客户端"""
        self.webrtc_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.webrtc_loop)

        try:
            self.webrtc_loop.run_until_complete(self._connect_webrtc())
        except Exception as e:
            logger.error(f"WebRTC client thread error: {e}")
        finally:
            if self.webrtc_loop.is_running():
                self.webrtc_loop.stop()
            self.webrtc_loop.close()
            self.webrtc_loop = None
            logger.info("WebRTC client thread exited")

    async def _connect_webrtc(self):
        """建立WebRTC连接"""
        self.session = aiohttp.ClientSession()

        config = RTCConfiguration(
            iceServers=None,
            bundlePolicy=RTCBundlePolicy.BALANCED
        )
        self.pc = RTCPeerConnection(config)

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_change():
            state = self.pc.iceConnectionState
            logger.info(f"ICE connection state: {state}")
            if state in ["failed", "disconnected", "closed"]:
                self.stop_event.set()

        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind != "video":
                return

            while self._is_active and not self.stop_event.is_set():
                try:
                    frame = await track.recv()
                    # 转换为numpy数组
                    img = frame.to_ndarray(format="bgr24")

                    # 将帧放入队列（如果队列满则替换最旧的帧）
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass

                    try:
                        self.frame_queue.put_nowait(img.copy())
                    except queue.Full:
                        pass  # 队列满，跳过此帧

                except Exception as e:
                    logger.error(f"Track receive error: {e}")
                    self.stop_event.set()
                    break

        # 添加收发器并创建Offer
        self.pc.addTransceiver("video", direction="recvonly")
        self.pc.addTransceiver("audio", direction="recvonly")

        await self.pc.setLocalDescription(await self.pc.createOffer())
        offer = self.pc.localDescription

        # 发送Offer并处理Answer
        try:
            async with self.session.post(
                    self.url,
                    data=offer.sdp,
                    headers={'Content-Type': 'application/sdp'},
                    timeout=aiohttp.ClientTimeout(total=self.connection_timeout)
            ) as resp:
                if resp.status not in [200, 201]:
                    raise Exception(f"Server response error: {resp.status}")

                answer_sdp = await resp.text()
                await self.pc.setRemoteDescription(RTCSessionDescription(
                    sdp=answer_sdp,
                    type='answer'
                ))
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            self.stop_event.set()
            return

        logger.info("WebRTC connection established")

        # 等待停止事件
        while self._is_active and not self.stop_event.is_set():
            await asyncio.sleep(0.5)

        # 清理资源
        if self.pc:
            await self.pc.close()
        if self.session:
            await self.session.close()
        logger.info("WebRTC connection closed")

    def _is_webrtc_connected(self) -> bool:
        """检查WebRTC连接状态"""
        return (self.client_thread and
                self.client_thread.is_alive() and
                not self.stop_event.is_set())

    async def _get_next_frame(self) -> Optional[FrameData]:
        """从WebRTC流获取下一帧"""
        try:
            if not self._is_webrtc_connected():
                return None

            # 非阻塞获取最新帧
            frame_data = None
            if not self.frame_queue.empty():
                try:
                    # 获取队列中最新的一帧，丢弃旧帧
                    while self.frame_queue.qsize() > 1:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break

                    frame_array = self.frame_queue.get_nowait()

                    # 转换为FrameData格式
                    height, width, channels = frame_array.shape
                    frame_data = {
                        "width": width,
                        "height": height,
                        "channels": channels,
                        "data": frame_array,  # 实际的numpy数组
                        "format": "BGR",
                        "webrtc_url": self.url
                    }

                    additional_metadata = {
                        "webrtc_source": True,
                        "stream_url": self.url,
                        "frame_size": f"{width}x{height}",
                        "data_type": "numpy_array"
                    }

                    # 使用基类的通用方法创建FrameData
                    return self._create_frame_data(frame_data, additional_metadata)

                except queue.Empty:
                    return None

            return None

        except Exception as e:
            logger.error(f"Error getting frame from WebRTC stream: {e}")
            return None

    async def _do_close(self) -> None:
        """关闭WebRTC连接"""
        try:
            logger.info("Closing WebRTC connection...")

            if self.stop_event:
                self.stop_event.set()

            if self.client_thread and self.client_thread.is_alive():
                self.client_thread.join(timeout=5.0)
                if self.client_thread.is_alive():
                    logger.warning("WebRTC client thread did not stop within 5 seconds")

            # 清理队列
            if self.frame_queue:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break

            logger.info("WebRTC connection closed")

        except Exception as e:
            logger.error(f"Error closing WebRTC connection: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取WebRTC特定的统计信息"""
        base_stats = super().get_statistics()

        webrtc_stats = {
            "webrtc_connected": self._is_webrtc_connected(),
            "frame_queue_size": self.frame_queue.qsize() if self.frame_queue else 0,
            "max_frames_buffer": self.max_frames_buffer,
            "stream_url": self.url,
        }

        base_stats.update(webrtc_stats)
        return base_stats