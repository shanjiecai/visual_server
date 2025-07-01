#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebRTC视频源实现
支持从WebRTC流获取视频帧，基于aiortc库
支持灵活切换、重连机制、连接状态监控
"""

import asyncio
import threading
import time
import queue
from typing import Dict, Any, Optional, List
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCBundlePolicy
from loguru import logger

from .base import BaseVideoSource
from core.interfaces import FrameData


class WebRTCConnectionState:
    """WebRTC连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CLOSED = "closed"


class WebRTCVideoSource(BaseVideoSource):
    """WebRTC视频源实现，支持灵活切换和重连"""

    def __init__(self, source_id: str, config: Dict[str, Any]):
        super().__init__(source_id, config)
        
        # 基础配置
        self.url = config.get("url")
        self.max_frames_buffer = config.get("max_frames_buffer", 1)
        self.connection_timeout = config.get("connection_timeout", 10)
        
        # 新增：灵活切换配置
        self.auto_reconnect = config.get("auto_reconnect", True)  # 自动重连
        self.max_reconnect_attempts = config.get("max_reconnect_attempts", 5)  # 最大重连次数
        self.reconnect_delay = config.get("reconnect_delay", 3.0)  # 重连延迟（秒）
        
        if not self.url:
            raise ValueError("url is required for WebRTCVideoSource")

        # WebRTC相关属性
        self.pc = None
        self.session = None
        self.frame_queue = None
        self.webrtc_loop = None
        self.client_thread = None
        self.stop_event = None
        
        # 新增：连接管理属性
        self.current_url = self.url
        self.connection_state = WebRTCConnectionState.DISCONNECTED
        self.reconnect_count = 0
        self.consecutive_failures = 0
        self.last_frame_time = 0
        self.connection_start_time = 0
        self.total_frames_received = 0
        self.connection_stats = {
            "ice_state": "unknown",
            "connection_state": "unknown",
            "total_connections": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "last_error": None,
            "frame_rate": 0.0
        }

    @staticmethod
    def check_webrtc_url_availability(url: str, timeout: float = 5.0) -> tuple[bool, str]:
        """检查WebRTC URL是否可用"""
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            
            if not parsed.scheme or not parsed.netloc:
                return False, f"Invalid URL format: {url}"
            
            if parsed.scheme not in ['http', 'https', 'ws', 'wss']:
                return False, f"Unsupported scheme: {parsed.scheme}"
            
            # TODO: 可以添加更详细的连通性检查，比如发送HTTP OPTIONS请求
            return True, f"URL format is valid: {url}"
            
        except Exception as e:
            return False, f"Error checking URL {url}: {e}"

    async def set_url(self, new_url: str, reconnect: bool = True) -> bool:
        """动态设置新的WebRTC URL"""
        try:
            # 检查URL有效性
            is_valid, message = self.check_webrtc_url_availability(new_url)
            if not is_valid:
                logger.error(f"Invalid WebRTC URL: {message}")
                return False
            
            logger.info(f"Setting new WebRTC URL: {new_url}")
            
            # 停止当前连接
            if self.connection_state != WebRTCConnectionState.DISCONNECTED:
                await self._disconnect()
            
            # 更新URL
            old_url = self.current_url
            self.current_url = new_url
            
            # 重新连接
            if reconnect and self._is_active:
                success = await self._reconnect()
                if not success:
                    # 恢复原URL
                    self.current_url = old_url
                    logger.error(f"Failed to connect to new URL, reverted to {old_url}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting WebRTC URL: {e}")
            return False

    async def _do_initialize(self) -> bool:
        """初始化WebRTC连接"""
        try:
            logger.info(f"Initializing WebRTC connection to: {self.current_url}")
            
            # 检查URL可用性
            is_available, message = self.check_webrtc_url_availability(self.current_url)
            if not is_available:
                logger.error(f"WebRTC URL check failed: {message}")
                return False
            
            logger.info(f"WebRTC URL pre-check passed: {message}")

            # 创建线程安全的帧队列
            self.frame_queue = queue.Queue(maxsize=self.max_frames_buffer)
            self.stop_event = threading.Event()

            # 重置统计信息
            self._reset_connection_stats()

            # 启动WebRTC客户端线程
            self.client_thread = threading.Thread(target=self._run_webrtc_client, daemon=True)
            self.client_thread.start()

            # 等待连接建立
            max_wait_time = self.connection_timeout
            wait_step = 0.5
            waited = 0
            
            while waited < max_wait_time:
                if self.connection_state == WebRTCConnectionState.CONNECTED:
                    logger.info("WebRTC connection established successfully")
                    return True
                elif self.connection_state == WebRTCConnectionState.FAILED:
                    logger.error("WebRTC connection failed during initialization")
                    return False
                
                await asyncio.sleep(wait_step)
                waited += wait_step

            logger.error("WebRTC connection timeout during initialization")
            return False

        except Exception as e:
            logger.error(f"Error initializing WebRTC source: {e}")
            return False

    def _reset_connection_stats(self):
        """重置连接统计信息"""
        self.connection_start_time = time.time()
        self.total_frames_received = 0
        self.last_frame_time = time.time()

    def _run_webrtc_client(self):
        """在后台线程中运行WebRTC客户端"""
        self.webrtc_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.webrtc_loop)

        try:
            self.webrtc_loop.run_until_complete(self._connect_webrtc())
        except Exception as e:
            logger.error(f"WebRTC client thread error: {e}")
            self.connection_state = WebRTCConnectionState.FAILED
            self.connection_stats["last_error"] = str(e)
        finally:
            if self.webrtc_loop.is_running():
                self.webrtc_loop.stop()
            self.webrtc_loop.close()
            self.webrtc_loop = None
            logger.info("WebRTC client thread exited")

    async def _connect_webrtc(self):
        """建立WebRTC连接（支持重连）"""
        while self._is_active and not self.stop_event.is_set():
            try:
                self.connection_state = WebRTCConnectionState.CONNECTING
                self.connection_stats["total_connections"] += 1
                
                await self._establish_connection()
                
                # 连接成功
                self.connection_state = WebRTCConnectionState.CONNECTED
                self.connection_stats["successful_connections"] += 1
                self.reconnect_count = 0
                self.consecutive_failures = 0
                
                logger.info(f"WebRTC connected successfully to {self.current_url}")
                
                # 保持连接并处理帧
                await self._maintain_connection()
                
            except Exception as e:
                logger.error(f"WebRTC connection error: {e}")
                self.connection_stats["failed_connections"] += 1
                self.connection_stats["last_error"] = str(e)
                self.consecutive_failures += 1
                
                # 清理连接
                await self._cleanup_connection()
                
                # 检查是否重连
                if self.auto_reconnect and self.reconnect_count < self.max_reconnect_attempts:
                    self.connection_state = WebRTCConnectionState.RECONNECTING
                    self.reconnect_count += 1
                    logger.info(f"Attempting reconnect {self.reconnect_count}/{self.max_reconnect_attempts} "
                              f"to {self.current_url} in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logger.error("Max reconnection attempts reached or auto-reconnect disabled")
                    self.connection_state = WebRTCConnectionState.FAILED
                    break

        # 最终清理
        await self._cleanup_connection()
        self.connection_state = WebRTCConnectionState.CLOSED

    async def _establish_connection(self):
        """建立单次WebRTC连接"""
        self.session = aiohttp.ClientSession()

        config = RTCConfiguration(
            iceServers=None,
            bundlePolicy=RTCBundlePolicy.BALANCED
        )
        self.pc = RTCPeerConnection(config)

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_change():
            state = self.pc.iceConnectionState
            self.connection_stats["ice_state"] = state
            logger.debug(f"ICE connection state: {state}")
            if state in ["failed", "disconnected", "closed"]:
                self.stop_event.set()

        @self.pc.on("connectionstatechange")
        async def on_connection_change():
            state = self.pc.connectionState
            self.connection_stats["connection_state"] = state
            logger.debug(f"Connection state: {state}")

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

                    # 更新统计信息
                    self.last_frame_time = time.time()
                    self.total_frames_received += 1

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
        async with self.session.post(
                self.current_url,
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

    async def _maintain_connection(self):
        """维护连接，监控健康状态"""
        last_health_check = time.time()
        
        while self._is_active and not self.stop_event.is_set():
            current_time = time.time()
            
            # 定期健康检查
            if current_time - last_health_check > 30.0:
                if not self._is_connection_healthy():
                    logger.warning("Connection health check failed, reconnecting...")
                    break
                last_health_check = current_time
            
            await asyncio.sleep(0.5)

    def _is_connection_healthy(self) -> bool:
        """检查连接健康状态"""
        if not self.pc:
            return False
        
        # 检查ICE连接状态
        if self.pc.iceConnectionState in ["failed", "disconnected", "closed"]:
            return False
        
        # 检查是否长时间没有收到帧
        if self.total_frames_received > 0:
            time_since_last_frame = time.time() - self.last_frame_time
            if time_since_last_frame > 30.0:
                logger.warning(f"No frames received for {time_since_last_frame:.1f}s")
                return False
        
        return True

    async def _cleanup_connection(self):
        """清理WebRTC连接资源"""
        if self.pc:
            try:
                await self.pc.close()
            except Exception as e:
                logger.debug(f"Error closing peer connection: {e}")
            self.pc = None
        
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.debug(f"Error closing HTTP session: {e}")
            self.session = None

    async def _disconnect(self):
        """断开当前连接"""
        if self.stop_event:
            self.stop_event.set()
        
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.join(timeout=3.0)

    async def _reconnect(self) -> bool:
        """重新连接"""
        try:
            # 停止当前连接
            await self._disconnect()
            
            # 重新初始化
            return await self._do_initialize()
            
        except Exception as e:
            logger.error(f"Error during reconnection: {e}")
            return False

    def _is_webrtc_connected(self) -> bool:
        """检查WebRTC连接状态"""
        return (self.connection_state == WebRTCConnectionState.CONNECTED and
                self.client_thread and
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
                        "webrtc_url": self.current_url
                    }

                    additional_metadata = {
                        "webrtc_source": True,
                        "stream_url": self.current_url,
                        "frame_size": f"{width}x{height}",
                        "data_type": "numpy_array",
                        "connection_state": self.connection_state,
                        "total_received": self.total_frames_received
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
            
            self.connection_state = WebRTCConnectionState.CLOSED

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

    def get_connection_info(self) -> Dict[str, Any]:
        """获取详细连接信息"""
        current_time = time.time()
        connection_duration = current_time - self.connection_start_time if self.connection_start_time > 0 else 0
        
        # 计算帧率
        frame_rate = 0.0
        if connection_duration > 0 and self.total_frames_received > 0:
            frame_rate = self.total_frames_received / connection_duration
        
        self.connection_stats["frame_rate"] = frame_rate
        
        return {
            "current_url": self.current_url,
            "connection_state": self.connection_state,
            "connection_duration": connection_duration,
            "total_frames_received": self.total_frames_received,
            "reconnect_count": self.reconnect_count,
            "consecutive_failures": self.consecutive_failures,
            "auto_reconnect": self.auto_reconnect,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "last_frame_time": self.last_frame_time,
            "time_since_last_frame": current_time - self.last_frame_time if self.last_frame_time > 0 else 0,
            **self.connection_stats
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取WebRTC特定的统计信息"""
        base_stats = super().get_statistics()

        webrtc_stats = {
            "webrtc_connected": self._is_webrtc_connected(),
            "frame_queue_size": self.frame_queue.qsize() if self.frame_queue else 0,
            "max_frames_buffer": self.max_frames_buffer,
            **self.get_connection_info()
        }

        base_stats.update(webrtc_stats)
        return base_stats

    # 新增：便捷方法
    async def force_reconnect(self) -> bool:
        """强制重连"""
        logger.info("Forcing WebRTC reconnection...")
        return await self._reconnect()

    def get_current_url_info(self) -> Dict[str, Any]:
        """获取当前URL信息"""
        return {
            "url": self.current_url,
            "connection_state": self.connection_state
        }

    async def test_url(self, url: str) -> bool:
        """测试指定URL的连通性"""
        try:
            is_valid, message = self.check_webrtc_url_availability(url)
            logger.info(f"URL test result for {url}: {message}")
            return is_valid
        except Exception as e:
            logger.error(f"Error testing URL {url}: {e}")
            return False