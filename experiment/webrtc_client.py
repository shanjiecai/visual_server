import asyncio
import threading
import time
import cv2
import numpy as np
from collections import deque
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCBundlePolicy
import aiohttp
import queue


class WebRTCStreamClient:
    def __init__(self, url, max_frames=1):
        self.url = url
        self.pc = None
        self.session = None
        self.running = False
        # 使用线程安全的队列存储帧
        self.frame_queue = queue.Queue(maxsize=max_frames)
        self.loop = None
        self.client_thread = None
        self.stop_event = threading.Event()

    def start(self):
        """启动客户端线程"""
        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        self.client_thread = threading.Thread(target=self._run_client, daemon=True)
        self.client_thread.start()
        print("客户端线程已启动")

    def _run_client(self):
        """在后台线程中运行客户端的事件循环"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._connect())
        except Exception as e:
            print(f"客户端线程异常: {str(e)}")
        finally:
            # 清理资源
            self.running = False
            if self.loop.is_running():
                self.loop.stop()
            self.loop.close()
            self.loop = None
            print("客户端线程已退出")

    async def _connect(self):
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
            print(f"[ICE状态] {state}")
            if state in ["failed", "disconnected", "closed"]:
                self.stop_event.set()

        @self.pc.on("track")
        async def on_track(track):
            print(f"接收到轨道: {track.kind}")
            if track.kind != "video":
                return

            while self.running and not self.stop_event.is_set():
                try:
                    frame = await track.recv()
                    img = frame.to_ndarray(format="bgr24")

                    # 将帧放入队列（如果队列满则替换最旧的帧）
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(img.copy())

                except Exception as e:
                    print(f"轨道接收错误: {str(e)}")
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
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status not in [200, 201]:
                    raise Exception(f"服务器响应错误: {resp.status}")

                answer_sdp = await resp.text()
                await self.pc.setRemoteDescription(RTCSessionDescription(
                    sdp=answer_sdp,
                    type='answer'
                ))
        except Exception as e:
            print(f"连接服务器失败: {str(e)}")
            self.stop_event.set()
            return

        print("WebRTC连接已建立")

        # 等待停止事件
        while self.running and not self.stop_event.is_set():
            await asyncio.sleep(0.5)

        # 清理资源
        if self.pc:
            await self.pc.close()
        if self.session:
            await self.session.close()
        print("WebRTC连接已关闭")

    def get_latest_frame(self):
        """获取最新的视频帧（如果没有帧则返回None）"""
        if not self.frame_queue.empty():
            try:
                # 获取队列中最新的一帧
                while self.frame_queue.qsize() > 1:
                    self.frame_queue.get_nowait()
                return self.frame_queue.get_nowait()
            except queue.Empty:
                return None
        return None

    def stop(self):
        """停止客户端"""
        if not self.running:
            return

        print("正在停止客户端...")
        self.running = False
        self.stop_event.set()

        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.join(timeout=5.0)
            if self.client_thread.is_alive():
                print("警告: 客户端线程未在5秒内停止")