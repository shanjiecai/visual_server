#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
后处理器基类模块
"""

import asyncio
import time
from loguru import logger
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from core.interfaces import IPostProcessor, ProcessingResult, FrameData


class BasePostProcessor(IPostProcessor):
    """后处理器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor_name = config.get("processor_name", self.__class__.__name__)
        self.enabled = config.get("enabled", True)
        self.timeout = config.get("timeout", 30.0)

        self._is_initialized = False
        self._processed_count = 0
        self._total_processing_time = 0.0

    async def initialize(self) -> bool:
        """初始化后处理器"""
        try:
            logger.info(f"Initializing post processor: {self.processor_name}")
            result = await self._do_initialize()
            if result:
                self._is_initialized = True
                logger.info(f"Post processor {self.processor_name} initialized successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to initialize post processor {self.processor_name}: {e}")
            return False

    @abstractmethod
    async def _do_initialize(self) -> bool:
        """子类实现具体的初始化逻辑"""
        pass

    async def process(self, frame_data: FrameData, aggregated_results: Dict[str, Any]) -> Optional[ProcessingResult]:
        """处理数据"""
        if not self.enabled:
            logger.debug(f"Post processor {self.processor_name} is disabled")
            return None

        if not self._is_initialized:
            raise RuntimeError(f"Post processor {self.processor_name} not initialized")

        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self._do_process(frame_data, aggregated_results),
                timeout=self.timeout
            )

            processing_time = time.time() - start_time
            self._processed_count += 1
            self._total_processing_time += processing_time

            logger.debug(f"Post processor {self.processor_name} processed data in {processing_time:.3f}s")
            return result

        except asyncio.TimeoutError:
            logger.warning(f"Post processor {self.processor_name} timeout after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error in post processor {self.processor_name}: {e}")
            return None

    @abstractmethod
    async def _do_process(self, frame_data: FrameData, aggregated_results: Dict[str, Any]) -> ProcessingResult:
        """子类实现具体的处理逻辑"""
        pass

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            logger.info(f"Cleaning up post processor: {self.processor_name}")
            await self._do_cleanup()
            logger.info(f"Post processor {self.processor_name} cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up post processor {self.processor_name}: {e}")

    async def _do_cleanup(self) -> None:
        """子类实现具体的清理逻辑"""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_processing_time = (
            self._total_processing_time / self._processed_count
            if self._processed_count > 0 else 0
        )

        return {
            "processor_name": self.processor_name,
            "is_initialized": self._is_initialized,
            "enabled": self.enabled,
            "processed_count": self._processed_count,
            "total_processing_time": self._total_processing_time,
            "average_processing_time": avg_processing_time
        }
