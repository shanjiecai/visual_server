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

from core.interfaces import IPostprocessor, ProcessingTask


class BasePostProcessor(IPostprocessor):
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

    async def execute(self, task: ProcessingTask) -> Dict[str, Any]:
        """执行后处理操作"""
        if not self.enabled:
            logger.debug(f"Post processor {self.processor_name} is disabled")
            return {"status": "disabled", "processor_name": self.processor_name}

        if not self._is_initialized:
            # 尝试自动初始化
            if not await self.initialize():
                raise RuntimeError(f"Post processor {self.processor_name} not initialized")

        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self._do_execute(task),
                timeout=self.timeout
            )

            processing_time = time.time() - start_time
            self._processed_count += 1
            self._total_processing_time += processing_time

            logger.debug(f"Post processor {self.processor_name} executed in {processing_time:.3f}s")
            
            # 确保返回结果包含必要字段
            if isinstance(result, dict):
                result.setdefault("processor_name", self.processor_name)
                result.setdefault("processing_time", processing_time)
            
            return result

        except asyncio.TimeoutError:
            logger.warning(f"Post processor {self.processor_name} timeout after {self.timeout}s")
            return {
                "status": "timeout", 
                "processor_name": self.processor_name,
                "timeout": self.timeout
            }
        except Exception as e:
            logger.error(f"Error in post processor {self.processor_name}: {e}")
            return {
                "status": "error", 
                "processor_name": self.processor_name,
                "error": str(e)
            }

    @abstractmethod
    async def _do_execute(self, task: ProcessingTask) -> Dict[str, Any]:
        """子类实现具体的执行逻辑"""
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
