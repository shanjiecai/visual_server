#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预处理器基类模块
提供预处理器的抽象基类和通用功能实现
"""

import asyncio
import time
from loguru import logger
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import uuid

from core.interfaces import IPreprocessor, FrameData, ProcessingResult


class BasePreprocessor(IPreprocessor):
    """预处理器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = logger.bind(name=f"{__name__}.{self.__class__.__name__}")
        self._is_initialized = False
        self._processing_count = 0
        self._total_processing_time = 0
        
        # 配置参数
        self.batch_size = config.get("batch_size", 1)
        self.max_concurrent = config.get("max_concurrent", 4)
        self.timeout = config.get("timeout", 30.0)
        self.enabled = config.get("enabled", True)
        
        # 线程池用于CPU密集型操作
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
    
    @property
    @abstractmethod
    def processor_name(self) -> str:
        """处理器名称"""
        pass
    
    async def initialize(self) -> bool:
        """初始化处理器"""
        if self._is_initialized:
            return True
        
        try:
            self._logger.info(f"Initializing preprocessor: {self.processor_name}")
            
            if not self.enabled:
                self._logger.info(f"Preprocessor {self.processor_name} is disabled")
                self._is_initialized = True
                return True
            
            success = await self._do_initialize()
            if success:
                self._is_initialized = True
                self._logger.info(f"Preprocessor {self.processor_name} initialized successfully")
            else:
                self._logger.error(f"Failed to initialize preprocessor: {self.processor_name}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Error initializing preprocessor {self.processor_name}: {e}")
            return False
    
    @abstractmethod
    async def _do_initialize(self) -> bool:
        """子类实现的初始化逻辑"""
        pass
    
    async def process(self, frame_data: FrameData) -> ProcessingResult:
        """处理单帧数据"""
        if not self._is_initialized:
            raise RuntimeError(f"Preprocessor {self.processor_name} is not initialized")
        
        if not self.enabled:
            return self._create_disabled_result(frame_data)
        
        async with self._semaphore:
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    self._do_process(frame_data),
                    timeout=self.timeout
                )
                processing_time = time.time() - start_time
                
                self._processing_count += 1
                self._total_processing_time += processing_time
                
                # 记录处理时间
                if hasattr(result, "processing_time"):  # 如果已有此属性
                    result.processing_time = processing_time
                
                self._logger.debug(f"Processed frame {frame_data.frame_id} in {processing_time:.3f}s")
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Processing timeout for frame {frame_data.frame_id}")
                return self._create_error_result(frame_data, "Processing timeout")
            except Exception as e:
                processing_time = time.time() - start_time
                self._logger.error(f"Error processing frame {frame_data.frame_id}: {e}")
                
                return ProcessingResult(
                    frame_id=frame_data.frame_id,
                    processor_name=self.processor_name,
                    result_data={"error": str(e)},
                    confidence=0.0,
                    metadata={
                        "processing_time": processing_time,
                        "status": "error",
                        "error_type": type(e).__name__
                    },
                    timestamp=time.time()
                )
    
    @abstractmethod
    async def _do_process(self, frame_data: FrameData) -> ProcessingResult:
        """子类实现的处理逻辑"""
        pass
    
    async def batch_process(self, frame_batch: List[FrameData]) -> List[ProcessingResult]:
        """批量处理帧数据"""
        if not self._is_initialized:
            raise RuntimeError(f"Preprocessor {self.processor_name} is not initialized")
        
        if not self.enabled:
            return [self._create_disabled_result(frame) for frame in frame_batch]
        
        try:
            # 如果子类没有实现批量处理，则使用并发单帧处理
            if hasattr(self, '_do_batch_process'):
                return await self._do_batch_process(frame_batch)
            else:
                # 并发处理单帧
                tasks = [self.process(frame) for frame in frame_batch]
                return await asyncio.gather(*tasks)
                
        except Exception as e:
            self._logger.error(f"Error in batch processing: {e}")
            # 返回错误结果
            return [
                ProcessingResult(
                    frame_id=frame.frame_id,
                    processor_name=self.processor_name,
                    result_data={"error": str(e)},
                    confidence=0.0,
                    metadata={"status": "error", "error_type": type(e).__name__},
                    timestamp=time.time()
                )
                for frame in frame_batch
            ]
    
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            self._logger.info(f"Cleaning up preprocessor: {self.processor_name}")
            
            await self._do_cleanup()
            
            # 关闭线程池
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
            
            self._is_initialized = False
            self._logger.info(f"Preprocessor {self.processor_name} cleaned up successfully")
            
        except Exception as e:
            self._logger.error(f"Error cleaning up preprocessor {self.processor_name}: {e}")
    
    async def _do_cleanup(self) -> None:
        """子类实现的清理逻辑"""
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取处理器健康状态"""
        average_processing_time = (
            self._total_processing_time / self._processing_count
            if self._processing_count > 0 else 0
        )
        
        return {
            "processor_name": self.processor_name,
            "is_initialized": self._is_initialized,
            "enabled": self.enabled,
            "processing_count": self._processing_count,
            "average_processing_time": average_processing_time,
            "total_processing_time": self._total_processing_time,
            "max_concurrent": self.max_concurrent,
            "batch_size": self.batch_size,
        }
    
    def _create_disabled_result(self, frame_data: FrameData) -> ProcessingResult:
        """创建禁用状态的结果"""
        return ProcessingResult(
            frame_id=frame_data.frame_id,
            processor_name=self.processor_name,
            result_data={"disabled": True},
            confidence=1.0,
            metadata={"status": "disabled"},
            timestamp=time.time()
        )
    
    def _create_error_result(self, frame_data: FrameData, error_message: str) -> ProcessingResult:
        """创建错误状态的结果"""
        return ProcessingResult(
            frame_id=frame_data.frame_id,
            processor_name=self.processor_name,
            result_data={"error": error_message},
            confidence=0.0,
            metadata={
                "status": "error",
                "error_message": error_message
            },
            timestamp=time.time()
        )
    
    def _create_result(self, frame_data: FrameData, result_data: Any, 
                      confidence: float = 1.0, additional_metadata: Optional[Dict] = None) -> ProcessingResult:
        """创建处理结果"""
        metadata = {
            "status": "success",
            "processor_config": {
                "batch_size": self.batch_size,
                "max_concurrent": self.max_concurrent,
            }
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # 使用当前时间戳或从frame_data获取
        current_timestamp = time.time()
        
        return ProcessingResult(
            frame_id=frame_data.frame_id,
            processor_name=self.processor_name,
            result_data=result_data,
            confidence=confidence,
            metadata=metadata,
            timestamp=current_timestamp
        )
    
    async def _run_in_thread(self, func, *args, **kwargs):
        """在线程池中运行CPU密集型操作"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, func, *args, **kwargs) 