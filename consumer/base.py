#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
消费者基类模块
包含大模型处理器、任务调度器、工作池和结果聚合器的基类实现
"""

import asyncio
import time
from loguru import logger
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from core.interfaces import (
    ILLMProcessor, ITaskScheduler, IWorkerPool, IResultAggregator,
    FrameData, ProcessingTask, ProcessingResult, ProcessingStatus
)


class BaseLLMProcessor(ILLMProcessor):
    """大模型处理器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "default")
        self.api_endpoint = config.get("api_endpoint")
        self.api_key = config.get("api_key")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 60.0)
        self.batch_size = config.get("batch_size", 1)
        
        self._is_initialized = False
        self._request_count = 0
        self._total_processing_time = 0.0
    
    async def initialize(self) -> bool:
        """初始化大模型处理器"""
        try:
            logger.info(f"Initializing LLM processor: {self.model_name}")
            result = await self._do_initialize()
            if result:
                self._is_initialized = True
                logger.info(f"LLM processor {self.model_name} initialized successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to initialize LLM processor {self.model_name}: {e}")
            return False
    
    @abstractmethod
    async def _do_initialize(self) -> bool:
        """子类实现具体的初始化逻辑"""
        pass
    
    async def process_task(self, task: ProcessingTask) -> ProcessingResult:
        """处理任务"""
        if not self._is_initialized:
            raise RuntimeError("LLM processor not initialized")
        
        start_time = time.time()
        
        try:
            # 重试机制
            last_exception = None
            for attempt in range(self.max_retries):
                try:
                    result = await asyncio.wait_for(
                        self._do_process_task(task),
                        timeout=self.timeout
                    )
                    
                    processing_time = time.time() - start_time
                    self._request_count += 1
                    self._total_processing_time += processing_time
                    
                    logger.debug(f"Processed task {task.task_id} in {processing_time:.3f}s")
                    return result
                    
                except asyncio.TimeoutError:
                    last_exception = "Request timeout"
                    logger.warning(f"Attempt {attempt + 1} timeout for task {task.task_id}")
                except Exception as e:
                    last_exception = str(e)
                    logger.warning(f"Attempt {attempt + 1} failed for task {task.task_id}: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
            
            raise RuntimeError(f"Task processing failed after {self.max_retries} attempts: {last_exception}")
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            raise
    
    @abstractmethod
    async def _do_process_task(self, task: ProcessingTask) -> ProcessingResult:
        """子类实现具体的任务处理逻辑"""
        pass
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            return await self._do_health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    @abstractmethod
    async def _do_health_check(self) -> bool:
        """子类实现具体的健康检查逻辑"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_processing_time = (
            self._total_processing_time / self._request_count 
            if self._request_count > 0 else 0
        )
        
        return {
            "model_name": self.model_name,
            "is_initialized": self._is_initialized,
            "request_count": self._request_count,
            "total_processing_time": self._total_processing_time,
            "average_processing_time": avg_processing_time
        }


class BaseTaskScheduler(ITaskScheduler):
    """任务调度器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_queue_size = config.get("max_queue_size", 1000)
        self.priority_enabled = config.get("priority_enabled", True)
        
        self._tasks: Dict[str, ProcessingTask] = {}
        self._pending_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._completed_tasks: Dict[str, ProcessingTask] = {}
    
    async def schedule_task(self, frame_data: FrameData, priority: int = 0) -> str:
        """调度任务"""
        task_id = f"task_{int(time.time() * 1000)}_{id(frame_data)}"
        
        task = ProcessingTask(
            task_id=task_id,
            frame_data=frame_data,
            processing_results=[],
            status=ProcessingStatus.PENDING,
            priority=priority,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self._tasks[task_id] = task
        
        try:
            await self._pending_queue.put(task)
            logger.debug(f"Scheduled task {task_id}")
            return task_id
        except asyncio.QueueFull:
            del self._tasks[task_id]
            raise RuntimeError("Task queue is full")
    
    async def get_next_task(self) -> Optional[ProcessingTask]:
        """获取下一个待处理任务"""
        try:
            task = await self._pending_queue.get()
            task.status = ProcessingStatus.PROCESSING
            task.updated_at = time.time()
            return task
        except asyncio.CancelledError:
            return None
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingStatus]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.status == ProcessingStatus.PENDING:
                task.status = ProcessingStatus.CANCELLED
                task.updated_at = time.time()
                return True
        return False
    
    def complete_task(self, task_id: str, result: ProcessingResult) -> None:
        """完成任务"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = ProcessingStatus.COMPLETED
            task.updated_at = time.time()
            task.processing_results.append(result)
            
            # 移动到已完成队列
            self._completed_tasks[task_id] = task
            del self._tasks[task_id]
    
    def cleanup_completed_tasks(self, before_timestamp: float) -> int:
        """清理已完成的任务"""
        count = 0
        to_remove = []
        
        for task_id, task in self._completed_tasks.items():
            if task.updated_at < before_timestamp:
                to_remove.append(task_id)
                count += 1
        
        for task_id in to_remove:
            del self._completed_tasks[task_id]
        
        return count


class BaseWorkerPool(IWorkerPool):
    """工作池基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_workers = config.get("max_workers", 4)
        self.queue_size = config.get("queue_size", 100)
        
        self._workers: List[asyncio.Task] = []
        self._task_queue = asyncio.Queue(maxsize=self.queue_size)
        self._shutdown_event = asyncio.Event()
        self._active_tasks = 0
        self._completed_tasks = 0
    
    async def start(self) -> None:
        """启动工作池"""
        logger.info(f"Starting worker pool with {self.max_workers} workers")
        
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._worker(f"worker_{i}"))
            self._workers.append(worker_task)
        
        logger.info("Worker pool started")
    
    async def _worker(self, worker_id: str) -> None:
        """工作器任务"""
        logger.debug(f"Worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # 获取任务
                task_func = await asyncio.wait_for(
                    self._task_queue.get(), 
                    timeout=1.0
                )
                
                self._active_tasks += 1
                
                try:
                    await task_func
                    self._completed_tasks += 1
                except Exception as e:
                    logger.error(f"Worker {worker_id} task failed: {e}")
                finally:
                    self._active_tasks -= 1
                    self._task_queue.task_done()
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def submit_task(self, task_coro) -> Any:
        """提交任务"""
        if self._shutdown_event.is_set():
            raise RuntimeError("Worker pool is shut down")
        
        future = asyncio.Future()
        
        async def task_wrapper():
            try:
                result = await task_coro
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        await self._task_queue.put(task_wrapper())
        return await future
    
    async def stop(self) -> None:
        """停止工作池"""
        logger.info("Stopping worker pool")
        
        # 设置关闭事件
        self._shutdown_event.set()
        
        # 等待所有任务完成
        await self._task_queue.join()
        
        # 取消所有工作器
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        logger.info("Worker pool stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """获取工作池状态"""
        return {
            "max_workers": self.max_workers,
            "active_workers": len([w for w in self._workers if not w.done()]),
            "active_tasks": self._active_tasks,
            "completed_tasks": self._completed_tasks,
            "queue_size": self._task_queue.qsize(),
            "is_shutdown": self._shutdown_event.is_set()
        }


class BaseResultAggregator(IResultAggregator):
    """结果聚合器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aggregation_strategy = config.get("strategy", "latest")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
        self._results_cache: Dict[str, List[ProcessingResult]] = {}
    
    async def aggregate_results(self, task_id: str, results: List[ProcessingResult]) -> Dict[str, Any]:
        """聚合处理结果"""
        if not results:
            return {"aggregated_result": None, "confidence": 0.0}
        
        # 缓存结果
        self._results_cache[task_id] = results
        
        try:
            if self.aggregation_strategy == "latest":
                return await self._aggregate_latest(results)
            elif self.aggregation_strategy == "highest_confidence":
                return await self._aggregate_highest_confidence(results)
            elif self.aggregation_strategy == "weighted_average":
                return await self._aggregate_weighted_average(results)
            else:
                logger.warning(f"Unknown aggregation strategy: {self.aggregation_strategy}")
                return await self._aggregate_latest(results)
                
        except Exception as e:
            logger.error(f"Error aggregating results for task {task_id}: {e}")
            return {"aggregated_result": None, "confidence": 0.0, "error": str(e)}
    
    async def _aggregate_latest(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """使用最新结果"""
        pass
    
    async def _aggregate_highest_confidence(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """使用置信度最高的结果"""
        pass
    
    async def _aggregate_weighted_average(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """使用加权平均"""
        pass
    
    async def should_trigger_postprocessing(self, aggregated_result: Dict[str, Any]) -> bool:
        """判断是否应该触发后处理"""
        confidence = aggregated_result.get("confidence", 0.0)
        return confidence >= self.confidence_threshold


# 具体实现类

class OpenAILLMProcessor(BaseLLMProcessor):
    """OpenAI大模型处理器实现"""
    
    async def _do_initialize(self) -> bool:
        """初始化OpenAI处理器"""
        pass
    
    async def _do_process_task(self, task: ProcessingTask) -> ProcessingResult:
        """处理任务"""
        pass
    
    async def _do_health_check(self) -> bool:
        """健康检查"""
        pass


class MockLLMProcessor(BaseLLMProcessor):
    """模拟大模型处理器实现"""
    
    async def _do_initialize(self) -> bool:
        """初始化模拟处理器"""
        pass
    
    async def _do_process_task(self, task: ProcessingTask) -> ProcessingResult:
        """模拟处理任务"""
        pass
    
    async def _do_health_check(self) -> bool:
        """模拟健康检查"""
        pass
