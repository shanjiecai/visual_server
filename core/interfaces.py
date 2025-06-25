#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心接口定义模块
定义视频流处理服务中所有组件的抽象接口，确保高度可扩展性
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum


@dataclass
class FrameData:
    """视频帧数据类"""
    frame_id: str
    timestamp: float
    raw_data: Any  # 原始帧数据
    metadata: Dict[str, Any]  # 元数据
    
    def copy_with_metadata(self, new_metadata: Dict[str, Any]) -> 'FrameData':
        """创建一个带有新元数据的帧数据副本
        
        Args:
            new_metadata: 新的元数据字典
            
        Returns:
            返回一个新的FrameData实例，只有元数据被更新
        """
        return FrameData(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            raw_data=self.raw_data,
            metadata=new_metadata
        )


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    frame_id: str
    processor_name: str
    result_data: Any
    confidence: float
    metadata: Dict[str, Any]
    timestamp: float = 0.0  # 处理时间戳，默认为0.0
    frame_data: Optional['FrameData'] = None  # 传递原始图片数据


class ProcessingStatus(Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingTask:
    """处理任务数据类"""
    task_id: str
    frame_data: FrameData
    processing_results: List[ProcessingResult]
    status: ProcessingStatus
    created_at: float
    updated_at: float


class IVideoSource(ABC):
    """视频源接口"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化视频源"""
        pass
    
    @abstractmethod
    async def get_frame_stream(self) -> AsyncIterator[FrameData]:
        """获取视频帧流"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭视频源"""
        pass
    
    @abstractmethod
    def is_active(self) -> bool:
        """检查视频源是否活跃"""
        pass


class IFrameProcessor(ABC):
    """帧处理器接口"""
    
    @property
    @abstractmethod
    def processor_name(self) -> str:
        """处理器名称"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化处理器"""
        pass
    
    @abstractmethod
    async def process(self, frame_data: FrameData) -> ProcessingResult:
        """处理单帧数据"""
        pass
    
    @abstractmethod
    async def batch_process(self, frame_batch: List[FrameData]) -> List[ProcessingResult]:
        """批量处理帧数据"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """获取处理器健康状态"""
        pass


class IPreprocessor(IFrameProcessor):
    """预处理器接口"""
    pass


class IPostprocessor(ABC):
    """后处理器接口"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化后处理器"""
        pass
    
    @abstractmethod
    async def execute(self, task: ProcessingTask) -> Dict[str, Any]:
        """执行后处理操作"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理后处理器资源"""
        pass


class IMessageQueue(ABC):
    """消息队列接口"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化队列"""
        pass
    
    @abstractmethod
    async def put(self, item: Any, priority: int = 0) -> bool:
        """向队列中添加项目"""
        pass
    
    @abstractmethod
    async def get(self, timeout: Optional[float] = None) -> Any:
        """从队列中获取项目"""
        pass
    
    @abstractmethod
    async def put_batch(self, items: List[Any]) -> bool:
        """批量添加项目"""
        pass
    
    @abstractmethod
    async def get_batch(self, batch_size: int, timeout: Optional[float] = None) -> List[Any]:
        """批量获取项目"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """获取队列大小"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭队列"""
        pass


class ILLMProcessor(ABC):
    """大模型处理器接口"""
    
    @abstractmethod
    async def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """处理任务"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


class ITaskScheduler(ABC):
    """任务调度器接口"""
    
    @abstractmethod
    async def schedule_task(self, frame_data: FrameData) -> str:
        """调度任务"""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> ProcessingStatus:
        """获取任务状态"""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        pass


class IWorkerPool(ABC):
    """工作池接口"""
    
    @abstractmethod
    async def start(self) -> None:
        """启动工作池"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """停止工作池"""
        pass
    
    @abstractmethod
    async def submit_task(self, task: Any) -> Any:
        """提交任务"""
        pass
    
    @abstractmethod
    def get_worker_count(self) -> int:
        """获取工作者数量"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取工作池状态"""
        pass


class IResultAggregator(ABC):
    """结果聚合器接口"""
    
    @abstractmethod
    async def aggregate_results(self, task_id: str, results: List[ProcessingResult]) -> Dict[str, Any]:
        """聚合处理结果"""
        pass


class ICache(ABC):
    """缓存接口"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass


class IConfigManager(ABC):
    """配置管理器接口"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        pass
    
    @abstractmethod
    def reload(self) -> bool:
        """重新加载配置"""
        pass


class IServiceRegistry(ABC):
    """服务注册器接口"""
    
    @abstractmethod
    def register(self, service_name: str, service_instance: Any) -> None:
        """注册服务"""
        pass
    
    @abstractmethod
    def get_service(self, service_name: str) -> Optional[Any]:
        """获取服务"""
        pass
    
    @abstractmethod
    def unregister(self, service_name: str) -> bool:
        """注销服务"""
        pass


class IMetricsCollector(ABC):
    """指标收集器接口"""
    
    @abstractmethod
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""
        pass
    
    @abstractmethod
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """增加计数器"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        pass
