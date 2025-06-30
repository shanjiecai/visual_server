#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心接口定义模块
定义视频流处理服务中所有组件的抽象接口，确保高度可扩展性

设计原则：
- 简单明确：接口功能单一，职责清晰
- 易于扩展：支持插件化组件开发
- 异步优先：所有I/O操作支持异步执行
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import time


@dataclass
class FrameData:
    """视频帧数据容器
    
    承载视频帧的基本信息和处理过程中的元数据
    """
    frame_id: str                    # 帧唯一标识符
    timestamp: float                 # 帧时间戳（UNIX时间）
    raw_data: Any                   # 原始帧数据（如OpenCV的numpy数组）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展元数据
    
    def copy_with_metadata(self, new_metadata: Dict[str, Any]) -> 'FrameData':
        """创建带有新元数据的帧数据副本
        
        Args:
            new_metadata: 要更新的元数据
            
        Returns:
            新的FrameData实例，保持原始数据不变
        """
        updated_metadata = self.metadata.copy()
        updated_metadata.update(new_metadata)
        
        return FrameData(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            raw_data=self.raw_data,
            metadata=updated_metadata
        )


@dataclass
class ProcessingResult:
    """帧处理结果数据容器
    
    记录单个处理器对帧的处理结果
    """
    frame_id: str                    # 对应的帧ID
    processor_name: str              # 处理器名称
    result_data: Any                # 处理结果数据
    confidence: float                # 结果置信度 [0.0-1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)  # 处理元数据
    timestamp: float = field(default_factory=time.time)     # 处理完成时间戳
    frame_data: Optional['FrameData'] = None                # 关联的原始帧数据


class ProcessingStatus(Enum):
    """处理任务状态枚举"""
    PENDING = "pending"              # 等待处理
    PROCESSING = "processing"        # 正在处理
    COMPLETED = "completed"          # 处理完成
    FAILED = "failed"               # 处理失败


@dataclass
class ProcessingTask:
    """处理任务数据容器
    
    包含完整的任务信息和处理历史
    """
    task_id: str                     # 任务唯一标识符
    frame_data: FrameData           # 待处理的帧数据
    processing_results: List[ProcessingResult] = field(default_factory=list)  # 处理结果列表
    status: ProcessingStatus = ProcessingStatus.PENDING  # 任务状态
    created_at: float = field(default_factory=time.time)  # 创建时间戳
    updated_at: float = field(default_factory=time.time)  # 最后更新时间戳


# ==================== 核心组件接口 ====================

class IVideoSource(ABC):
    """视频源抽象接口
    
    定义视频数据的获取方式，支持文件、摄像头、流等多种来源
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化视频源
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    async def get_frame_stream(self) -> AsyncIterator[FrameData]:
        """获取视频帧流
        
        Yields:
            FrameData: 视频帧数据
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭视频源，释放资源"""
        pass
    
    @abstractmethod
    def is_active(self) -> bool:
        """检查视频源是否活跃
        
        Returns:
            bool: 视频源是否正在工作
        """
        pass


class IFrameProcessor(ABC):
    """帧处理器抽象接口
    
    定义对视频帧进行处理的通用方法
    """
    
    @property
    @abstractmethod
    def processor_name(self) -> str:
        """处理器名称，用于标识和日志记录"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化处理器
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    async def process(self, frame_data: FrameData) -> ProcessingResult:
        """处理单帧数据
        
        Args:
            frame_data: 待处理的帧数据
            
        Returns:
            ProcessingResult: 处理结果
        """
        pass
    
    async def batch_process(self, frame_batch: List[FrameData]) -> List[ProcessingResult]:
        """批量处理帧数据（默认实现为逐个处理）
        
        Args:
            frame_batch: 帧数据批次
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        results = []
        for frame_data in frame_batch:
            result = await self.process(frame_data)
            results.append(result)
        return results
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理处理器资源"""
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取处理器健康状态（默认实现）
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {"status": "healthy", "processor": self.processor_name}


class IPreprocessor(IFrameProcessor):
    """预处理器接口
    
    用于视频帧的预处理，如目标检测、特征提取等
    """
    pass


class IPostprocessor(ABC):
    """后处理器抽象接口
    
    对处理结果进行后处理，如发送通知、保存结果等
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化后处理器
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    async def execute(self, task: ProcessingTask) -> Dict[str, Any]:
        """执行后处理操作
        
        Args:
            task: 完成处理的任务
            
        Returns:
            Dict[str, Any]: 后处理结果
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理后处理器资源"""
        pass


class IMessageQueue(ABC):
    """消息队列抽象接口
    
    提供异步消息传递能力，解耦生产者和消费者
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化队列连接
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    async def put(self, item: Any, priority: int = 0) -> bool:
        """向队列添加消息
        
        Args:
            item: 消息内容
            priority: 消息优先级（数值越小优先级越高）
            
        Returns:
            bool: 是否成功添加
        """
        pass
    
    @abstractmethod
    async def get(self, timeout: Optional[float] = None) -> Any:
        """从队列获取消息
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            Any: 队列中的消息
            
        Raises:
            asyncio.TimeoutError: 获取超时
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭队列连接"""
        pass


class ILLMProcessor(ABC):
    """大语言模型处理器抽象接口
    
    处理需要LLM能力的任务，如图像理解、对话生成等
    """
    
    @abstractmethod
    async def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """处理LLM任务
        
        Args:
            task: 包含帧数据和预处理结果的任务
            
        Returns:
            Dict[str, Any]: LLM处理结果
        """
        pass
    
    async def health_check(self) -> bool:
        """健康检查（默认实现）
        
        Returns:
            bool: 服务是否健康
        """
        return True


# ==================== 辅助接口（简化版） ====================

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


class IConfigManager(ABC):
    """配置管理接口"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
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
