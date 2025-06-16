#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
组件工厂模块
使用工厂模式创建各种组件实例，支持插件化扩展
"""

from typing import Dict, Type, Any, Optional
from abc import ABC, abstractmethod
import importlib
from loguru import logger

from .interfaces import (
    IVideoSource, IFrameProcessor, IPreprocessor, IPostprocessor,
    IMessageQueue, ILLMProcessor, ITaskScheduler, IWorkerPool,
    IResultAggregator, ICache, IConfigManager, IServiceRegistry,
    IMetricsCollector
)
from utils.config import config_manager


class IComponentFactory(ABC):
    """组件工厂接口"""

    @abstractmethod
    def create(self, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件实例"""
        pass

    @abstractmethod
    def register_component(self, component_type: str, component_class: Type) -> None:
        """注册组件类"""
        pass

    @abstractmethod
    def get_available_components(self) -> Dict[str, Type]:
        """获取可用组件列表"""
        pass


class ComponentFactory(IComponentFactory):
    """通用组件工厂实现"""

    def __init__(self):
        self._components: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}  # 单例模式缓存

    def create(self, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件实例"""
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")

        component_class = self._components[component_type]

        # 检查是否为单例
        if config.get("singleton", False):
            if component_type not in self._instances:
                self._instances[component_type] = component_class(**config.get("init_params", {}))
            return self._instances[component_type]

        return component_class(**config.get("init_params", {}))

    def register_component(self, component_type: str, component_class: Type) -> None:
        """注册组件类"""
        self._components[component_type] = component_class
        logger.info(f"Registered component: {component_type}")

    def get_available_components(self) -> Dict[str, Type]:
        """获取可用组件列表"""
        return self._components.copy()


class VideoSourceFactory(ComponentFactory):
    """视频源工厂"""

    def create(self, component_type: str, config: Dict[str, Any]) -> IVideoSource:
        instance = super().create(component_type, config)
        if not isinstance(instance, IVideoSource):
            raise TypeError(f"Component {component_type} must implement IVideoSource")
        return instance


class PreprocessorFactory(ComponentFactory):
    """预处理器工厂"""

    def create(self, component_type: str, config: Dict[str, Any]) -> IPreprocessor:
        instance = super().create(component_type, config)
        if not isinstance(instance, IPreprocessor):
            raise TypeError(f"Component {component_type} must implement IPreprocessor")
        return instance


class PostprocessorFactory(ComponentFactory):
    """后处理器工厂"""

    def create(self, component_type: str, config: Dict[str, Any]) -> IPostprocessor:
        instance = super().create(component_type, config)
        if not isinstance(instance, IPostprocessor):
            raise TypeError(f"Component {component_type} must implement IPostprocessor")
        return instance


class QueueFactory(ComponentFactory):
    """队列工厂"""

    def create(self, component_type: str, config: Dict[str, Any]) -> IMessageQueue:
        instance = super().create(component_type, config)
        if not isinstance(instance, IMessageQueue):
            raise TypeError(f"Component {component_type} must implement IMessageQueue")
        return instance


class LLMProcessorFactory(ComponentFactory):
    """大模型处理器工厂"""

    def create(self, component_type: str, config: Dict[str, Any]) -> ILLMProcessor:
        instance = super().create(component_type, config)
        if not isinstance(instance, ILLMProcessor):
            raise TypeError(f"Component {component_type} must implement ILLMProcessor")
        return instance


class PluginLoader:
    """插件加载器"""

    def __init__(self):
        pass

    def load_plugin(self, module_path: str, class_name: str) -> Type:
        """动态加载插件类"""
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            logger.info(f"Loaded plugin: {module_path}.{class_name}")
            return plugin_class
        except Exception as e:
            logger.error(f"Failed to load plugin {module_path}.{class_name}: {e}")
            raise

    def load_plugins_from_config(self, plugins_config: Dict[str, Dict[str, str]]) -> Dict[str, Type]:
        """从配置中加载插件"""
        loaded_plugins = {}
        for plugin_name, plugin_info in plugins_config.items():
            module_path = plugin_info.get("module")
            class_name = plugin_info.get("class")

            if module_path and class_name:
                try:
                    plugin_class = self.load_plugin(module_path, class_name)
                    loaded_plugins[plugin_name] = plugin_class
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name}: {e}")

        return loaded_plugins


class ServiceFactory:
    """服务工厂主类，管理所有类型的工厂"""

    def __init__(self):
        self._plugin_loader = PluginLoader()

        # 初始化各类型工厂
        self._factories: Dict[str, ComponentFactory] = {
            "video_source": VideoSourceFactory(),
            "preprocessor": PreprocessorFactory(),
            "postprocessor": PostprocessorFactory(),
            "queue": QueueFactory(),
            "llm_processor": LLMProcessorFactory(),
            "generic": ComponentFactory(),
        }

        self._initialize_factories()

    def _initialize_factories(self) -> None:
        """初始化工厂，加载插件和内置组件"""
        try:
            # 加载内置组件
            self._register_builtin_components()

            # 加载插件组件
            plugins_config = config_manager.get("plugins", {})
            if plugins_config:
                self._load_plugin_components(plugins_config)

        except Exception as e:
            logger.error(f"Failed to initialize factories: {e}")
            raise

    def _register_builtin_components(self) -> None:
        """注册内置组件"""
        # 注册视频源
        try:
            from producer.file_source import FileVideoSource
            from producer.camera_source import CameraVideoSource
            from producer.rtmp_source import RTMPVideoSource
            from producer.webrtc_source import WebRTCVideoSource

            video_factory = self._factories["video_source"]
            video_factory.register_component("file", FileVideoSource)
            video_factory.register_component("camera", CameraVideoSource)
            video_factory.register_component("rtmp", RTMPVideoSource)
            video_factory.register_component("webrtc", WebRTCVideoSource)

        except ImportError as e:
            logger.warning(f"Some video source components not available: {e}")

        # 注册预处理器
        try:
            from preprocessor.yolo_detector import YOLODetectorProcessor
            from preprocessor.similar_frame_filter import SimilarFrameFilterProcessor
            from preprocessor.memory_extractor import MemoryExtractorProcessor

            preprocessor_factory = self._factories["preprocessor"]
            preprocessor_factory.register_component("yolo_detector", YOLODetectorProcessor)
            preprocessor_factory.register_component("similar_frame_filter", SimilarFrameFilterProcessor)
            preprocessor_factory.register_component("memory_extractor", MemoryExtractorProcessor)

        except ImportError as e:
            logger.warning(f"Some preprocessor components not available: {e}")

        # 注册后处理器
        try:
            from postprocessor.dialogue_initiator import DialogueInitiatorPostprocessor
            from postprocessor.notification_sender import NotificationPostprocessor
            from postprocessor.greeting_printer import GreetingPrinterPostprocessor

            postprocessor_factory = self._factories["postprocessor"]
            postprocessor_factory.register_component("dialogue_initiator", DialogueInitiatorPostprocessor)
            postprocessor_factory.register_component("notification_sender", NotificationPostprocessor)
            postprocessor_factory.register_component("greeting_printer", GreetingPrinterPostprocessor)

        except ImportError as e:
            logger.warning(f"Some postprocessor components not available: {e}")

        # 注册消息队列
        try:
            from message_queue.memory_queue import InMemoryQueue
            from message_queue.kafka_queue import KafkaQueue

            queue_factory = self._factories["queue"]
            queue_factory.register_component("memory", InMemoryQueue)
            queue_factory.register_component("kafka", KafkaQueue)

        except ImportError as e:
            logger.warning(f"Some queue components not available: {e}")

        # 注册LLM处理器
        try:
            from consumer.openai_vlm import OpenAIVLMProcessor

            llm_factory = self._factories["llm_processor"]
            llm_factory.register_component("openai_vlm", OpenAIVLMProcessor)

        except ImportError as e:
            logger.warning(f"Some LLM processor components not available: {e}")

        # 注册缓存组件
        try:
            from utils.cache import InMemoryCache

            generic_factory = self._factories["generic"]
            generic_factory.register_component("cache", InMemoryCache)

        except ImportError as e:
            logger.warning(f"Cache components not available: {e}")

    def _load_plugin_components(self, plugins_config: Dict[str, Any]) -> None:
        """加载插件组件"""
        try:
            loaded_plugins = self._plugin_loader.load_plugins_from_config(plugins_config)
            
            for plugin_name, plugin_class in loaded_plugins.items():
                # 根据插件类型注册到相应工厂
                # 这里可以根据需要扩展插件分类逻辑
                self._factories["generic"].register_component(plugin_name, plugin_class)
                
        except Exception as e:
            logger.error(f"Error loading plugin components: {e}")

    def get_factory(self, factory_type: str) -> ComponentFactory:
        """获取指定类型的工厂"""
        if factory_type not in self._factories:
            raise ValueError(f"Unknown factory type: {factory_type}")
        return self._factories[factory_type]

    def create_component(self, factory_type: str, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件实例"""
        factory = self.get_factory(factory_type)
        return factory.create(component_type, config)

    def register_component(self, factory_type: str, component_type: str, component_class: Type) -> None:
        """注册组件类"""
        factory = self.get_factory(factory_type)
        factory.register_component(component_type, component_class)

    def get_available_components(self, factory_type: Optional[str] = None) -> Dict[str, Dict[str, Type]]:
        """获取可用组件列表"""
        if factory_type:
            factory = self.get_factory(factory_type)
            return {factory_type: factory.get_available_components()}
        
        return {
            name: factory.get_available_components()
            for name, factory in self._factories.items()
        }
