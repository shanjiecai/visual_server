#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
组件工厂模块
使用工厂模式创建各种组件实例，支持插件化扩展

设计原则：
- 简单实用：专注核心功能，避免过度设计
- 类型安全：严格的类型检查和错误处理
- 配置驱动：通过配置文件管理组件注册
"""

from typing import Dict, Type, Any, Optional
from abc import ABC, abstractmethod
import importlib
from loguru import logger

from .interfaces import (
    IVideoSource, IPreprocessor, IPostprocessor,
    IMessageQueue, ILLMProcessor
)


class ComponentFactory:
    """通用组件工厂
    
    负责创建和管理各种类型的组件实例
    支持单例模式和插件动态加载
    """

    def __init__(self):
        # 组件类型注册表：组件名 -> 组件类
        self._components: Dict[str, Type] = {}
        # 单例实例缓存：组件名 -> 实例
        self._singletons: Dict[str, Any] = {}

    def register_component(self, component_name: str, component_class: Type) -> None:
        """注册组件类
        
        Args:
            component_name: 组件名称，用于配置中引用
            component_class: 组件类，必须实现对应接口
        """
        self._components[component_name] = component_class
        logger.info(f"注册组件: {component_name} -> {component_class.__name__}")

    def create_component(self, component_name: str, config: Dict[str, Any]) -> Any:
        """创建组件实例
        
        Args:
            component_name: 组件名称
            config: 组件配置，包含init_params和singleton选项
            
        Returns:
            组件实例
            
        Raises:
            ValueError: 组件未注册
            TypeError: 组件类型不匹配
        """
        if component_name not in self._components:
            available = list(self._components.keys())
            raise ValueError(f"组件 '{component_name}' 未注册。可用组件: {available}")

        component_class = self._components[component_name]
        
        # 检查是否为单例模式
        if config.get("singleton", False):
            if component_name not in self._singletons:
                init_params = config.get("init_params", {})
                self._singletons[component_name] = component_class(**init_params)
                logger.debug(f"创建单例组件: {component_name}")
            return self._singletons[component_name]

        # 创建新实例
        init_params = config.get("init_params", {})
        instance = component_class(**init_params)
        logger.debug(f"创建组件实例: {component_name}")
        return instance

    def get_available_components(self) -> Dict[str, Type]:
        """获取所有已注册的组件
        
        Returns:
            组件名称到组件类的映射
        """
        return self._components.copy()

    def load_plugin(self, module_path: str, class_name: str, component_name: str) -> None:
        """动态加载插件组件
        
        Args:
            module_path: 模块路径，如 'plugins.custom_detector'
            class_name: 类名，如 'CustomDetector'
            component_name: 注册的组件名称
            
        Raises:
            ImportError: 模块或类无法导入
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            self.register_component(component_name, plugin_class)
            logger.info(f"加载插件: {component_name} 来自 {module_path}.{class_name}")
        except Exception as e:
            logger.error(f"加载插件失败 {component_name}: {e}")
            raise


class TypedComponentFactory(ComponentFactory):
    """类型安全的组件工厂基类
    
    为特定接口类型提供类型检查
    """
    
    def __init__(self, expected_interface: Type):
        super().__init__()
        self.expected_interface = expected_interface

    def create_component(self, component_name: str, config: Dict[str, Any]) -> Any:
        """创建组件实例并验证类型
        
        Args:
            component_name: 组件名称
            config: 组件配置
            
        Returns:
            符合接口类型的组件实例
            
        Raises:
            TypeError: 组件未实现期望的接口
        """
        instance = super().create_component(component_name, config)
        
        # 类型检查
        if not isinstance(instance, self.expected_interface):
            raise TypeError(
                f"组件 '{component_name}' 未实现接口 {self.expected_interface.__name__}"
            )
        
        return instance


class ServiceFactory:
    """服务工厂主类
    
    管理所有类型的组件工厂，提供统一的创建接口
    """

    def __init__(self):
        # 初始化各类型的组件工厂
        self._factories: Dict[str, ComponentFactory] = {
            "video_source": TypedComponentFactory(IVideoSource),
            "preprocessor": TypedComponentFactory(IPreprocessor),
            "postprocessor": TypedComponentFactory(IPostprocessor),
            "message_queue": TypedComponentFactory(IMessageQueue),
            "llm_processor": TypedComponentFactory(ILLMProcessor),
        }
        
        # 注册内置组件
        self._register_builtin_components()

    def _register_builtin_components(self) -> None:
        """注册内置组件"""
        try:
            # 注册视频源组件
            self._register_video_sources()
            # 注册预处理器组件
            self._register_preprocessors()
            # 注册后处理器组件
            self._register_postprocessors()
            # 注册消息队列组件
            self._register_message_queues()
            # 注册LLM处理器组件
            self._register_llm_processors()
            
            logger.info("内置组件注册完成")
            
        except Exception as e:
            logger.error(f"注册内置组件失败: {e}")
            raise

    def _register_video_sources(self) -> None:
        """注册视频源组件"""
        try:
            from producer.file_source import FileVideoSource
            from producer.camera_source import CameraVideoSource
            
            video_factory = self._factories["video_source"]
            video_factory.register_component("file", FileVideoSource)
            video_factory.register_component("camera", CameraVideoSource)
            
            # 可选组件（不强制依赖）
            try:
                from producer.rtmp_source import RTMPVideoSource
                video_factory.register_component("rtmp", RTMPVideoSource)
            except ImportError:
                logger.warning("RTMP视频源组件不可用，跳过注册")
                
            try:
                from producer.webrtc_source import WebRTCVideoSource
                video_factory.register_component("webrtc", WebRTCVideoSource)
            except ImportError:
                logger.warning("WebRTC视频源组件不可用，跳过注册")
                
        except ImportError as e:
            logger.error(f"注册视频源组件失败: {e}")

    def _register_preprocessors(self) -> None:
        """注册预处理器组件"""
        try:
            from preprocessor.similar_frame_filter import SimilarFrameFilterProcessor
            from preprocessor.yolo_detector import YOLODetectorProcessor
            
            preprocessor_factory = self._factories["preprocessor"]
            preprocessor_factory.register_component("similar_frame_filter", SimilarFrameFilterProcessor)
            preprocessor_factory.register_component("yolo_detector", YOLODetectorProcessor)
            
        except ImportError as e:
            logger.error(f"注册预处理器组件失败: {e}")

    def _register_postprocessors(self) -> None:
        """注册后处理器组件"""
        try:
            from postprocessor.dialogue_initiator import DialogueInitiatorProcessor
            
            postprocessor_factory = self._factories["postprocessor"]
            postprocessor_factory.register_component("dialogue_initiator", DialogueInitiatorProcessor)
            
        except ImportError as e:
            logger.error(f"注册后处理器组件失败: {e}")

    def _register_message_queues(self) -> None:
        """注册消息队列组件"""
        try:
            from message_queue.memory_queue import MemoryQueue
            
            queue_factory = self._factories["message_queue"]
            queue_factory.register_component("memory", MemoryQueue)
            
            # 可选的Kafka队列
            try:
                from message_queue.kafka_queue import KafkaQueue
                queue_factory.register_component("kafka", KafkaQueue)
            except ImportError:
                logger.warning("Kafka队列组件不可用，跳过注册")
                
        except ImportError as e:
            logger.error(f"注册消息队列组件失败: {e}")

    def _register_llm_processors(self) -> None:
        """注册LLM处理器组件"""
        try:
            from consumer.openai_vlm import OpenAIVLMProcessor
            
            llm_factory = self._factories["llm_processor"]
            llm_factory.register_component("openai_vlm", OpenAIVLMProcessor)
            
        except ImportError as e:
            logger.error(f"注册LLM处理器组件失败: {e}")

    def get_factory(self, factory_type: str) -> ComponentFactory:
        """获取指定类型的工厂
        
        Args:
            factory_type: 工厂类型 (video_source, preprocessor, etc.)
            
        Returns:
            对应的组件工厂
            
        Raises:
            ValueError: 工厂类型不存在
        """
        if factory_type not in self._factories:
            available = list(self._factories.keys())
            raise ValueError(f"工厂类型 '{factory_type}' 不存在。可用类型: {available}")
        
        return self._factories[factory_type]

    def create_component(self, factory_type: str, component_name: str, config: Dict[str, Any]) -> Any:
        """创建指定类型的组件
        
        Args:
            factory_type: 工厂类型
            component_name: 组件名称
            config: 组件配置
            
        Returns:
            组件实例
        """
        factory = self.get_factory(factory_type)
        return factory.create_component(component_name, config)

    def load_plugins_from_config(self, plugins_config: Dict[str, Dict[str, str]]) -> None:
        """从配置加载插件
        
        Args:
            plugins_config: 插件配置字典
                格式: {
                    "custom_detector": {
                        "factory_type": "preprocessor",
                        "module": "plugins.detectors",
                        "class": "CustomDetector"
                    }
                }
        """
        for plugin_name, plugin_info in plugins_config.items():
            try:
                factory_type = plugin_info.get("factory_type")
                module_path = plugin_info.get("module")
                class_name = plugin_info.get("class")
                
                if not all([factory_type, module_path, class_name]):
                    logger.error(f"插件 {plugin_name} 配置不完整")
                    continue
                
                factory = self.get_factory(factory_type)
                factory.load_plugin(module_path, class_name, plugin_name)
                
            except Exception as e:
                logger.error(f"加载插件 {plugin_name} 失败: {e}")

    def get_all_components(self) -> Dict[str, Dict[str, Type]]:
        """获取所有工厂的可用组件
        
        Returns:
            工厂类型到组件映射的字典
        """
        all_components = {}
        for factory_type, factory in self._factories.items():
            all_components[factory_type] = factory.get_available_components()
        return all_components


# 全局服务工厂实例
service_factory = ServiceFactory()
