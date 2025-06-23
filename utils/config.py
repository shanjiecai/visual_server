#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理模块
提供统一的配置管理入口
"""

import os
import yaml
from loguru import logger
from typing import Dict, Any, Optional

from core.interfaces import IConfigManager


class ConfigManager(IConfigManager):
    """配置管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_file: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: Optional[str] = None):
        if self._initialized:
            return
            
        self.config_file = config_file or "config.yaml"
        self._config: Dict[str, Any] = {}
        self._loaded = False
        self._initialized = True
        
        # 自动加载配置
        self.load_config()
    
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"Config file not found: {self.config_file}")
                return False
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            
            # 处理环境变量
            self._process_env_variables()
            
            self._loaded = True
            logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading config file {self.config_file}: {e}")
            return False
    
    def reload(self) -> bool:
        """重新加载配置"""
        return self.load_config()
    
    def _process_env_variables(self):
        """处理环境变量替换"""
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                # 处理 ${ENV_VAR} 或 ${ENV_VAR:default_value} 格式
                env_var = obj[2:-1]
                if ":" in env_var:
                    var_name, default = env_var.split(":", 1)
                    return os.getenv(var_name, default)
                else:
                    return os.getenv(env_var, obj)
            else:
                return obj
        
        self._config = replace_env_vars(self._config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点分隔的嵌套键"""
        if not self._loaded:
            logger.warning("Configuration not loaded")
            return default
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值，支持点分隔的嵌套键"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config file {self.config_file}: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config, new_config)
    
    def has_key(self, key: str) -> bool:
        """检查配置键是否存在"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return False
        return True
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置的某个section"""
        return self.get(section, {})
    
    @classmethod
    def get_instance(cls, config_file: Optional[str] = None) -> 'ConfigManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(config_file)
        return cls._instance


# 全局配置管理器实例
config_manager = ConfigManager.get_instance()


def get_config(key: str, default: Any = None) -> Any:
    """全局配置获取函数"""
    return config_manager.get(key, default)


def set_config(key: str, value: Any) -> None:
    """全局配置设置函数"""
    config_manager.set(key, value)


def reload_config() -> bool:
    """重新加载配置"""
    return config_manager.reload() 