#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI语言模型处理器实现
专门处理文本理解、类别提取等纯语言任务
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from loguru import logger
from openai import OpenAI


class OpenAILLMProcessor:
    """OpenAI语言模型处理器，专门处理文本任务"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM处理器
        
        Args:
            config: LLM配置信息，包含base_url、api_key、model等
        """
        self.config = config
        self.client: Optional[OpenAI] = None
        self.model: Optional[str] = None
        self.max_tokens = config.get("max_tokens", 64)
        self.temperature = config.get("temperature", 0.3)
        self.timeout = config.get("timeout", 30.0)
        self._initialized = False

    async def initialize(self) -> bool:
        """初始化LLM客户端"""
        try:
            base_url = self.config.get("base_url", "http://10.112.0.32:5239/v1")
            api_key = self.config.get("api_key", "EMPTY")
            self.model = self.config.get("model", "qwen2.5-7b-test")

            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )

            self._initialized = True
            logger.info(f"LLM处理器已初始化: {self.model}")
            return True

        except Exception as e:
            logger.error(f"LLM处理器初始化失败: {e}")
            return False

    async def extract_categories_from_question(self, available_categories: List[str], 
                                             question: str, max_count: int = 3) -> List[str]:
        """
        从问题中提取相关类别
        
        Args:
            available_categories: 可用的类别列表
            question: 用户问题
            max_count: 最大返回类别数量
            
        Returns:
            相关类别列表
        """
        if not self._initialized or not self.client:
            logger.warning("LLM处理器未初始化，使用简单匹配策略")
            return self._simple_category_matching(available_categories, question, max_count)

        try:
            # 构造类别列表
            formatted_categories = [f"{i}-{item}" for i, item in enumerate(available_categories)]
            
            # 使用类别提取提示词
            system_prompt = """你是一位视觉语言理解专家。用户提出了一个问题，你需要根据语义，在提供的类目中找出最相关的一项或多项。

即使用户没有直接提到类目的名字，也请结合含义判断是否相关。

请从类目列表中返回最相关的项在列表中的位置（从0开始的索引），不必解释原因，不能返回不在列表中的项。

输出格式：只返回索引数字，如果有多个用逗号分隔，例如：0,2,5"""

            messages = [
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user',
                    'content': f"问题：{question}\n类目列表：{formatted_categories}\n\n请返回最相关的类目索引："
                }
            ]
            
            # 调用语言模型
            response = await self._call_llm(messages, max_tokens=64)
            
            # 解析索引
            indices = self._parse_category_indices(response, len(available_categories))
            
            # 返回对应的类别
            relevant_categories = []
            for idx in indices:
                if 0 <= idx < len(available_categories):
                    relevant_categories.append(available_categories[idx])
            
            # 如果没有找到相关类别，返回前几个
            if not relevant_categories:
                relevant_categories = available_categories[:min(max_count, 2)]
            
            return relevant_categories[:max_count]
            
        except Exception as e:
            logger.error(f"提取类别失败: {e}")
            # 降级策略：使用简单的关键词匹配
            return self._simple_category_matching(available_categories, question, max_count)

    async def _call_llm(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None) -> str:
        """
        调用语言模型
        
        Args:
            messages: 消息列表
            max_tokens: 最大token数
            
        Returns:
            模型响应内容
        """
        if not self._initialized or not self.client:
            raise ValueError("LLM处理器未初始化")
            
        try:
            tokens = max_tokens or self.max_tokens
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    stream=False
                )
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

    def _simple_category_matching(self, available_categories: List[str], 
                                 question: str, max_count: int = 3) -> List[str]:
        """
        简单的类别匹配策略（降级方案）
        
        Args:
            available_categories: 可用类别列表
            question: 用户问题
            max_count: 最大返回数量
            
        Returns:
            匹配的类别列表
        """
        question_lower = question.lower()
        matched_categories = []
        
        for category in available_categories:
            if category.lower() in question_lower:
                matched_categories.append(category)
        
        if not matched_categories:
            matched_categories = available_categories[:min(max_count, 2)]
            
        return matched_categories[:max_count]

    def _parse_category_indices(self, response: str, max_index: int) -> List[int]:
        """
        解析类别索引
        
        Args:
            response: LLM响应内容
            max_index: 最大索引值
            
        Returns:
            解析出的索引列表
        """
        try:
            response = response.strip()
            numbers = re.findall(r'\d+', response)
            
            indices = []
            for num_str in numbers:
                try:
                    idx = int(num_str)
                    if 0 <= idx < max_index:
                        indices.append(idx)
                except ValueError:
                    continue
            
            return indices
            
        except Exception as e:
            logger.error(f"解析类别索引失败: {e}")
            return []

    async def health_check(self) -> bool:
        """健康检查"""
        if not self._initialized or not self.client:
            return False
            
        try:
            # 发送一个简单的测试请求
            test_messages = [
                {"role": "user", "content": "Hello, this is a health check."}
            ]

            response = await self._call_llm(test_messages, max_tokens=10)
            return response is not None and len(response) > 0

        except Exception as e:
            logger.error(f"LLM健康检查失败: {e}")
            return False

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized and self.client is not None
