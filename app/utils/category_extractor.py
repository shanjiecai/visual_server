#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
类别提取器，用于从用户问题中提取相关的物品类别
"""

import time
from typing import List, Dict, Any, Optional
from loguru import logger

from consumer.openai_llm import OpenAILLMProcessor


class CategoryExtractor:
    """类别提取器，用于从用户问题中提取相关类别"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化类别提取器
        
        Args:
            llm_config: 语言模型配置
        """
        self.llm_config = llm_config
        self.llm_processor: Optional[OpenAILLMProcessor] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化LLM处理器"""
        try:
            self.llm_processor = OpenAILLMProcessor(self.llm_config)
            success = await self.llm_processor.initialize()
            if success:
                self._initialized = True
                logger.info("类别提取器初始化成功")
            else:
                logger.error("类别提取器初始化失败")
            return success
        except Exception as e:
            logger.error(f"类别提取器初始化异常: {e}")
            return False
    
    async def extract_categories_from_question(self, available_categories: List[str], 
                                             question: str, max_count: int = 3) -> List[str]:
        """
        从用户问题中提取相关类别
        参考visual_memory_demo_clip.py的实现逻辑
        
        Args:
            available_categories: 可用的类别列表
            question: 用户问题
            max_count: 最大返回类别数量
            
        Returns:
            相关类别列表
        """
        if not self._initialized or not self.llm_processor:
            logger.warning("类别提取器未初始化，使用简单匹配策略")
            return self._simple_category_matching(available_categories, question, max_count)
        
        try:
            start_time = time.time()
            
            # 格式化类别列表，添加索引
            formatted_categories = [f"{i}-{item}" for i, item in enumerate(available_categories)]
            
            # 构造提示词（参考visual_memory_demo_clip.py）
            system_prompt = """ ## Background ##
你是一位视觉语言理解专家。用户提出了一个问题，你需要根据语义，在以下类目中找出最相关的一项或多项。

即使用户没有直接提到类目的名字，也请结合含义判断是否相关。

请从类目列表中返回最相关的项（如"水杯"和"手机"等）在列表中的位置，不必解释原因，不能返回不在列表中的项。"""
            
            user_prompt = f"问题：{question}\n从提供类目列表{formatted_categories}中返回最相关的项在列表中的位置，不必解释原因。"
            
            # 调用LLM提取类别
            response = await self.llm_processor._call_llm([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ], max_tokens=64)
            
            logger.info(f"类别提取耗时: {time.time() - start_time:.2f}秒")
            logger.info(f"LLM响应: {response}")
            
            # 解析响应，提取索引
            if response:
                # 解析索引（可能是逗号分隔的数字）
                indices = self._parse_category_indices(response, len(available_categories))
                
                # 根据索引获取对应的类别
                relevant_categories = []
                for idx in indices:
                    if 0 <= idx < len(available_categories):
                        relevant_categories.append(available_categories[idx])
                
                # 如果找到了有效类别，返回
                if relevant_categories:
                    logger.info(f"LLM识别出的相关类别: {relevant_categories}")
                    return relevant_categories[:max_count]
            
            # 如果LLM没有返回有效结果，使用简单匹配
            logger.warning("LLM未返回有效的类别索引，使用简单匹配策略")
            return self._simple_category_matching(available_categories, question, max_count)
            
        except Exception as e:
            logger.error(f"LLM类别提取失败: {e}，使用简单匹配策略")
            return self._simple_category_matching(available_categories, question, max_count)
    
    def _parse_category_indices(self, response: str, max_index: int) -> List[int]:
        """
        解析类别索引
        参考OpenAILLMProcessor中的实现
        
        Args:
            response: LLM响应内容
            max_index: 最大索引值
            
        Returns:
            解析出的索引列表
        """
        import re
        
        try:
            response = response.strip()
            # 查找所有数字
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
    
    def _simple_category_matching(self, available_categories: List[str], 
                                 question: str, max_count: int = 3) -> List[str]:
        """
        简单的类别匹配策略（降级方案）
        参考visual_memory_demo_clip.py的逻辑
        
        Args:
            available_categories: 可用类别列表
            question: 用户问题
            max_count: 最大返回数量
            
        Returns:
            匹配的类别列表
        """
        question_lower = question.lower()
        matched_categories = []
        
        # 中文到英文的关键词映射
        keyword_mapping = {
            '水杯': 'cup',
            '杯子': 'cup',
            '手机': 'mobile phone',
            '电话': 'mobile phone',
            '桌子': 'dining table',
            '餐桌': 'dining table',
            '电视': 'television',
            '电视机': 'television',
            '人': 'person',
            '冰箱': 'Refrigerator',
            '微波炉': 'microwave oven',
            '洗衣机': 'washer',
            '床': 'four-poster bed',
            '花': 'flower',
            '遥控器': 'remote control',
            '书': 'comic book',
            '电脑': 'desktop computer',
            '计算机': 'desktop computer',
            '键盘': 'computer keyboard',
            '洗碗机': 'dishwasher',
            '勺子': 'wooden spoon',
            '刀': 'paper knife',
            '尺子': 'ruler',
            '自行车': 'tandem bicycle'
        }
        
        # 首先尝试中文关键词匹配
        for chinese_keyword, english_keyword in keyword_mapping.items():
            if chinese_keyword in question:
                if english_keyword in available_categories:
                    matched_categories.append(english_keyword)
        
        # 然后尝试英文关键词直接匹配
        for category in available_categories:
            if category.lower() in question_lower:
                if category not in matched_categories:  # 避免重复
                    matched_categories.append(category)
        
        # 如果没有直接匹配，返回前几个类别作为兜底
        if not matched_categories:
            matched_categories = available_categories[:min(max_count, 2)]
            logger.info(f"未找到直接匹配的类别，使用默认类别: {matched_categories}")
        else:
            logger.info(f"找到匹配的类别: {matched_categories}")
            
        return matched_categories[:max_count]
    
    def extract_categories_sync(self, available_categories: List[str], 
                               question: str, max_count: int = 3) -> List[str]:
        """
        同步版本的类别提取（用于非async环境）
        
        Args:
            available_categories: 可用类别列表
            question: 用户问题
            max_count: 最大返回数量
            
        Returns:
            相关类别列表
        """
        # 如果未初始化，直接使用简单匹配
        if not self._initialized:
            return self._simple_category_matching(available_categories, question, max_count)
        
        # 尝试运行异步提取，如果失败则使用简单匹配
        try:
            import asyncio
            
            # 检查是否已经在事件循环中
            try:
                current_loop = asyncio.get_running_loop()
                # 如果已经在事件循环中，直接使用简单匹配
                logger.warning("已在异步环境中，使用简单匹配策略")
                return self._simple_category_matching(available_categories, question, max_count)
            except RuntimeError:
                # 没有运行中的事件循环，可以创建新的
                pass
            
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(
                    self.extract_categories_from_question(available_categories, question, max_count)
                )
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"同步类别提取失败: {e}")
            return self._simple_category_matching(available_categories, question, max_count)
