#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI通用视觉大模型处理器实现
支持通过消息metadata配置的prompt进行多模态理解
"""

import asyncio
import base64
import time
import json
from typing import Dict, Any, List, Optional
from loguru import logger
import openai
from openai import OpenAI

from .base import BaseLLMProcessor
from core.interfaces import ProcessingTask, ProcessingResult


class OpenAIVLMProcessor(BaseLLMProcessor):
    """OpenAI通用视觉大模型处理器实现"""

    async def _do_initialize(self) -> bool:
        """初始化OpenAI客户端"""
        try:
            # 获取配置
            self.processor_name = self.config.get("processor_name", "openai_vlm_processor")
            self.base_url = self.config.get("base_url", "http://cc.komect.com/llm/vlgroup/")
            self.api_key = self.config.get("api_key", "EMPTY")
            self.model_name = self.config.get("model_name", "Qwen2.5-VL-72B-Instruct-AWQ")
            self.max_tokens = self.config.get("max_tokens", 1024)
            self.temperature = self.config.get("temperature", 0.7)
            self.stream = self.config.get("stream", False)

            # 初始化OpenAI客户端
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

            # 默认系统提示（如果消息中没有指定）
            self.default_system_prompt = self.config.get(
                "default_system_prompt",
                self._get_default_system_prompt()
            )

            logger.info(f"OpenAI VLM client initialized: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI VLM client: {e}")
            return False

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        return """你是一个专业的视觉理解助手，能够分析图像内容并根据用户要求提供准确的回答。

请根据提供的图像和具体任务要求进行分析，提供详细、准确的描述和分析结果。

请用中文回答，保持回答简洁明了且富有信息量。"""

    async def _do_process_task(self, task: ProcessingTask) -> ProcessingResult:
        """处理包含图像的任务"""
        try:
            # 解析任务数据
            message_data = self._parse_task_message(task)

            # 检查是否有图像数据
            if not message_data.get("image_base64") and not message_data.get("images_base64"):
                raise ValueError("No image data found in task")

            # 构建消息
            messages = self._build_messages(message_data)

            # 调用OpenAI API
            if self.stream:
                response = await self._stream_completion(messages)
            else:
                response = await self._complete_chat(messages)

            # 处理响应
            result = self._process_response(response, message_data)

            return ProcessingResult(
                frame_id=message_data.get("frame_id", "unknown"),
                processor_name=self.processor_name,
                result_data=result,
                confidence=result.get("confidence", 0.8),
                metadata={
                    "model": self.model_name,
                    "task_type": message_data.get("task_type", "general"),
                    "processing_time": result.get("processing_time", 0),
                    "source": message_data.get("source", "unknown")
                }
            )

        except Exception as e:
            logger.error(f"OpenAI VLM processing failed for task {task.task_id}: {e}")
            raise

    def _parse_task_message(self, task: ProcessingTask) -> Dict[str, Any]:
        """解析任务消息数据"""
        # 检查是否是Kafka消息格式
        if hasattr(task.frame_data, 'raw_data'):
            raw_data = task.frame_data.raw_data

            # 如果是字符串（JSON消息）
            if isinstance(raw_data, str):
                try:
                    return json.loads(raw_data)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON message")

            # 如果是字典
            elif isinstance(raw_data, dict):
                return raw_data

        # 兜底：从ProcessingTask中提取信息
        return {
            "frame_id": task.frame_data.frame_id,
            "timestamp": task.frame_data.timestamp,
            "image_base64": None,
            "images_base64": [],  # 新增多图支持
            "task_type": "general",
            "prompt": None,
            "metadata": {}
        }

    def _build_messages(self, message_data: Dict[str, Any]) -> List[Dict]:
        """构建消息内容"""
        # 获取系统提示
        system_prompt = message_data.get("system_prompt", self.default_system_prompt)

        # 获取用户提示
        user_prompt = message_data.get("prompt", message_data.get("user_prompt", "请分析这些图像。"))

        # 构建消息内容
        content = []

        # 处理多图情况
        images_base64 = message_data.get("images_base64", [])
        if images_base64:
            # 如果有多图数组，使用多图数组
            for image_base64 in images_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })
        elif message_data.get("image_base64"):
            # 兼容单图情况
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{message_data['image_base64']}"
                }
            })

        # 添加文本提示
        content.append({
            "type": "text",
            "text": user_prompt
        })

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]

        return messages

    async def _complete_chat(self, messages: List[Dict]) -> Dict[str, Any]:
        """非流式聊天完成"""
        start_time = time.time()

        try:
            formatted_messages = self._format_messages_for_log(messages)
            logger.info(f"OpenAI API call: {formatted_messages}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )

            processing_time = time.time() - start_time

            return {
                "type": "complete",
                "response": response,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    async def _stream_completion(self, messages: List[Dict]) -> Dict[str, Any]:
        """流式聊天完成"""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                timeout=self.timeout
            )

            # 收集流式响应
            content_parts = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)

            processing_time = time.time() - start_time

            return {
                "type": "stream",
                "content": "".join(content_parts),
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"OpenAI streaming API call failed: {e}")
            raise

    def _process_response(self, response: Dict[str, Any], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理API响应"""
        result = {
            "frame_id": message_data.get("frame_id"),
            "timestamp": message_data.get("timestamp"),
            "task_type": message_data.get("task_type", "general"),
            "processing_time": response.get("processing_time", 0),
            "model": self.model_name,
            "source_metadata": message_data.get("metadata", {})
        }

        if response["type"] == "complete":
            # 非流式响应
            api_response = response["response"]
            content = api_response.choices[0].message.content

            result.update({
                "response_type": "complete",
                "content": content,
                "usage": api_response.usage.model_dump() if api_response.usage else {},
                "finish_reason": api_response.choices[0].finish_reason
            })

        elif response["type"] == "stream":
            # 流式响应
            content = response["content"]
            result.update({
                "response_type": "stream",
                "content": content
            })

        # 计算置信度
        result["confidence"] = self._calculate_confidence(content, message_data)

        return result

    def _calculate_confidence(self, content: str, message_data: Dict[str, Any]) -> float:
        """计算处理置信度"""
        base_confidence = 0.7

        # 根据响应内容质量调整置信度
        if content and len(content) > 50:
            base_confidence += 0.1

        # 如果有详细分析，进一步增加置信度
        if content and any(word in content for word in ["分析", "观察", "描述", "检测"]):
            base_confidence += 0.1

        # 根据任务类型调整
        task_type = message_data.get("task_type", "general")
        if task_type in ["person_detection", "object_detection"]:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    async def _do_health_check(self) -> bool:
        """OpenAI健康检查"""
        try:
            # 发送一个简单的测试请求
            test_messages = [
                {"role": "user", "content": "Hello, this is a health check."}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=test_messages,
                max_tokens=10,
                timeout=5.0
            )

            return response.choices[0].message.content is not None

        except Exception as e:
            logger.error(f"OpenAI VLM health check failed: {e}")
            return False

    def _format_messages_for_log(self, messages):
        def shorten_url(url):
            if url.startswith("data:image"):
                # 只保留前后20位
                return url[:20] + "...(base64省略)..." + url[-20:]
            return url

        def process_content(content):
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "image_url":
                        img_url = item["image_url"]["url"]
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": shorten_url(img_url)}
                        })
                    else:
                        new_content.append(item)
                return new_content
            return content

        # 处理每条message
        formatted = []
        for msg in messages:
            msg_copy = msg.copy()
            if isinstance(msg_copy.get("content"), list):
                msg_copy["content"] = process_content(msg_copy["content"])
            formatted.append(msg_copy)
        return formatted
