#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI通用视觉大模型处理器实现
专注于核心的多模态理解功能
"""

import time
import json
from typing import Dict, Any, List, Optional
from loguru import logger
from openai import OpenAI

from .base import BaseLLMProcessor
from core.interfaces import ProcessingTask, ProcessingResult


class OpenAIVLMProcessor(BaseLLMProcessor):
    """OpenAI通用视觉大模型处理器实现"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    async def _do_initialize(self) -> bool:
        """初始化OpenAI客户端"""
        try:
            # 获取VLM配置
            self.processor_name = self.config.get("processor_name", "openai_vlm_processor")
            self.base_url = self.config.get("base_url", "http://cc.komect.com/llm/vlgroup/")
            self.api_key = self.config.get("api_key", "EMPTY")
            self.model_name = self.config.get("model_name", "Qwen2.5-VL-72B-Instruct-AWQ")
            self.max_tokens = self.config.get("max_tokens", 1024)
            self.temperature = self.config.get("temperature", 0.7)
            self.stream = self.config.get("stream", False)

            # 初始化VLM客户端
            self.vlm_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

            # 默认系统提示
            self.default_system_prompt = self.config.get(
                "default_system_prompt",
                self._get_default_system_prompt()
            )

            logger.info(f"OpenAI VLM处理器已初始化: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI VLM processor: {e}")
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
            
            logger.info(f"处理VLM任务: {message_data.get('task_type', 'general')}")

            # 处理通用VLM任务
            return await self._process_general_vlm_task(task, message_data)

        except Exception as e:
            logger.error(f"OpenAI VLM processing failed for task {task.task_id}: {e}")
            raise

    async def _process_general_vlm_task(self, task: ProcessingTask, message_data: Dict[str, Any]) -> ProcessingResult:
        """处理通用VLM任务"""
        # 检查图像数据
        if not self._has_image_data(message_data):
            raise ValueError("No image data found in task")

        # 获取任务配置
        task_config = self._get_vlm_task_config(message_data)
        
        # 构建消息
        messages = self._build_vlm_messages(message_data, task_config)

        # 获取VLM配置
        vlm_config = task_config.get("vlm_config", {})
        
        # 调用VLM API
        response = await self._call_vlm_api(messages, vlm_config)

        # 处理响应
        result = self._process_response(response, message_data, task_config)

        return ProcessingResult(
            frame_id=message_data.get("frame_id", "unknown"),
            processor_name=self.processor_name,
            result_data=result,
            confidence=result.get("confidence", 0.8),
            metadata={
                "model": vlm_config.get("model", self.model_name),
                "task_type": task_config.get("task_type", "general"),
                "processing_time": result.get("processing_time", 0),
                "source": message_data.get("source", "unknown")
            }
        )

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
            "images_base64": [],
            "task_type": "general",
            "prompt": None,
            "metadata": {}
        }

    def _get_vlm_task_config(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取VLM任务配置"""
        # 从vlm_task_config获取
        if "vlm_task_config" in message_data:
            vlm_config_dict = message_data["vlm_task_config"]
            # 根据检测结果或消息中的task_type确定任务类型
            task_type = self._determine_task_type_from_message(message_data, vlm_config_dict)
            if task_type and task_type in vlm_config_dict:
                logger.info(f"使用vlm_task_config中的任务: {task_type}")
                return vlm_config_dict[task_type]
        
        # 从消息数据中直接获取（旧格式）
        if "system_prompt" in message_data or "user_prompt" in message_data:
            logger.info("使用消息中的直接prompt配置")
            return {
                "task_type": message_data.get("task_type", "general"),
                "system_prompt": message_data.get("system_prompt", self.default_system_prompt),
                "user_prompt": message_data.get("user_prompt", message_data.get("prompt", "请分析这张图像。")),
                "vlm_config": {}
            }
        
        # 默认配置
        logger.info("使用默认任务配置")
        return {
            "task_type": "general_analysis",
            "system_prompt": self.default_system_prompt,
            "user_prompt": "请分析这张图像。",
            "vlm_config": {}
        }

    def _determine_task_type_from_message(self, message_data: Dict[str, Any], vlm_config_dict: Dict[str, Any]) -> str:
        """根据消息数据确定任务类型"""
        # 1. 如果消息中指定了task_type，优先使用
        if "task_type" in message_data:
            task_type = message_data["task_type"]
            if task_type in vlm_config_dict:
                return task_type
        
        # 2. 根据检测结果确定任务类型
        detected_categories = message_data.get("categories", [])
        if detected_categories:
            # 如果检测到人员且有person_detection配置，优先使用
            if "person" in detected_categories and "person_detection" in vlm_config_dict:
                return "person_detection"
            
            # 如果检测到物体且有object_detection配置
            if detected_categories and "object_detection" in vlm_config_dict:
                return "object_detection"
        
        # 3. 默认使用general_analysis
        if "general_analysis" in vlm_config_dict:
            return "general_analysis"
        
        # 4. 如果以上都没有，返回第一个可用的配置
        return list(vlm_config_dict.keys())[0] if vlm_config_dict else "general_analysis"

    def _has_image_data(self, message_data: Dict[str, Any]) -> bool:
        """检查是否有图像数据"""
        return bool(message_data.get("image_base64") or message_data.get("images_base64"))

    def _build_vlm_messages(self, message_data: Dict[str, Any], task_config: Dict[str, Any] = None) -> List[Dict]:
        """构建VLM消息内容"""
        if task_config is None:
            task_config = self._get_vlm_task_config(message_data)
            
        # 获取系统提示和用户提示
        system_prompt = task_config.get("system_prompt", self.default_system_prompt)
        user_prompt = task_config.get("user_prompt", "请分析这些图像。")

        # 构建消息内容
        content = []

        # 处理多图情况
        images_base64 = message_data.get("images_base64", [])
        if images_base64:
            for image_base64 in images_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })
        elif message_data.get("image_base64"):
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

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

    async def _call_vlm_api(self, messages: List[Dict], vlm_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """调用VLM API"""
        if vlm_config is None:
            vlm_config = {}
            
        start_time = time.time()

        try:
            # 使用配置中的参数，如果没有则使用默认值
            model = vlm_config.get("model", self.model_name)
            max_tokens = vlm_config.get("max_tokens", self.max_tokens)
            temperature = vlm_config.get("temperature", self.temperature)
            
            # 格式化消息用于日志
            formatted_messages = self._format_messages_for_log(messages)
            logger.info(f"VLM API调用: {formatted_messages}")
            
            if self.stream:
                response = self.vlm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
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
            else:
                response = self.vlm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.timeout
                )

                processing_time = time.time() - start_time
                return {
                    "type": "complete",
                    "response": response,
                    "processing_time": processing_time
                }

        except Exception as e:
            logger.error(f"VLM API调用失败: {e}")
            raise

    def _process_response(self, response: Dict[str, Any], message_data: Dict[str, Any], task_config: Dict[str, Any]) -> Dict[str, Any]:
        """处理API响应"""
        result = {
            "frame_id": message_data.get("frame_id"),
            "timestamp": message_data.get("timestamp"),
            "task_type": task_config.get("task_type", "general"),
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
        if task_type in ["object_detection", "person_detection"]:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    async def _do_health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查VLM客户端
            test_messages = [
                {"role": "user", "content": "Hello, this is a health check."}
            ]

            response = self.vlm_client.chat.completions.create(
                model=self.model_name,
                messages=test_messages,
                max_tokens=10,
                timeout=5.0
            )

            return response.choices[0].message.content is not None

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def _format_messages_for_log(self, messages):
        """格式化消息用于日志输出"""
        def shorten_url(url):
            if url.startswith("data:image"):
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

        formatted = []
        for msg in messages:
            msg_copy = msg.copy()
            if isinstance(msg_copy.get("content"), list):
                msg_copy["content"] = process_content(msg_copy["content"])
            formatted.append(msg_copy)
        return formatted

    def _create_result(self, task: ProcessingTask, answer: Any, 
                      additional_metadata: Optional[Dict] = None) -> ProcessingResult:
        """创建处理结果"""
        metadata = {
            "processor": "openai_vlm",
            "timestamp": time.time()
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return ProcessingResult(
            frame_id=task.frame_data.frame_id,
            processor_name="openai_vlm",
            result_data={"answer": answer} if isinstance(answer, str) else answer,
            confidence=0.9,
            metadata=metadata,
            timestamp=time.time()
        )
    
    def _create_error_result(self, task: ProcessingTask, error_message: str) -> ProcessingResult:
        """创建错误结果"""
        return ProcessingResult(
            frame_id=task.frame_data.frame_id,
            processor_name="openai_vlm",
            result_data={"error": error_message},
            confidence=0.0,
            metadata={
                "processor": "openai_vlm",
                "error": True,
                "timestamp": time.time()
            },
            timestamp=time.time()
        )
