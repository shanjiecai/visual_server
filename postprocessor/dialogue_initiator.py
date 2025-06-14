import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json

from core.interfaces import ProcessingTask
from postprocessor.base import BasePostProcessor


class DialogueInitiatorPostprocessor(BasePostProcessor):
    """对话发起后处理器"""

    @property
    def processor_name(self) -> str:
        return "dialogue_initiator"

    async def _do_initialize(self) -> bool:
        """初始化对话发起器"""
        try:
            # 配置对话参数
            self.dialogue_api_endpoint = self.config.get("dialogue_api_endpoint")
            self.dialogue_templates = self.config.get("dialogue_templates", {})
            self.user_context = self.config.get("user_context", {})

            self._logger.info("Dialogue initiator initialized")
            return True

        except Exception as e:
            self._logger.error(f"Failed to initialize dialogue initiator: {e}")
            return False

    async def _do_execute(self, task: ProcessingTask) -> Dict[str, Any]:
        """执行对话发起"""
        try:
            # 分析处理结果，确定对话类型
            dialogue_type = self._determine_dialogue_type(task)

            if dialogue_type:
                # 生成对话内容
                dialogue_content = await self._generate_dialogue_content(dialogue_type, task)

                # 发起对话
                dialogue_result = await self._initiate_dialogue(dialogue_content)

                return {
                    "task_id": task.task_id,
                    "processor_name": self.processor_name,
                    "status": "completed",
                    "dialogue_type": dialogue_type,
                    "dialogue_content": dialogue_content,
                    "dialogue_result": dialogue_result,
                }
            else:
                return {
                    "task_id": task.task_id,
                    "processor_name": self.processor_name,
                    "status": "skipped",
                    "reason": "No dialogue trigger conditions met"
                }

        except Exception as e:
            raise RuntimeError(f"Dialogue initiation failed: {e}")

    def _determine_dialogue_type(self, task: ProcessingTask) -> Optional[str]:
        """确定对话类型"""
        # 检查是否检测到人员
        for result in task.processing_results:
            if "yolo" in result.processor_name.lower():
                detections = result.result_data.get("detections", [])
                person_count = len([d for d in detections if d.get("class") == "person"])

                if person_count > 0:
                    if person_count == 1:
                        return "single_person_greeting"
                    else:
                        return "multiple_person_greeting"

        # 检查是否有记忆信息
        for result in task.processing_results:
            if "memory" in result.processor_name.lower():
                memories = result.result_data.get("memories", [])
                if memories:
                    return "memory_based_conversation"

        return None

    async def _generate_dialogue_content(self, dialogue_type: str, task: ProcessingTask) -> Dict[str, Any]:
        """生成对话内容"""
        template = self.dialogue_templates.get(dialogue_type, {})

        if dialogue_type == "single_person_greeting":
            return {
                "text": template.get("text", "您好！我注意到您在这里，有什么我可以帮助您的吗？"),
                "tone": "friendly",
                "priority": "normal"
            }

        elif dialogue_type == "multiple_person_greeting":
            return {
                "text": template.get("text", "大家好！欢迎来到这里，有什么我可以为大家服务的吗？"),
                "tone": "welcoming",
                "priority": "normal"
            }

        elif dialogue_type == "memory_based_conversation":
            # 基于记忆信息生成个性化对话
            memory_info = self._extract_memory_info(task)
            return {
                "text": f"根据之前的互动，我记得您对{memory_info}感兴趣，今天有什么新的需求吗？",
                "tone": "personal",
                "priority": "high",
                "context": memory_info
            }

        return {"text": "您好！", "tone": "neutral", "priority": "low"}

    async def _initiate_dialogue(self, dialogue_content: Dict[str, Any]) -> Dict[str, Any]:
        """发起对话"""
        # 模拟对话发起过程
        await asyncio.sleep(0.2)  # 模拟API调用延迟

        # 这里应该调用实际的对话API
        return {
            "dialogue_id": f"dialogue_{int(time.time() * 1000)}",
            "status": "initiated",
            "content": dialogue_content,
            "timestamp": time.time()
        }

    def _extract_memory_info(self, task: ProcessingTask) -> str:
        """提取记忆信息"""
        for result in task.processing_results:
            if "memory" in result.processor_name.lower():
                memories = result.result_data.get("memories", [])
                if memories:
                    # 提取第一个记忆的描述
                    return memories[0].get("description", "相关内容")
        return "一般信息"
