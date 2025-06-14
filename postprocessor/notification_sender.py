import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json

from core.interfaces import ProcessingTask
from postprocessor.base import BasePostProcessor


class NotificationPostprocessor(BasePostProcessor):
    """通知发送后处理器"""
    
    @property
    def processor_name(self) -> str:
        return "notification_sender"
    
    async def _do_initialize(self) -> bool:
        """初始化通知发送器"""
        try:
            self.notification_channels = self.config.get("notification_channels", ["email", "sms"])
            self.notification_rules = self.config.get("notification_rules", [])
            self.admin_contacts = self.config.get("admin_contacts", [])
            
            self._logger.info(f"Notification sender initialized with channels: {self.notification_channels}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize notification sender: {e}")
            return False
    
    async def _do_execute(self, task: ProcessingTask) -> Dict[str, Any]:
        """执行通知发送"""
        try:
            notifications_sent = []
            
            # 检查通知规则
            for rule in self.notification_rules:
                if self._should_send_notification(rule, task):
                    notification = await self._send_notification(rule, task)
                    notifications_sent.append(notification)
            
            return {
                "task_id": task.task_id,
                "processor_name": self.processor_name,
                "status": "completed",
                "notifications_sent": notifications_sent,
                "notification_count": len(notifications_sent)
            }
            
        except Exception as e:
            raise RuntimeError(f"Notification sending failed: {e}")
    
    def _should_send_notification(self, rule: Dict[str, Any], task: ProcessingTask) -> bool:
        """判断是否应该发送通知"""
        rule_type = rule.get("type")
        
        if rule_type == "person_detected":
            # 检测到人员时发送通知
            for result in task.processing_results:
                if "yolo" in result.processor_name.lower():
                    detections = result.result_data.get("detections", [])
                    person_count = len([d for d in detections if d.get("class") == "person"])
                    min_count = rule.get("min_person_count", 1)
                    if person_count >= min_count:
                        return True
        
        elif rule_type == "low_confidence":
            # 置信度低时发送通知
            threshold = rule.get("confidence_threshold", 0.3)
            for result in task.processing_results:
                if result.confidence < threshold:
                    return True
        
        elif rule_type == "anomaly_detected":
            # 检测到异常时发送通知
            for result in task.processing_results:
                if result.result_data.get("anomaly", False):
                    return True
        
        return False
    
    async def _send_notification(self, rule: Dict[str, Any], task: ProcessingTask) -> Dict[str, Any]:
        """发送通知"""
        # 生成通知内容
        notification_content = self._generate_notification_content(rule, task)
        
        # 模拟发送通知
        await asyncio.sleep(0.1)  # 模拟发送延迟
        
        notification = {
            "notification_id": f"notif_{int(time.time() * 1000)}",
            "rule_type": rule.get("type"),
            "channels": rule.get("channels", self.notification_channels),
            "content": notification_content,
            "recipients": rule.get("recipients", self.admin_contacts),
            "status": "sent",
            "timestamp": time.time()
        }
        
        self._logger.info(f"Sent notification: {notification['notification_id']}")
        return notification
    
    def _generate_notification_content(self, rule: Dict[str, Any], task: ProcessingTask) -> Dict[str, Any]:
        """生成通知内容"""
        rule_type = rule.get("type")
        
        if rule_type == "person_detected":
            return {
                "subject": "人员检测警报",
                "message": f"在帧 {task.frame_data.frame_id} 中检测到人员活动",
                "priority": "normal",
                "frame_info": {
                    "frame_id": task.frame_data.frame_id,
                    "timestamp": task.frame_data.timestamp
                }
            }
        
        elif rule_type == "low_confidence":
            return {
                "subject": "处理置信度警告",
                "message": f"帧 {task.frame_data.frame_id} 的处理结果置信度较低，需要人工确认",
                "priority": "medium",
                "confidence_info": [
                    {"processor": r.processor_name, "confidence": r.confidence}
                    for r in task.processing_results
                ]
            }
        
        elif rule_type == "anomaly_detected":
            return {
                "subject": "异常检测警报",
                "message": f"在帧 {task.frame_data.frame_id} 中检测到异常情况",
                "priority": "high",
                "anomaly_details": "详细异常信息"
            }
        
        return {
            "subject": "系统通知",
            "message": f"任务 {task.task_id} 触发了通知规则",
            "priority": "low"
        }
