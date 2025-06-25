#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视觉记忆查询API路由
提供HTTP接口进行视觉记忆查询和管理
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from utils.memory_storage import VisualMemoryStorage
from consumer.openai_vlm import OpenAIVLMProcessor
from core.interfaces import ProcessingTask, FrameData, ProcessingStatus
from app.category_extractor import CategoryExtractor
from common.response.response_schema import ResponseModel
from common.response.response_code import CustomResponseCode


# Pydantic 模型定义
class MemoryQueryRequest(BaseModel):
    question: str = Field(..., description="要查询的问题")
    session_id: Optional[str] = Field(None, description="会话ID，可选")


class ClearMemoryRequest(BaseModel):
    confirm: bool = Field(..., description="确认清空所有记忆")


class MemoryQueryResponse(BaseModel):
    answer: str
    session_id: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class MemoryStatsResponse(BaseModel):
    overall_stats: Dict[str, Any]
    category_stats: Dict[str, Any]


class CategoriesResponse(BaseModel):
    categories: List[str]
    count: int


class FrameInfo(BaseModel):
    frame_id: str
    timestamp: float
    categories: List[str]
    metadata: Optional[Dict[str, Any]] = None
    created_at: float
    has_image: bool


class FramesResponse(BaseModel):
    category: str
    frames: List[FrameInfo]
    count: int


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    memory_stats: Dict[str, Any]


class MemoryAPIService:
    """视觉记忆查询API服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化记忆存储
        memory_config = config.get("memory_storage", {})
        self.memory_storage = VisualMemoryStorage(memory_config)

        # 初始化记忆查询处理器
        vlm_config = config.get("memory_vlm", {})
        # 启用记忆存储功能
        vlm_config["enable_memory_storage"] = True
        vlm_config["memory_storage"] = memory_config
        self.memory_vlm_processor = OpenAIVLMProcessor(vlm_config)

        # 初始化类别提取器
        llm_config = config.get("memory_vlm", {}).get("llm_config", {})
        self.category_extractor = CategoryExtractor(llm_config)

        # 线程池用于异步处理
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("Memory API service initialized")

    async def initialize(self) -> bool:
        """异步初始化服务"""
        try:
            # 初始化记忆VLM处理器
            success = await self.memory_vlm_processor.initialize()
            if not success:
                logger.error("Failed to initialize memory VLM processor")
                return False

            # 初始化类别提取器
            category_success = await self.category_extractor.initialize()
            if not category_success:
                logger.warning("Failed to initialize category extractor, will use simple matching")

            logger.info("Memory API service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory API service: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            # 关闭线程池
            self.executor.shutdown(wait=True)
            logger.info("Memory API service cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _process_memory_query(self, question: str, session_id: str) -> Dict[str, Any]:
        """处理记忆查询（同步版本）"""
        try:
            # 获取记忆存储实例
            memory_storage = self.get_memory_storage()
            
            # 获取所有可用的类别
            available_categories = memory_storage.get_all_categories()
            
            if not available_categories:
                return {
                    "answer": "暂无记忆数据，无法回答问题",
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "metadata": {"categories_found": 0}
                }
            
            logger.info(f"可用类别: {available_categories}")
            
            # 使用类别提取器提取相关类别
            relevant_categories = self.category_extractor.extract_categories_sync(
                available_categories, 
                question, 
                max_count=3
            )
            
            logger.info(f"识别出的相关类别: {relevant_categories}")
            
            # 根据相关类别获取记忆帧
            relevant_frames = []
            max_frames_per_category = 2  # 每个类别最多获取2张图片
            
            for category in relevant_categories[:3]:  # 限制最多3个类别
                category_frames = memory_storage.get_frames_by_category(category, max_frames_per_category)
                relevant_frames.extend(category_frames)
                logger.info(f"类别 '{category}' 找到 {len(category_frames)} 张相关图片")
            
            if not relevant_frames:
                return {
                    "answer": "未找到相关的记忆图片，无法回答问题",
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "metadata": {
                        "categories_searched": relevant_categories,
                        "frames_found": 0
                    }
                }
            
            logger.info(f"总共找到 {len(relevant_frames)} 张相关图片")
            
            # 创建模拟的处理任务
            frame_data = FrameData(
                frame_id=f"query_{int(time.time())}",
                timestamp=time.time(),
                raw_data=None,
                metadata={
                    "question": question, 
                    "session_id": session_id,
                    "relevant_categories": relevant_categories,
                    "relevant_frames": [frame.frame_id for frame in relevant_frames]
                }
            )

            task = ProcessingTask(
                task_id=f"memory_query_{session_id}_{int(time.time())}",
                frame_data=frame_data,
                processing_results=[],
                status=ProcessingStatus.PENDING,
                created_at=time.time(),
                updated_at=time.time()
            )

            # 将相关帧添加到任务的元数据中
            task.metadata = {
                "relevant_frames": relevant_frames,
                "relevant_categories": relevant_categories
            }

            # 使用asyncio运行异步处理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # 初始化VLM处理器（如果还未初始化）
                if not hasattr(self.memory_vlm_processor,
                               '_is_initialized') or not self.memory_vlm_processor._is_initialized:
                    initialization_success = loop.run_until_complete(self.memory_vlm_processor.initialize())
                    if not initialization_success:
                        raise RuntimeError("Failed to initialize memory VLM processor")

                # 处理任务
                result = loop.run_until_complete(self.memory_vlm_processor.process_task(task))

                # 提取结果
                answer = result.result_data.get("answer", "无法获取答案")

                return {
                    "answer": answer,
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "metadata": {
                        **result.metadata,
                        "categories_searched": relevant_categories,
                        "frames_used": len(relevant_frames)
                    }
                }

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Error processing memory query: {e}")
            raise

    def get_memory_storage(self) -> VisualMemoryStorage:
        """获取记忆存储实例"""
        # 优先从VLM处理器获取
        if hasattr(self.memory_vlm_processor, 'memory_storage') and self.memory_vlm_processor.memory_storage:
            return self.memory_vlm_processor.memory_storage
        # 兜底返回本地存储
        return self.memory_storage


# 全局服务实例
_memory_service: Optional[MemoryAPIService] = None


def get_memory_service() -> MemoryAPIService:
    """获取记忆服务实例"""
    global _memory_service
    if _memory_service is None:
        # 默认配置
        default_config = {
            "memory_storage": {
                "storage_dir": "memory_storage",
                "max_frames_per_category": 100,
                "max_total_frames": 1000,
                "cleanup_interval": 300,
                "frame_ttl": 3600
            },
            "memory_vlm": {
                "vlm_config": {
                    "base_url": "http://cc.komect.com/llm/vlgroup/",
                    "api_key": "EMPTY",
                    "model": "Qwen2.5-VL-72B-Instruct-AWQ",
                    "max_tokens": 128,
                    "temperature": 0.7
                },
                "llm_config": {
                    "base_url": "http://10.112.0.32:5239/v1",
                    "api_key": "",
                    "model": "qwen2.5-7b-test"
                },
                "max_frames_per_query": 3,
                "category_extraction_enabled": True,
                "prompts": {
                    "memory_qa_system_prompt": """你具有高级图像分析系统，拥有连续多帧图像的观察结果。

请你根据以下连续拍摄的图像，分析用户的问题，并根据图像中的内容进行推理判断。

要求：
1. 仔细观察图像中的物体、位置、状态等细节
2. 根据图像内容直接回答问题，不要编造信息
3. 如果涉及空间位置，请只返回物体当前所在的位置描述
4. 保持回答简洁明确，不要解释过程
5. 禁止在回答中出现"图片"和"图像"等词语

注意：你看到的图像是真实的连续观察结果，请基于这些视觉信息回答用户问题。""",
                    "category_extraction_prompt": """你是一位视觉语言理解专家。用户提出了一个问题，你需要根据语义，在提供的类目中找出最相关的一项或多项。

即使用户没有直接提到类目的名字，也请结合含义判断是否相关。

请从类目列表中返回最相关的项在列表中的位置（从0开始的索引），不必解释原因，不能返回不在列表中的项。

输出格式：只返回索引数字，如果有多个用逗号分隔，例如：0,2,5"""
                }
            }
        }
        _memory_service = MemoryAPIService(default_config)
    return _memory_service


def create_memory_router() -> APIRouter:
    """创建记忆API路由器"""
    router = APIRouter(prefix="/memory")

    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """健康检查接口"""
        service = get_memory_service()
        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            memory_stats=service.memory_storage.get_memory_stats()
        )

    @router.get("/stats", response_model=ResponseModel)
    async def get_memory_stats():
        """获取记忆统计信息"""
        try:
            service = get_memory_service()
            stats = service.memory_storage.get_memory_stats()
            category_stats = service.memory_storage.get_category_stats()

            return ResponseModel(
                code=CustomResponseCode.HTTP_200.code,
                msg=CustomResponseCode.HTTP_200.msg,
                data=MemoryStatsResponse(
                    overall_stats=stats,
                    category_stats=category_stats
                )
            )

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/categories", response_model=ResponseModel)
    async def get_categories():
        """获取所有可用类别"""
        try:
            service = get_memory_service()
            categories = service.memory_storage.get_all_categories()

            return ResponseModel(
                code=CustomResponseCode.HTTP_200.code,
                msg=CustomResponseCode.HTTP_200.msg,
                data=CategoriesResponse(
                    categories=categories,
                    count=len(categories)
                )
            )

        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/frames/{category}", response_model=ResponseModel)
    async def get_frames_by_category(category: str, max_count: int = 10):
        """根据类别获取记忆帧"""
        try:
            service = get_memory_service()
            frames = service.memory_storage.get_frames_by_category(category, max_count)

            # 转换为响应模型
            frame_data = []
            for frame in frames:
                frame_data.append(FrameInfo(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    categories=frame.categories,
                    metadata=frame.metadata,
                    created_at=frame.created_at,
                    has_image=bool(frame.base64_data)
                ))

            return ResponseModel(
                code=CustomResponseCode.HTTP_200.code,
                msg=CustomResponseCode.HTTP_200.msg,
                data=FramesResponse(
                    category=category,
                    frames=frame_data,
                    count=len(frame_data)
                )
            )

        except Exception as e:
            logger.error(f"Error getting frames for category {category}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/query", response_model=ResponseModel)
    async def query_memory(request: MemoryQueryRequest):
        """查询视觉记忆"""
        try:
            service = get_memory_service()
            question = request.question
            session_id = request.session_id or f"session_{int(time.time())}"

            logger.info(f"Memory query received: {question}")

            # 在后台执行查询
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                service.executor,
                service._process_memory_query,
                question,
                session_id
            )

            return ResponseModel(
                code=CustomResponseCode.HTTP_200.code,
                msg=CustomResponseCode.HTTP_200.msg,
                data=MemoryQueryResponse(**result)
            )

        except Exception as e:
            logger.error(f"Error processing memory query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/query_stream", response_model=ResponseModel)
    async def query_memory_stream(request: MemoryQueryRequest):
        """流式查询视觉记忆"""
        try:
            service = get_memory_service()
            question = request.question
            session_id = request.session_id or f"session_{int(time.time())}"

            async def generate():
                try:
                    # 发送开始信号
                    yield f"data: {json.dumps({'status': 'processing', 'message': '正在查询记忆...'}, ensure_ascii=False)}\n\n"

                    # 在线程池中处理查询
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        service.executor,
                        service._process_memory_query,
                        question,
                        session_id
                    )

                    # 发送结果
                    yield f"data: {json.dumps({'status': 'completed', 'result': result}, ensure_ascii=False)}\n\n"

                except Exception as e:
                    logger.error(f"Error in stream processing: {e}")
                    yield f"data: {json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        except Exception as e:
            logger.error(f"Error setting up memory query stream: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/clear", response_model=ResponseModel)
    async def clear_memory(request: ClearMemoryRequest):
        """清空所有记忆"""
        try:
            if not request.confirm:
                raise HTTPException(
                    status_code=400,
                    detail="Please set confirm=true to clear all memory"
                )

            service = get_memory_service()
            service.memory_storage.clear_all()

            return ResponseModel(
                code=CustomResponseCode.HTTP_200.code,
                msg="All memory cleared successfully",
                data=None
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/latest_frame", response_model=ResponseModel)
    async def get_latest_frame():
        """获取最新的帧信息"""
        try:
            service = get_memory_service()
            frame_info = service.memory_storage.get_latest_frame_info()

            if frame_info:
                # 移除base64数据以节省带宽
                frame_info_copy = frame_info.copy()
                if 'base64_data' in frame_info_copy:
                    frame_info_copy['has_image'] = True
                    del frame_info_copy['base64_data']

                return ResponseModel(
                    code=CustomResponseCode.HTTP_200.code,
                    msg=CustomResponseCode.HTTP_200.msg,
                    data=frame_info_copy
                )
            else:
                return ResponseModel(
                    code=CustomResponseCode.HTTP_200.code,
                    msg="No frames available",
                    data=None
                )

        except Exception as e:
            logger.error(f"Error getting latest frame: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router


def create_memory_api_service(config: Dict[str, Any]) -> MemoryAPIService:
    """创建记忆API服务实例"""
    global _memory_service
    _memory_service = MemoryAPIService(config)
    return _memory_service
