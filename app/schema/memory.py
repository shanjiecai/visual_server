from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


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
    