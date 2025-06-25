#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from fastapi import APIRouter
from app.memory_router import create_memory_router
# from app.task.api.router import v1 as task_v1


router = APIRouter()

# 包含记忆查询API路由
memory_router = create_memory_router()
router.include_router(memory_router, tags=['memory'])

# router.include_router(task_v1, prefix='/task', tags=['task'])
