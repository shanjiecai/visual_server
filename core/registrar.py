#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import Depends, FastAPI

from common.exception.exception_handler import register_exception
from common.log import set_custom_logfile, setup_logging
from core.conf import settings
from middleware.state_middleware import StateMiddleware
from app.router import router as main_router


@asynccontextmanager
async def register_init(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    启动初始化

    :param app: FastAPI 应用实例
    :return:
    """
    # 创建数据库表
    yield


def register_app() -> FastAPI:
    """注册 FastAPI 应用"""
    app = FastAPI(
        title=settings.FASTAPI_TITLE,
        version=settings.FASTAPI_VERSION,
        description=settings.FASTAPI_DESCRIPTION,
        docs_url=settings.FASTAPI_DOCS_URL,
        redoc_url=settings.FASTAPI_REDOC_URL,
        openapi_url=settings.FASTAPI_OPENAPI_URL,
        lifespan=register_init,
    )

    # 注册组件
    register_logger()
    register_middleware(app)
    register_router(app)
    register_exception(app)

    return app


def register_logger() -> None:
    """注册日志"""
    setup_logging()
    set_custom_logfile()


def register_middleware(app: FastAPI) -> None:
    """
    注册中间件（执行顺序从下往上）

    :param app: FastAPI 应用实例
    :return:
    """

    # State
    app.add_middleware(StateMiddleware)

    # Trace ID (必须)
    app.add_middleware(CorrelationIdMiddleware, validator=False)

    # CORS（必须放在最下面）
    if settings.MIDDLEWARE_CORS:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
            expose_headers=settings.CORS_EXPOSE_HEADERS,
        )


def register_router(app: FastAPI) -> None:
    """
    注册路由

    :param app: FastAPI 应用实例
    :return:
    """
    dependencies = []

    # API
    app.include_router(main_router, dependencies=dependencies)
