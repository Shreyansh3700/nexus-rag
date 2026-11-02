from typing import Any

import asyncpg
from arq.connections import ArqRedis
from fastapi import Request
from langchain_core.runnables import Runnable
from pymilvus import MilvusClient


def get_db(request: Request) -> asyncpg.Pool:
    return request.app.state.db


def get_chain(request: Request) -> Runnable[Any, Any]:
    return request.app.state.chain


def get_title_chain(request: Request) -> Runnable[Any, Any]:
    return request.app.state.title_chain


def get_milvus(request: Request) -> MilvusClient:
    return request.app.state.milvus


def get_redis(request: Request) -> ArqRedis:
    return request.app.state.redis
