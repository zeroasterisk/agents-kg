"""Pydantic request/response models for the agents-kg REST API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1)
    source_type: Literal["authoritative", "theoretical"] = "authoritative"


class IngestResponse(BaseModel):
    job_ids: list[int]
    submitted_by: str
    status: str = "queued"


class JobStatus(BaseModel):
    job_id: int
    uri: str
    status: str
    stage: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class HistoryItem(BaseModel):
    job_id: int
    uri: str
    status: str
    stage: str | None = None
    submitted_by: str | None = None
    source_type: str | None = None
    created_at: str
    updated_at: str


class QueryRequest(BaseModel):
    cypher: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    results: list[dict]


class HealthResponse(BaseModel):
    status: str
    neo4j: str
