import json
import os
from pathlib import Path
from typing import Final

import litellm
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import APIError
from pydantic import BaseModel, ConfigDict, Field

from grounded_agent.agent import (
    SOURCES,
    ChatResult,
    run_agent,
)
from grounded_agent.results import GroundingProvider, GroundingVerdict

litellm.api_base = os.environ.get("LITELLM_API_BASE", "http://localhost:4000/v1")
litellm.api_key = os.environ.get("LITELLM_API_KEY")
app: Final = FastAPI(title="Grounded agent starter")
DEMO: Final = os.environ.get("GROUNDING_DEMO_MODE") == "true"
GROUNDING_PROVIDER: Final = GroundingProvider(os.environ.get("GROUNDING_PROVIDER", "judge"))


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    question: str = Field(min_length=1, max_length=2000)


class FixtureMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: str
    content: str | None = None


class FixtureRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    messages: tuple[FixtureMessage, ...]
    tools: tuple[object, ...] | None = None
    response_format: object | None = None


class JudgePayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    answer: str
    sources: tuple[FixtureMessage, ...]


def fixture_verdict(request: FixtureRequest) -> str:
    payload: Final = JudgePayload.model_validate_json(request.messages[-1].content or "{}")
    if "unlimited" in payload.answer.lower() or "no limit" in payload.answer.lower():
        return GroundingVerdict(
            is_grounded=False,
            reason="The answer claims unlimited remote work. The handbook allows up to three remote days per week.",
            source_ids=("remote-work",),
        ).model_dump_json()
    source_id: Final = "remote-work" if "remote" in payload.answer.lower() else "home-office"
    return GroundingVerdict(
        is_grounded=True,
        reason=(
            "The retrieved handbook supports the remote work schedule and manager exception."
            if source_id == "remote-work"
            else "The retrieved policy supports the allowance, covered equipment, and receipt deadline."
        ),
        source_ids=(source_id,),
    ).model_dump_json()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/brand.svg")
async def brand() -> FileResponse:
    return FileResponse(Path(__file__).with_name("brand.svg"), media_type="image/svg+xml")


@app.get("/api/documents")
async def documents() -> dict[str, object]:
    return {"sources": [source.model_dump() for source in SOURCES], "demo": DEMO}


@app.post("/api/chat", response_model=ChatResult)
async def chat(request: ChatRequest) -> ChatResult:
    try:
        return await run_agent(request.question, provider=GROUNDING_PROVIDER)
    except (APIError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail="The agent could not answer. Check the server configuration.",
        ) from error


def fixture_answer(question: str) -> str:
    if "unlimited" in question.lower():
        return "You can work remotely every day with no limit. Office attendance is optional."
    if "remote" in question.lower():
        return (
            "You can work remotely up to three days each week. Teams meet in the office on Tuesdays and Thursdays. "
            "Your manager can approve an exception for a specific week."
        )
    return (
        "You receive a one-time home office allowance of $500. It covers a desk, chair, monitor, keyboard, or mouse. "
        "Submit your receipts within 30 days of purchase."
    )


def fixture_message(request: FixtureRequest) -> dict[str, object]:
    question: Final = next(
        (message.content or "" for message in request.messages if message.role == "user"),
        "",
    )
    if not request.tools:
        return {
            "role": "assistant",
            "content": (fixture_verdict(request) if request.response_format else fixture_answer(question)),
        }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "fixture-search",
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "arguments": json.dumps({"query": question}),
                },
            }
        ],
    }


@app.post("/fixture/v1/chat/completions")
async def fixture(request: FixtureRequest) -> dict[str, object]:
    if not DEMO:
        raise HTTPException(status_code=404)
    return {
        "id": "fixture-completion",
        "object": "chat.completion",
        "created": 0,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": fixture_message(request),
                "finish_reason": "tool_calls" if request.tools else "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
