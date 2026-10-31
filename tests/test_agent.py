from typing import Final

import pytest
from pydantic import TypeAdapter

from grounded_agent.agent import (
    AllMessageValues,
    ChatCompletionToolParam,
    CheckStatus,
    ModelResponse,
    Source,
    run_agent,
    search_documents,
)
from grounded_agent.results import EvidenceType, GroundingProvider


async def complete_fixture(
    messages: list[AllMessageValues], tools: list[ChatCompletionToolParam] | None
) -> ModelResponse:
    if tools:
        return ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "search-1",
                                "type": "function",
                                "function": {
                                    "name": "search_documents",
                                    "arguments": '{"query":"home office equipment allowance"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        )
    return ModelResponse(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The allowance is $500."},
            }
        ]
    )


@pytest.mark.asyncio
async def test_should_return_the_guardrails_verdict_with_the_retrieved_evidence() -> None:
    async def check(answer: str, messages: list[AllMessageValues]) -> object:
        assert answer == "The allowance is $500."
        message: Final = TypeAdapter(dict[str, str]).validate_python(messages[0])
        evidence: Final = TypeAdapter(tuple[Source, ...]).validate_json(message["content"])
        assert any(source.id == "home-office" for source in evidence)
        return {
            "is_grounded": True,
            "reason": "The policy supports the amount.",
            "source_ids": ["home-office"],
        }

    result: Final = await run_agent("What is the equipment allowance?", complete_fixture, check)
    assert result.status is CheckStatus.grounded
    assert result.verdict is not None
    assert result.verdict.source_ids == ("home-office",)
    assert any(source.id == "home-office" for source in result.sources)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_result, expected_status",
    [
        (
            {
                "is_grounded": False,
                "reason": "The answer exceeds the policy limit.",
                "source_ids": ["home-office"],
            },
            CheckStatus.unsupported,
        ),
        (
            {
                "is_grounded": "true",
                "reason": "A string is not a verdict.",
                "source_ids": ["home-office"],
            },
            CheckStatus.unavailable,
        ),
        (
            {"is_grounded": True, "reason": "Missing evidence.", "source_ids": []},
            CheckStatus.unavailable,
        ),
        (
            {
                "is_grounded": True,
                "reason": "Invented citation.",
                "source_ids": ["unretrieved"],
            },
            CheckStatus.unavailable,
        ),
        ({"response_text": "The allowance is $500."}, CheckStatus.unavailable),
    ],
)
async def test_should_never_show_a_grounded_badge_for_an_unsupported_or_invalid_result(
    native_result: object, expected_status: CheckStatus
) -> None:
    async def check(answer: str, messages: list[AllMessageValues]) -> object:
        return native_result

    result: Final = await run_agent("What is the equipment allowance?", complete_fixture, check)
    assert result.status is expected_status
    assert result.answer == "The allowance is $500."


@pytest.mark.asyncio
async def test_should_preserve_the_answer_without_a_badge_when_the_check_fails() -> None:
    async def check(answer: str, messages: list[AllMessageValues]) -> object:
        raise ValueError("The guardrail returned malformed JSON")

    result: Final = await run_agent("What is the equipment allowance?", complete_fixture, check)
    assert result.status is CheckStatus.unavailable
    assert result.verdict is None
    assert result.answer == "The allowance is $500."


def test_should_return_no_documents_when_the_query_has_no_matching_evidence() -> None:
    assert search_documents("astronomy quasars nebulae") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "detected", "expected_status"),
    [(0.95, False, CheckStatus.grounded), (0.3, True, CheckStatus.unsupported), (0.95, True, CheckStatus.unsupported)],
)
async def test_should_read_bedrock_grounding_scores_without_treating_report_only_action_as_a_pass(
    score: float, detected: bool, expected_status: CheckStatus
) -> None:
    question: Final = "What is the equipment allowance?"

    async def check(answer: str, messages: list[AllMessageValues]) -> object:
        assert messages[0]["role"] == "system"
        assert messages[-1] == {"role": "user", "content": question}
        return {
            "action": "NONE",
            "assessments": [
                {
                    "contextualGroundingPolicy": {
                        "filters": [{"type": "GROUNDING", "score": score, "threshold": 0.8, "detected": detected}]
                    }
                }
            ],
        }

    result: Final = await run_agent(question, complete_fixture, check, provider=GroundingProvider.bedrock)
    assert result.status is expected_status
    assert result.evidence_type is EvidenceType.checked
    assert result.verdict is not None
    assert result.verdict.source_ids == tuple(source.id for source in result.sources)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_result",
    [
        {"action": "NONE", "assessments": []},
        {"action": "NONE"},
        {
            "assessments": [
                {"contextualGroundingPolicy": {"filters": [{"type": "RELEVANCE", "score": 0.95, "threshold": 0.8}]}}
            ]
        },
        [
            {
                "assessments": [
                    {"contextualGroundingPolicy": {"filters": [{"type": "GROUNDING", "score": 0.95, "threshold": 0.8}]}}
                ]
            },
            {"assessments": []},
        ],
    ],
)
async def test_should_show_unavailable_when_bedrock_did_not_evaluate_grounding(native_result: object) -> None:
    async def check(answer: str, messages: list[AllMessageValues]) -> object:
        return native_result

    result: Final = await run_agent(
        "What is the equipment allowance?", complete_fixture, check, provider=GroundingProvider.bedrock
    )
    assert result.status is CheckStatus.unavailable
    assert result.verdict is None
    assert result.evidence_type is EvidenceType.checked
