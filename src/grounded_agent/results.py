from enum import Enum
from itertools import chain
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class GroundingProvider(str, Enum):
    judge = "judge"
    bedrock = "bedrock"


class EvidenceType(str, Enum):
    cited = "cited"
    checked = "checked"


class GroundingVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    is_grounded: bool = Field(strict=True)
    reason: str = Field(min_length=1)
    source_ids: tuple[str, ...]


class _FilterType(str, Enum):
    grounding = "GROUNDING"
    relevance = "RELEVANCE"


class _BedrockFilter(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: _FilterType
    score: float = Field(strict=True, ge=0, le=1)
    threshold: float = Field(strict=True, ge=0, le=1)
    detected: bool | None = Field(default=None, strict=True)


class _BedrockPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)
    filters: tuple[_BedrockFilter, ...] = ()


class _BedrockAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)
    grounding: _BedrockPolicy | None = Field(default=None, alias="contextualGroundingPolicy")


class _BedrockResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    assessments: tuple[_BedrockAssessment, ...] = ()


_BEDROCK_RESULT: Final = TypeAdapter(_BedrockResult | tuple[_BedrockResult, ...])


def _filters(result: _BedrockResult) -> tuple[_BedrockFilter, ...]:
    return tuple(
        chain.from_iterable(
            assessment.grounding.filters for assessment in result.assessments if assessment.grounding is not None
        )
    )


def parse_verdict(
    native_result: object, provider: GroundingProvider, checked_source_ids: tuple[str, ...]
) -> GroundingVerdict:
    if provider is GroundingProvider.judge:
        return GroundingVerdict.model_validate(native_result)

    parsed: Final = _BEDROCK_RESULT.validate_python(native_result)
    results: Final = (parsed,) if isinstance(parsed, _BedrockResult) else parsed
    grounding_by_result: Final = tuple(
        tuple(item for item in _filters(result) if item.type is _FilterType.grounding) for result in results
    )
    if not grounding_by_result or any(not checks for checks in grounding_by_result):
        raise ValueError("Bedrock returned no contextual grounding assessment")
    grounding: Final = tuple(chain.from_iterable(grounding_by_result))
    is_grounded: Final = all(item.score >= item.threshold and item.detected is not True for item in grounding)
    scores: Final = ", ".join(f"{item.score:.2f} (threshold {item.threshold:.2f})" for item in grounding)
    outcome: Final = "passed" if is_grounded else "found unsupported content"
    summary: Final = f"Amazon Bedrock grounding check {outcome}. Scores: {scores}."
    return GroundingVerdict(
        is_grounded=is_grounded,
        reason=f"{summary} The documents below were checked together.",
        source_ids=checked_source_ids,
    )
