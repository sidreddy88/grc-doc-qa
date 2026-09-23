import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.core.exceptions import ConfigurationError, UpstreamServiceError
from app.models import QuestionType
from app.services.llm import StructuredLLM, TokenUsage

logger = logging.getLogger(__name__)


class ClassifierPath(StrEnum):
    HEURISTIC = "heuristic"
    LLM = "llm"
    DEFAULT = "default"


@dataclass(slots=True)
class Classification:
    question_type: QuestionType
    path: ClassifierPath
    items: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


_CHECKLIST_CUE = re.compile(r"\b(which|any|all|each) of the following\b|\bselect all\b|\bcheck all\b", re.I)
_EXPLAIN_CUE = re.compile(
    r"\b(describe|explain|elaborate|outline|detail|provide details|if (yes|so|applicable)|how do|how does|how is|how are|why)\b",
    re.I,
)
_BOOLEAN_START = re.compile(
    r"^(is|are|do|does|did|has|have|had|was|were|can|could|will|would|should|shall|must|may)\b", re.I
)
_FACTUAL_START = re.compile(
    r"^(which|what|where|who|whom|when|how (many|much|often|long)|list|name|specify|identify|"
    r"please (specify|list|name|identify|provide|state))\b",
    re.I,
)
_INLINE_OPTIONS = re.compile(r"[?:]\s*(?P<options>[^?:]+?)\s*$")
_LEADING_CONJUNCTION = re.compile(r"^(?:or|and)\s+", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•●▪]|\(?[a-z0-9]{1,2}[.)])\s+(?P<item>.+?)\s*$", re.I)


def extract_checklist_items(question: str) -> list[str]:
    """Pull enumerated options from a 'which of the following' question (bullets or a trailing list)."""
    lines = question.splitlines()
    bulleted = [match.group("item") for line in lines[1:] if (match := _BULLET.match(line))]
    if len(bulleted) >= 2:
        return bulleted
    plain_lines = [line.strip() for line in lines[1:] if line.strip()]
    if len(plain_lines) >= 2:
        return plain_lines

    # Inline form: "...following? APM, EUM, or DEM" / "...following: APM; EUM; DEM"
    match = _INLINE_OPTIONS.search(question)
    if not match:
        return []
    parts = [part.strip(" .") for part in re.split(r"[;,]", match.group("options"))]
    if len(parts) < 2:
        parts = [part.strip(" .") for part in re.split(r"\s+or\s+", match.group("options"))]
    items = [_LEADING_CONJUNCTION.sub("", part) for part in parts]
    items = [item for item in items if item]
    return items if len(items) >= 2 else []


def classify_heuristically(question: str) -> Classification | None:
    text = question.strip()
    first_line = text.splitlines()[0] if text else ""

    if _CHECKLIST_CUE.search(first_line):
        items = extract_checklist_items(text)
        if items:
            return Classification(QuestionType.CHECKLIST, ClassifierPath.HEURISTIC, items=items)

    # Several questions in one item, or an explicit request for detail, needs a descriptive answer.
    if text.count("?") > 1 or _EXPLAIN_CUE.search(text):
        return Classification(QuestionType.EXPLANATORY, ClassifierPath.HEURISTIC)
    if _BOOLEAN_START.match(first_line):
        return Classification(QuestionType.BOOLEAN, ClassifierPath.HEURISTIC)
    if _FACTUAL_START.match(first_line):
        return Classification(QuestionType.FACTUAL, ClassifierPath.HEURISTIC)
    return None


class _ClassifierOutput(BaseModel):
    question_type: Literal["boolean", "factual", "explanatory", "checklist"] = Field(
        description="boolean: yes/no; factual: a specific fact or list; explanatory: a description of a process "
        "or policy; checklist: asks which of several enumerated options apply."
    )
    items: list[str] = Field(description="For checklist questions, the enumerated options; otherwise empty.")


_CLASSIFIER_SYSTEM = (
    "You classify security-questionnaire questions by the shape of answer they need. "
    "Return only the classification; do not answer the question."
)


class QueryClassifier:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    async def classify(self, question: str) -> Classification:
        heuristic = classify_heuristically(question)
        if heuristic is not None:
            return heuristic
        try:
            result = await self._llm.generate(_CLASSIFIER_SYSTEM, f"Question:\n{question}", _ClassifierOutput)
        except ConfigurationError:
            raise
        except UpstreamServiceError:
            # Explanatory is the safe default: widest retrieval and the most complete answer format.
            logger.warning("classifier_fallback_failed")
            return Classification(QuestionType.EXPLANATORY, ClassifierPath.DEFAULT)

        question_type = QuestionType(result.value.question_type)
        items = [item.strip() for item in result.value.items if item.strip()]
        if question_type is QuestionType.CHECKLIST and len(items) < 2:
            question_type = QuestionType.EXPLANATORY
        return Classification(question_type, ClassifierPath.LLM, items=items, usage=result.usage)
