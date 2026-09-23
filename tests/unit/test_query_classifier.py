import json
from pathlib import Path

import pytest

from app.core.exceptions import ConfigurationError, UpstreamServiceError
from app.models import QuestionType
from app.services.query_classifier import (
    ClassifierPath,
    QueryClassifier,
    classify_heuristically,
    extract_checklist_items,
)
from tests.fakes import ScriptedLLM

SAMPLE_QUESTIONS = json.loads((Path(__file__).parent.parent / "fixtures" / "sample_questions.json").read_text())


def test_spec_sample_questions_are_all_classified_without_an_llm_call():
    types = [classify_heuristically(q).question_type for q in SAMPLE_QUESTIONS]
    assert types == [
        QuestionType.EXPLANATORY,  # two questions: criteria + SLAs
        QuestionType.EXPLANATORY,  # "If yes, describe."
        QuestionType.FACTUAL,
        QuestionType.FACTUAL,      # "Please specify ..."
        QuestionType.CHECKLIST,
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is data encrypted at rest?", QuestionType.BOOLEAN),
        ("Does the Hypervisor lock accounts after 3-5 invalid login attempts?", QuestionType.BOOLEAN),
        ("Where are your data centres located?", QuestionType.FACTUAL),
        ("How many employees have production access?", QuestionType.FACTUAL),
        ("How do you manage encryption keys?", QuestionType.EXPLANATORY),
        ("Do you have an incident response plan? If so, describe it.", QuestionType.EXPLANATORY),
    ],
)
def test_heuristic_rules(question, expected):
    result = classify_heuristically(question)
    assert result.question_type is expected
    assert result.path is ClassifierPath.HEURISTIC


def test_unmatched_phrasing_returns_none():
    assert classify_heuristically("Encryption key rotation schedule.") is None


@pytest.mark.parametrize(
    ("question", "items"),
    [
        (SAMPLE_QUESTIONS[4], [
            "Application Performance Monitoring (APM)",
            "End User Monitoring (EUM)",
            "Digital Experience Monitoring (DEM)",
        ]),
        ("Which of the following are used? APM, EUM, or DEM", ["APM", "EUM", "DEM"]),
        ("Which of the following apply: SSO; MFA; SCIM", ["SSO", "MFA", "SCIM"]),
        ("Which of the following apply?\n1. Daily backups\n2. Weekly restore tests", ["Daily backups", "Weekly restore tests"]),
    ],
)
def test_extract_checklist_items(question, items):
    assert extract_checklist_items(question) == items


def test_checklist_cue_without_items_is_not_a_checklist():
    result = classify_heuristically("Which of the following controls do you have?")
    assert result.question_type is not QuestionType.CHECKLIST


async def test_llm_fallback_used_only_when_heuristics_miss():
    llm = ScriptedLLM({"_ClassifierOutput": [{"question_type": "boolean", "items": []}]})
    classifier = QueryClassifier(llm)

    heuristic = await classifier.classify("Is MFA enforced?")
    assert heuristic.path is ClassifierPath.HEURISTIC
    assert llm.calls == []

    fallback = await classifier.classify("Encryption key rotation schedule.")
    assert (fallback.question_type, fallback.path) == (QuestionType.BOOLEAN, ClassifierPath.LLM)
    assert fallback.usage.calls == 1


async def test_llm_checklist_without_items_downgrades_to_explanatory():
    llm = ScriptedLLM({"_ClassifierOutput": [{"question_type": "checklist", "items": ["only one"]}]})
    result = await QueryClassifier(llm).classify("Monitoring tooling in use.")
    assert result.question_type is QuestionType.EXPLANATORY


async def test_llm_failure_defaults_to_explanatory():
    llm = ScriptedLLM({"_ClassifierOutput": [UpstreamServiceError("down")]})
    result = await QueryClassifier(llm).classify("Monitoring tooling in use.")
    assert (result.question_type, result.path) == (QuestionType.EXPLANATORY, ClassifierPath.DEFAULT)


async def test_configuration_errors_are_not_swallowed():
    llm = ScriptedLLM({"_ClassifierOutput": [ConfigurationError("no key")]})
    with pytest.raises(ConfigurationError):
        await QueryClassifier(llm).classify("Monitoring tooling in use.")
