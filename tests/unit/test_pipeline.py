import json

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, UpstreamServiceError
from app.models import NOT_FOUND_ANSWER, DocumentType
from app.services.pipeline import QAPipeline
from tests.fakes import (
    ScriptedLLM,
    approve_all,
    grounded_synthesis,
    make_test_pipeline,
    parse_sources,
    reject_all,
)

KB = [
    {"id": "cloud", "question": "Which cloud provider hosts the service?", "answer": "Google Cloud Platform (GCP) hosts all production workloads."},
    {"id": "enc", "question": "Is data encrypted at rest?", "answer": "Yes, AES-256 encryption is applied to all disks at rest."},
    {"id": "apm", "question": "Is application performance monitoring used?", "answer": "Application performance monitoring alerts on latency and error rates."},
]
DOCUMENT = json.dumps(KB).encode()


def make_pipeline(llm: ScriptedLLM, **overrides) -> QAPipeline:
    settings = Settings(_env_file=None, retrieval_candidates=3, rerank_candidates=3, top_k_default=2, **overrides)
    return make_test_pipeline(settings, llm)


async def run(pipeline: QAPipeline, *questions: str):
    return await pipeline.answer(list(questions), DOCUMENT, DocumentType.JSON)


async def test_grounded_answer_carries_verified_citation_metadata():
    llm = ScriptedLLM({"_SynthesisOutput": [grounded_synthesis("GCP hosts the service.")], "_JudgeOutput": [approve_all]})
    output = await run(make_pipeline(llm), "Which cloud provider hosts the service?")
    [result] = output.results
    assert result.answer == "GCP hosts the service."
    assert result.citations[0].source_id == "cloud"
    assert result.citations[0].excerpt.startswith("question: Which cloud provider")
    assert output.stats.usage.calls == 2  # synthesis + judge; heuristics classified the question


async def test_fabricated_quote_fails_layer_one_without_calling_the_judge():
    fabricated = {"supported": True, "answer": "AWS hosts it.", "items": [],
                  "citations": [{"source": "S1", "quote": "Amazon Web Services hosts all production workloads"}]}
    llm = ScriptedLLM({"_SynthesisOutput": [fabricated]})
    [result] = (await run(make_pipeline(llm), "Which cloud provider hosts the service?")).results
    assert result.answer == NOT_FOUND_ANSWER
    assert result.citations == []
    assert llm.calls_for("_JudgeOutput") == []


async def test_real_quote_supporting_a_different_claim_fails_layer_two():
    # The quote exists verbatim, so Layer 1 passes; the claim goes beyond it, so the judge must reject it.
    overreach = grounded_synthesis("GCP hosts the service with 99.99% uptime in three regions.")
    llm = ScriptedLLM({"_SynthesisOutput": [overreach], "_JudgeOutput": [reject_all]})
    [result] = (await run(make_pipeline(llm), "Which cloud provider hosts the service?")).results
    assert result.answer == NOT_FOUND_ANSWER
    assert len(llm.calls_for("_JudgeOutput")) == 1


async def test_model_reporting_not_found_is_returned_verbatim():
    llm = ScriptedLLM({"_SynthesisOutput": [{"supported": False, "answer": NOT_FOUND_ANSWER, "citations": [], "items": []}]})
    [result] = (await run(make_pipeline(llm), "Is data encrypted at rest?")).results
    assert result.answer == NOT_FOUND_ANSWER


async def test_irrelevant_question_skips_all_llm_calls():
    llm = ScriptedLLM()
    [result] = (await run(make_pipeline(llm), "What is the capital of France?")).results
    assert result.answer == NOT_FOUND_ANSWER
    assert llm.calls == []


async def test_checklist_reports_each_option_and_rejects_unsupported_ones():
    def checklist(system, user):
        sources = parse_sources(user)
        apm_label = next(label for label, text in sources.items() if "Application performance monitoring alerts" in text)
        return {"supported": True, "answer": "One of three is performed.", "citations": [], "items": [
            {"item": "Application Performance Monitoring (APM)", "supported": True, "answer": "Yes, APM alerts on latency.",
             "citations": [{"source": apm_label, "quote": "Application performance monitoring alerts on latency"}]},
            {"item": "End User Monitoring (EUM)", "supported": False, "answer": NOT_FOUND_ANSWER, "citations": []},
            {"item": "Digital Experience Monitoring (DEM)", "supported": True, "answer": "Yes.",
             "citations": [{"source": "S1", "quote": "Digital experience monitoring is performed weekly"}]},
        ]}

    llm = ScriptedLLM({"_SynthesisOutput": [checklist], "_JudgeOutput": [approve_all]})
    question = ("Which of the following, if any, are performed as part of your monitoring process for the service?\n"
                "- Application Performance Monitoring (APM)\n- End User Monitoring (EUM)\n- Digital Experience Monitoring (DEM)")
    [result] = (await run(make_pipeline(llm), question)).results
    assert [(i.item, i.supported) for i in result.items] == [
        ("Application Performance Monitoring (APM)", True),
        ("End User Monitoring (EUM)", False),
        ("Digital Experience Monitoring (DEM)", False),  # fabricated quote dropped by Layer 1
    ]
    assert result.citations[0].source_id == "apm"
    assert result.answer.splitlines()[0] == "Application Performance Monitoring (APM): Yes, APM alerts on latency."


async def test_each_part_of_a_multi_part_question_gets_its_own_evidence():
    llm = ScriptedLLM({"_SynthesisOutput": [grounded_synthesis()], "_JudgeOutput": [approve_all]})
    settings = Settings(_env_file=None, retrieval_candidates=3, rerank_candidates=3,
                        top_k_default=1, top_k_explanatory=1, top_k_question_part=1)
    question = "Which cloud provider hosts the service, as well as how data is encrypted at rest?"
    await make_test_pipeline(settings, llm).answer([question], DOCUMENT, DocumentType.JSON)
    sources = parse_sources(llm.calls_for("_SynthesisOutput")[0]).values()
    # With k=1 a single retrieval pass would surface only one of these records.
    assert any("Google Cloud Platform" in text for text in sources)
    assert any("AES-256" in text for text in sources)


_NO_OPTION = {"supported": False, "answer": NOT_FOUND_ANSWER, "citations": [], "items": [
    {"item": "End User Monitoring (EUM)", "supported": False, "answer": NOT_FOUND_ANSWER, "citations": []},
    {"item": "Digital Experience Monitoring (DEM)", "supported": False, "answer": NOT_FOUND_ANSWER, "citations": []},
]}
_MONITORING = ("Which of the following, if any, are performed as part of your monitoring process for the service?\n"
               "- End User Monitoring (EUM)\n- Digital Experience Monitoring (DEM)")


async def test_checklist_with_no_matching_option_is_asked_again_as_an_open_question():
    summary = "Neither is named; application performance monitoring alerts on latency and error rates."
    llm = ScriptedLLM({"_SynthesisOutput": [_NO_OPTION, grounded_synthesis(summary)], "_JudgeOutput": [approve_all]})
    [result] = (await run(make_pipeline(llm), _MONITORING)).results
    assert result.answer == summary
    assert result.citations
    assert [(i.item, i.supported) for i in result.items] == [
        ("End User Monitoring (EUM)", False),
        ("Digital Experience Monitoring (DEM)", False),
    ]
    assert "Question type: explanatory" in llm.calls_for("_SynthesisOutput")[1]


async def test_checklist_fallback_that_finds_nothing_stays_not_found():
    not_found = {"supported": False, "answer": NOT_FOUND_ANSWER, "citations": [], "items": []}
    llm = ScriptedLLM({"_SynthesisOutput": [_NO_OPTION, not_found]})
    [result] = (await run(make_pipeline(llm), _MONITORING)).results
    assert result.answer == NOT_FOUND_ANSWER
    assert [i.supported for i in result.items] == [False, False]
    assert llm.calls_for("_JudgeOutput") == []


async def test_duplicate_questions_are_answered_once_and_returned_in_order():
    llm = ScriptedLLM({"_SynthesisOutput": [grounded_synthesis()], "_JudgeOutput": [approve_all]})
    q = "Is data encrypted at rest?"
    output = await run(make_pipeline(llm), q, "Which cloud provider hosts the service?", q)
    assert [r.question for r in output.results] == [q, "Which cloud provider hosts the service?", q]
    assert output.stats.unique_questions == 2
    assert len(llm.calls_for("_SynthesisOutput")) == 2


async def test_one_failing_question_does_not_sink_the_others():
    def flaky(system, user):
        if "encrypted" in user.split("Question:")[-1]:
            raise UpstreamServiceError("timeout")
        return grounded_synthesis()(system, user)

    llm = ScriptedLLM({"_SynthesisOutput": [flaky], "_JudgeOutput": [approve_all]})
    ok, failed = (await run(make_pipeline(llm), "Which cloud provider hosts the service?", "Is data encrypted at rest?")).results
    assert ok.error is None and ok.citations
    assert failed.error.code == "upstream_unavailable"


async def test_all_questions_failing_is_a_request_level_upstream_error():
    llm = ScriptedLLM({"_SynthesisOutput": [UpstreamServiceError("down")]})
    with pytest.raises(UpstreamServiceError):
        await run(make_pipeline(llm), "Is data encrypted at rest?")


async def test_configuration_error_aborts_the_request():
    llm = ScriptedLLM({"_SynthesisOutput": [ConfigurationError("no key")]})
    with pytest.raises(ConfigurationError):
        await run(make_pipeline(llm), "Is data encrypted at rest?", "Which cloud provider hosts the service?")


async def test_repeat_request_is_served_from_the_answer_cache():
    llm = ScriptedLLM({"_SynthesisOutput": [grounded_synthesis()], "_JudgeOutput": [approve_all]})
    pipeline = make_pipeline(llm)
    first = await run(pipeline, "Is data encrypted at rest?")
    calls_after_first = len(llm.calls)
    second = await run(pipeline, "Is data encrypted at rest?")
    assert len(llm.calls) == calls_after_first
    assert second.results[0].answer == first.results[0].answer
    assert (first.stats.answer_cache_hits, second.stats.answer_cache_hits) == (0, 1)
    assert second.stats.index_cache_hit is True


async def test_reworded_question_hits_semantically_and_reports_the_asked_wording():
    llm = ScriptedLLM({"_SynthesisOutput": [grounded_synthesis()], "_JudgeOutput": [approve_all]})
    pipeline = make_pipeline(llm)
    await run(pipeline, "Is data encrypted at rest?")
    # Fake embeddings drop stopwords, so this rewording embeds identically but normalizes differently.
    output = await run(pipeline, "Is the data encrypted at rest?")
    assert output.stats.answer_cache_hits == 1
    assert output.results[0].question == "Is the data encrypted at rest?"
    assert len(llm.calls_for("_SynthesisOutput")) == 1
