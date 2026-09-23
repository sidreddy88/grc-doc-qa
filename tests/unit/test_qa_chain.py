from app.models import NOT_FOUND_ANSWER, Chunk, QuestionType
from app.services.qa_chain import AnswerSynthesizer, build_user_prompt
from tests.fakes import ScriptedLLM

CONTEXTS = [
    Chunk(chunk_id=0, text="GCP Cloud hosting provider", page=17, section="3.4 Third Party Access:"),
    Chunk(chunk_id=1, text="question: Where is data stored?\nanswer: US Central", source_id="row-9"),
]


def test_prompt_labels_sources_with_location_and_wraps_them_as_data():
    prompt = build_user_prompt("Which cloud providers do you rely on?", QuestionType.FACTUAL, [], CONTEXTS)
    assert prompt.startswith("<sources>")
    assert "[S1] (page 17, section: 3.4 Third Party Access:)\nGCP Cloud hosting provider" in prompt
    assert "[S2] (record row-9)" in prompt
    assert "Question type: factual." in prompt
    assert prompt.rstrip().endswith("Which cloud providers do you rely on?")


def test_checklist_prompt_lists_options():
    prompt = build_user_prompt("Which of the following?", QuestionType.CHECKLIST, ["APM", "EUM"], CONTEXTS)
    assert "Options to evaluate:\n- APM\n- EUM" in prompt


async def test_citations_resolve_labels_to_chunks_and_drop_unknown_labels():
    llm = ScriptedLLM({"_SynthesisOutput": [{
        "supported": True, "answer": "GCP.", "items": [],
        "citations": [{"source": "[s1]", "quote": "GCP Cloud hosting"}, {"source": "S9", "quote": "made up"}],
    }]})
    draft = await AnswerSynthesizer(llm).synthesize("Which cloud?", QuestionType.FACTUAL, [], CONTEXTS)
    assert [(c.chunk.page, c.quote) for c in draft.citations] == [(17, "GCP Cloud hosting")]


async def test_unsupported_output_is_normalized_to_not_found_without_citations():
    llm = ScriptedLLM({"_SynthesisOutput": [{
        "supported": False, "answer": "Probably AWS.", "items": [],
        "citations": [{"source": "S1", "quote": "GCP Cloud hosting"}],
    }]})
    draft = await AnswerSynthesizer(llm).synthesize("Which cloud?", QuestionType.FACTUAL, [], CONTEXTS)
    assert (draft.supported, draft.answer, draft.citations) == (False, NOT_FOUND_ANSWER, [])


def _item(name: str, supported: bool = True) -> dict:
    return {"item": name, "supported": supported, "answer": "Yes." if supported else NOT_FOUND_ANSWER,
            "citations": [{"source": "S1", "quote": "GCP Cloud hosting"}] if supported else []}


async def test_checklist_items_match_by_name_and_never_borrow_another_options_answer():
    llm = ScriptedLLM({"_SynthesisOutput": [{
        "supported": True, "answer": "summary", "citations": [], "items": [_item("end user monitoring (eum)")],
    }]})
    draft = await AnswerSynthesizer(llm).synthesize(
        "Which of the following?", QuestionType.CHECKLIST, ["APM", "End User Monitoring (EUM)"], CONTEXTS
    )
    assert [(i.item, i.supported) for i in draft.items] == [("APM", False), ("End User Monitoring (EUM)", True)]


async def test_checklist_items_fall_back_to_position_when_counts_match():
    llm = ScriptedLLM({"_SynthesisOutput": [{
        "supported": True, "answer": "summary", "citations": [],
        "items": [_item("Application Performance Monitoring"), _item("End-user monitoring", supported=False)],
    }]})
    draft = await AnswerSynthesizer(llm).synthesize("Which?", QuestionType.CHECKLIST, ["APM", "EUM"], CONTEXTS)
    assert [(i.item, i.supported) for i in draft.items] == [("APM", True), ("EUM", False)]


async def test_checklist_option_the_model_skipped_is_not_found():
    llm = ScriptedLLM({"_SynthesisOutput": [{"supported": False, "answer": "none", "citations": [], "items": []}]})
    draft = await AnswerSynthesizer(llm).synthesize("Which?", QuestionType.CHECKLIST, ["APM", "EUM"], CONTEXTS)
    assert [(i.answer, i.supported) for i in draft.items] == [(NOT_FOUND_ANSWER, False)] * 2
    assert draft.supported is False
