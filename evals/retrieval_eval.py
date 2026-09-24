"""Retrieval eval (does the evidence reach the model?) and, with --answers, answer eval.

    python -m evals.retrieval_eval                    # retrieval only: no LLM calls, free
    python -m evals.retrieval_eval --answers          # also run the full pipeline with gpt-4o-mini (~$0.01)
    python -m evals.retrieval_eval --json out.json    # also write per-question results

Runs the service's real indexing and retrieval (classification by the heuristic rules only;
questions they don't match are treated as explanatory, the LLM classifier's fallback).
Settings can be overridden with the usual environment variables, e.g. TOP_K_DEFAULT=8.

Metrics, over answerable questions:
- context recall: share whose evidence is among the chunks sent to the model
- MRR: mean reciprocal rank of the first evidence chunk in that context (0 if absent)
- wrongly gated: answerable questions the relevance gate would answer "Not found" without the LLM
and over unanswerable ones: how many the gate stops before any LLM call.

Answer metrics (--answers; needs OPENAI_API_KEY):
- answered with evidence: answerable questions answered (not "Not found") that cite a page holding evidence
- answered, other pages: answered, but no citation is on an evidence page (possibly wrong, possibly just other support)
- correct "Not found": unanswerable questions answered "Not found"
- sample targets passed: questions with a 'target' whose answer meets all its must_mention / must_not_mention checks
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import Settings
from app.deps import build_pipeline
from app.models import NOT_FOUND_ANSWER, Chunk, DocumentType, QuestionType
from app.services.query_classifier import classify_heuristically
from app.services.text import fold_for_match, normalize_whitespace

EVAL_FILE = Path(__file__).with_name("soc2_retrieval.json")


def _norm(text: str) -> str:
    return fold_for_match(normalize_whitespace(text))


def check_target(item: dict, answer: str) -> list[str]:
    """Failed checks for a question with a target answer; [] means it passes."""
    text = _norm(answer)
    failures = [f"missing one of {group}" for group in item.get("must_mention", [])
                if not any(_norm(phrase) in text for phrase in group)]
    failures += [f"mentions {phrase!r}" for phrase in item.get("must_not_mention", []) if _norm(phrase) in text]
    if answer.strip() == NOT_FOUND_ANSWER:
        failures.insert(0, "answered Not found")
    return failures


def _is_evidence(chunk: Chunk, evidence: list[str]) -> bool:
    text = _norm(chunk.text)
    return any(_norm(phrase) in text for phrase in evidence)


async def run(eval_file: Path = EVAL_FILE, answers: bool = False) -> dict:
    spec = json.loads(eval_file.read_text())
    settings = Settings() if answers else Settings(_env_file=None)  # the answer eval needs the API key from .env
    pipeline = build_pipeline(settings)
    document = (Path(__file__).resolve().parent.parent / spec["document"]).read_bytes()
    index, _ = await pipeline._index_service.get_index(document, DocumentType.PDF)

    # A label that matches no chunk is a broken label (or a quote split across chunks): fail loudly.
    unmatched = [
        (item["id"], phrase)
        for item in spec["questions"]
        for phrase in item["evidence"]
        if not any(_is_evidence(chunk, [phrase]) for chunk in index.chunks)
    ]
    if unmatched:
        raise SystemExit(f"Evidence phrases found in no chunk: {unmatched}")

    results = []
    for item in spec["questions"]:
        question = item["question"]
        classification = classify_heuristically(question)
        question_type = classification.question_type if classification else QuestionType.EXPLANATORY
        items = classification.items if classification else []
        vector = pipeline._embeddings.encode_queries([question])[0]
        hits = await pipeline._retrieve(index, question, vector, question_type, items)
        top_relevance = max((hit.relevance for hit in hits), default=0.0)
        rank = next((i for i, hit in enumerate(hits, 1) if _is_evidence(hit.chunk, item["evidence"])), None)
        results.append({
            "id": item["id"],
            "answerable": item["answerable"],
            "type": question_type.value,
            "contexts": len(hits),
            "evidence_rank": rank,
            "gated": top_relevance < settings.min_relevance,
            "top_relevance": round(top_relevance, 5),
        })

    if answers:
        output = await pipeline.answer([item["question"] for item in spec["questions"]], document, DocumentType.PDF)
        for item, result, answer in zip(spec["questions"], results, output.results, strict=True):
            evidence_pages = {c.page for c in index.chunks if _is_evidence(c, item["evidence"])}
            result["answered"] = answer.answer != NOT_FOUND_ANSWER
            result["cites_evidence"] = any(c.page in evidence_pages for c in answer.citations)
            result["answer"] = answer.answer
            if "target" in item:
                result["target_failures"] = check_target(item, answer.answer)

    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    summary = {
        "answerable": len(answerable),
        "context_recall": round(sum(r["evidence_rank"] is not None for r in answerable) / len(answerable), 3),
        "mrr": round(sum(1 / r["evidence_rank"] for r in answerable if r["evidence_rank"]) / len(answerable), 3),
        "wrongly_gated": sum(r["gated"] for r in answerable),
        "unanswerable": len(unanswerable),
        "unanswerable_gated": sum(r["gated"] for r in unanswerable),
        "mean_contexts": round(sum(r["contexts"] for r in results) / len(results), 1),
    }
    if answers:
        summary["answered_with_evidence"] = sum(r["answered"] and r["cites_evidence"] for r in answerable)
        summary["answered_other_pages"] = sum(r["answered"] and not r["cites_evidence"] for r in answerable)
        summary["answerable_not_found"] = sum(not r["answered"] for r in answerable)
        summary["correct_not_found"] = sum(not r["answered"] for r in unanswerable)
        targeted = [r for r in results if "target_failures" in r]
        summary["sample_targets_passed"] = f"{sum(not r['target_failures'] for r in targeted)}/{len(targeted)}"
        summary["llm_cost_usd"] = round(
            (output.stats.usage.input_tokens * settings.llm_input_usd_per_mtok
             + output.stats.usage.output_tokens * settings.llm_output_usd_per_mtok) / 1_000_000, 4)
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write per-question results to this file")
    parser.add_argument("--answers", action="store_true", help="also run the full pipeline with the LLM")
    args = parser.parse_args()
    report = asyncio.run(run(answers=args.answers))
    for r in report["results"]:
        if r["answerable"]:
            outcome = f"rank {r['evidence_rank']}/{r['contexts']}" if r["evidence_rank"] else f"MISS (0/{r['contexts']})"
        else:
            outcome = "gated" if r["gated"] else f"passes gate (top {r['top_relevance']})"
        flag = "  <- gated" if r["answerable"] and r["gated"] else ""
        if "answered" in r:
            verdict = ("answered+evidence" if r["cites_evidence"] else "answered, other pages") if r["answered"] else "Not found"
            flag += f"  | {verdict}"
        if "target_failures" in r:
            flag += "  | TARGET " + ("PASS" if not r["target_failures"] else "FAIL: " + "; ".join(r["target_failures"]))
        print(f"{r['id']:<22} {r['type']:<11} {outcome}{flag}")
    print("\n" + json.dumps(report["summary"], indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
