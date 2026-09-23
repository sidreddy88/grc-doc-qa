"""Two independent grounding checks applied after synthesis.

Layer 1 (deterministic, free): every cited quote must actually occur in the
chunk it claims to come from. Returned excerpts are always taken from the
document text, never from the model's copy of it.

Layer 2 (one gpt-4o-mini call): an LLM judge checks that the answer's claims
are entailed by the verified excerpts. This catches real quotes attached to
claims they don't support, which Layer 1 cannot see.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from app.models import Chunk
from app.services.llm import StructuredLLM, TokenUsage
from app.services.qa_chain import DraftCitation
from app.services.text import fold_for_match, normalize_whitespace

_MIN_QUOTE_CHARS = 12
_MIN_FUZZY_RATIO = 0.9
_MIN_BLOCK_CHARS = 4
_MAX_SPAN_STRETCH = 1.5


@dataclass(frozen=True, slots=True)
class VerifiedCitation:
    chunk: Chunk
    excerpt: str


def locate_quote(quote: str, chunk_text: str) -> str | None:
    """Return the span of chunk_text that the quote reproduces, or None if it isn't really there.

    Exact match after case/quote/whitespace folding first; otherwise a fuzzy
    match that tolerates small copying slips (dropped punctuation, a changed word)
    but requires at least 90% of the quote to appear in order.
    """
    needle = fold_for_match(normalize_whitespace(quote))
    if len(needle) < _MIN_QUOTE_CHARS:
        return None
    # Only newlines are folded to spaces here, so offsets still index the original text.
    haystack = fold_for_match(chunk_text).replace("\n", " ").replace("\t", " ")

    start = haystack.find(needle)
    if start != -1:
        return chunk_text[start : start + len(needle)]

    blocks = [
        block
        for block in SequenceMatcher(None, needle, haystack, autojunk=False).get_matching_blocks()
        if block.size >= _MIN_BLOCK_CHARS
    ]
    if not blocks or sum(block.size for block in blocks) / len(needle) < _MIN_FUZZY_RATIO:
        return None
    start, end = blocks[0].b, blocks[-1].b + blocks[-1].size
    # Matching fragments scattered across the chunk are not a quote.
    if end - start > len(needle) * _MAX_SPAN_STRETCH:
        return None
    return chunk_text[start:end]


def verify_citations(citations: list[DraftCitation]) -> list[VerifiedCitation]:
    verified: list[VerifiedCitation] = []
    seen: set[tuple[int, str]] = set()
    for citation in citations:
        chunk, quote = citation.chunk, citation.quote
        excerpt = locate_quote(quote, chunk.text)
        if excerpt is None:
            continue
        key = (chunk.chunk_id, excerpt)
        if key not in seen:
            seen.add(key)
            verified.append(VerifiedCitation(chunk=chunk, excerpt=excerpt))
    return verified


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    text: str
    evidence: list[str]


class _Verdict(BaseModel):
    claim_id: str
    faithful: bool
    reason: str = Field(description="One short sentence explaining the verdict.")


class _JudgeOutput(BaseModel):
    verdicts: list[_Verdict]


JUDGE_SYSTEM_PROMPT = """You verify answers produced for a compliance questionnaire. For each claim you are \
given the evidence excerpts it cites. Decide whether the evidence fully supports the claim.

A claim is NOT faithful if it states anything the evidence does not: extra specifics (names, numbers, \
frequencies, durations), a stronger or broader statement than the evidence makes, or a responsibility \
attributed to the wrong party. Paraphrasing is fine. Statements that the document does not address some \
part of the question are acceptable and should not count against the claim.
Return one verdict per claim_id."""


def build_judge_prompt(question: str, claims: list[Claim]) -> str:
    blocks = []
    for claim in claims:
        evidence = "\n".join(f"  - {excerpt}" for excerpt in claim.evidence)
        blocks.append(f"claim_id: {claim.claim_id}\nclaim: {claim.text}\nevidence:\n{evidence}")
    return f"Question:\n{question}\n\n" + "\n\n".join(blocks)


class FaithfulnessJudge:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    async def judge(self, question: str, claims: list[Claim]) -> tuple[dict[str, _Verdict], TokenUsage]:
        """Return verdicts keyed by claim_id. Claims the judge omits are treated as unfaithful by callers."""
        if not claims:
            return {}, TokenUsage()
        result = await self._llm.generate(JUDGE_SYSTEM_PROMPT, build_judge_prompt(question, claims), _JudgeOutput)
        return {verdict.claim_id: verdict for verdict in result.value.verdicts}, result.usage
