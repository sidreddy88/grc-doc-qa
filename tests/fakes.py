"""Deterministic stand-ins for the ML models so tests never download weights or call APIs."""

import hashlib
import re

import numpy as np

from app.services.embeddings import LocalEmbeddings
from app.services.text import tokenize

_DIM = 256


class FakeEmbeddings(LocalEmbeddings):
    """Hashed bag-of-words vectors: texts sharing tokens get high cosine similarity."""

    def __init__(self) -> None:
        super().__init__(model_name="fake", query_prefix="")
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), _DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


class ScriptedLLM:
    """StructuredLLM double. Responses are queued per schema class name.

    A queued item may be a dict (validated into the schema), an exception (raised),
    or a callable taking (system, user) and returning either of those.
    """

    def __init__(self, script: dict[str, list] | None = None, usage: tuple[int, int] = (100, 20)) -> None:
        self.script = {name: list(items) for name, items in (script or {}).items()}
        self.calls: list[tuple[str, str]] = []
        self._usage = usage

    async def generate(self, system: str, user: str, schema):
        from app.services.llm import LLMResult, TokenUsage

        name = schema.__name__
        self.calls.append((name, user))
        queue = self.script.get(name)
        if not queue:
            raise AssertionError(f"Unexpected LLM call for {name}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(item) and not isinstance(item, type):
            item = item(system, user)
        if isinstance(item, BaseException):
            raise item
        return LLMResult(
            value=schema.model_validate(item),
            usage=TokenUsage(input_tokens=self._usage[0], output_tokens=self._usage[1], calls=1),
        )

    def calls_for(self, name: str) -> list[str]:
        return [user for schema, user in self.calls if schema == name]


_SOURCE_BLOCK = re.compile(r"^\[(S\d+)\] \([^)]*\)\n(.+?)(?=\n\n\[S\d+\] |\n</sources>)", re.M | re.S)


def parse_sources(user_prompt: str) -> dict[str, str]:
    return {label: text for label, text in _SOURCE_BLOCK.findall(user_prompt)}


def grounded_synthesis(answer: str = "Grounded answer.", source: str = "S1", quote_chars: int = 60):
    """Synthesis response that quotes the start of a real source from the prompt."""

    def respond(system: str, user: str) -> dict:
        text = parse_sources(user)[source]
        return {"supported": True, "answer": answer, "citations": [{"source": source, "quote": text[:quote_chars]}], "items": []}

    return respond


def approve_all(system: str, user: str) -> dict:
    return {"verdicts": [{"claim_id": cid, "faithful": True, "reason": "ok"} for cid in re.findall(r"^claim_id: (\S+)", user, re.M)]}


def reject_all(system: str, user: str) -> dict:
    return {"verdicts": [{"claim_id": cid, "faithful": False, "reason": "unsupported"} for cid in re.findall(r"^claim_id: (\S+)", user, re.M)]}


class FakeReranker:
    """Relevance = fraction of query tokens present in the passage."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return [0.0] * len(passages)
        return [len(query_tokens & set(tokenize(passage))) / len(query_tokens) for passage in passages]
