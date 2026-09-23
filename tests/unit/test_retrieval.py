import json

import pytest

from app.core.config import Settings
from app.models import DocumentType
from app.services.indexing import IndexService
from app.services.retrieval import HybridRetriever, fuse_scores, reciprocal_rank_fusion
from app.services.text import tokenize
from tests.fakes import FakeEmbeddings, FakeReranker

KB = [
    {"id": "cloud", "question": "Which cloud provider hosts the service?", "answer": "Google Cloud Platform (GCP)."},
    {"id": "enc", "question": "Is data encrypted at rest?", "answer": "Yes, AES-256 on all disks."},
    {"id": "mfa", "question": "Is MFA enforced?", "answer": "MFA is required for SSH access."},
    {"id": "bg", "question": "Are background checks performed?", "answer": "Yes, for all new employees."},
]


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, retrieval_candidates=4, rerank_candidates=4)


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


async def _index(settings, embeddings, rows=KB):
    service = IndexService(embeddings, settings)
    return await service.get_index(json.dumps(rows).encode(), DocumentType.JSON)


def test_tokenize_keeps_compound_identifiers_and_drops_stopwords():
    assert tokenize("Is the data encrypted with AES-256 and TLS1.2 per CC6.1?") == [
        "data", "encrypted", "aes-256", "tls1.2", "per", "cc6.1",
    ]


def test_fuse_scores_normalizes_each_list_and_weights_them():
    fused = fuse_scores({1: 0.9, 2: 0.5}, {2: 12.0, 3: 4.0}, vector_weight=0.7)
    assert fused[1] == pytest.approx(0.7)   # best vector hit, absent from BM25
    assert fused[2] == pytest.approx(0.3)   # worst vector hit, best BM25 hit
    assert fused[3] == pytest.approx(0.0)


def test_fuse_scores_handles_empty_lexical_list():
    assert fuse_scores({1: 0.4}, {}, vector_weight=0.7) == {1: pytest.approx(0.7)}


def test_rrf_rewards_items_ranked_well_by_both():
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])[0] == 2


async def test_retrieves_the_matching_row_first(settings, embeddings):
    index, cached = await _index(settings, embeddings)
    assert cached is False
    retriever = HybridRetriever(FakeReranker(), settings)
    query = "Which cloud provider do you use?"
    hits = retriever.retrieve(index, query, embeddings.encode_queries([query])[0], k=2)
    assert hits[0].chunk.source_id == "cloud"
    assert len(hits) == 2
    assert 0.0 <= hits[0].relevance <= 1.0


async def test_lexical_signal_finds_exact_identifier(settings, embeddings):
    index, _ = await _index(settings, embeddings)
    query = "AES-256"
    hits = HybridRetriever(FakeReranker(), settings).retrieve(index, query, embeddings.encode_queries([query])[0], k=1)
    assert hits[0].chunk.source_id == "enc"


async def test_index_is_cached_by_content_hash(settings, embeddings):
    service = IndexService(embeddings, settings)
    document = json.dumps(KB).encode()
    first, first_cached = await service.get_index(document, DocumentType.JSON)
    second, second_cached = await service.get_index(document, DocumentType.JSON)
    assert (first_cached, second_cached) == (False, True)
    assert first is second
    assert embeddings.calls == 1
