import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from rank_bm25 import BM25Okapi

from app.core.config import Settings
from app.models import Chunk, DocumentType
from app.services.cache import AsyncTTLCache
from app.services.chunking import chunk_document
from app.services.document_loader import load_document
from app.services.embeddings import LocalEmbeddings
from app.services.text import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentIndex:
    doc_hash: str
    doc_type: DocumentType
    chunks: list[Chunk]
    vector_store: FAISS
    bm25: BM25Okapi


def document_hash(document: bytes) -> str:
    return hashlib.sha256(document).hexdigest()


def build_index(doc_hash: str, doc_type: DocumentType, chunks: list[Chunk], embeddings: LocalEmbeddings) -> DocumentIndex:
    """Embed chunks into FAISS and build a BM25 index over the same texts. CPU-bound."""
    texts = [chunk.index_text for chunk in chunks]
    vectors = embeddings.encode(texts)
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors.tolist(), strict=True)),
        embedding=embeddings,
        metadatas=[{"chunk_id": chunk.chunk_id} for chunk in chunks],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    # BM25Okapi divides by average document length, so give token-less chunks a placeholder.
    bm25 = BM25Okapi([tokenize(text) or ["_"] for text in texts])
    return DocumentIndex(doc_hash=doc_hash, doc_type=doc_type, chunks=chunks, vector_store=vector_store, bm25=bm25)


class IndexService:
    """Parses, chunks, and indexes a document once per content hash."""

    def __init__(self, embeddings: LocalEmbeddings, settings: Settings) -> None:
        self._embeddings = embeddings
        self._settings = settings
        self._cache: AsyncTTLCache[DocumentIndex] = AsyncTTLCache(
            max_entries=settings.index_cache_entries, ttl_seconds=settings.index_cache_ttl_s
        )

    async def get_index(self, document: bytes, doc_type: DocumentType) -> tuple[DocumentIndex, bool]:
        """Return (index, cache_hit)."""
        doc_hash = document_hash(document)

        async def build() -> DocumentIndex:
            started = time.perf_counter()
            loaded = await asyncio.to_thread(load_document, document, doc_type, self._settings)
            chunks = chunk_document(loaded, self._settings)
            index = await asyncio.to_thread(build_index, doc_hash, doc_type, chunks, self._embeddings)
            logger.info(
                "index_built",
                extra={
                    "doc_hash": doc_hash[:12],
                    "doc_type": doc_type.value,
                    "chunks": len(chunks),
                    "build_ms": round((time.perf_counter() - started) * 1000),
                },
            )
            return index

        return await self._cache.get_or_create(doc_hash, build)
