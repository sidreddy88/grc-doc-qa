import threading

import numpy as np
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """LangChain Embeddings backed by a local sentence-transformers model.

    Local on purpose: the OpenAI key is reserved for gpt-4o-mini generation, and a
    small CPU model is fast enough for single-document corpora (a few hundred chunks).
    Vectors are L2-normalized so inner product equals cosine similarity.
    """

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = None
        self._load_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    from app.services.model_utils import realign_parameters

                    model = SentenceTransformer(self.model_name, device="cpu")
                    realign_parameters(model)
                    self._model = model
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._get_model().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.encode([f"{self.query_prefix}{query}" for query in queries])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.encode_queries([text])[0].tolist()
