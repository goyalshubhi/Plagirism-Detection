import os
import hashlib
import threading

try:
    from sentence_transformers import SentenceTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False


class EmbeddingService:
    """
    Lazy-loading embedding service:
    - DOES NOT load model at import time
    - DOES NOT load model inside __init__
    - Loads model only when an embedding method is called
    """

    _model = None
    _model_lock = threading.Lock()

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # don't load here

    def _get_model(self):
        # Disable model in Vercel / serverless env if you want
        if not HAS_MODEL or os.getenv("VERCEL"):
            return None

        # already loaded
        if EmbeddingService._model is not None:
            return EmbeddingService._model

        # load once (thread-safe)
        with EmbeddingService._model_lock:
            if EmbeddingService._model is None:
                EmbeddingService._model = SentenceTransformer(self.model_name)

        return EmbeddingService._model

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        if not text:
            return []
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(text), step):
            chunks.append(text[i:i + chunk_size])
            if i + chunk_size >= len(text):
                break
        return chunks

    def encode_chunks(self, text):
        """Generate embeddings for each chunk of text"""
        model = self._get_model()
        if not model:
            return [], []

        chunks = self.chunk_text(text)
        if not chunks:
            return [], []

        # faster: encode list in one call
        embeddings = model.encode(chunks)
        return chunks, embeddings

    def generate_text_embedding(self, text):
        """Return one embedding for the whole text (mean of chunk embeddings)"""
        model = self._get_model()
        if not model:
            return []

        chunks = self.chunk_text(text)
        if not chunks:
            return []

        embeddings = model.encode(chunks)

        # import numpy only when needed
        import numpy as np
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding.tolist()

    @staticmethod
    def hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
