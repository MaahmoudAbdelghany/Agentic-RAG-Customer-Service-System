"""Embedding model loader for the RAG pipeline.

Provides a cached singleton instance of the HuggingFace embedding model
used throughout the application for document ingestion and query embedding.
"""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level cache for the embedding model singleton
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached singleton instance of the HuggingFace embedding model.

    The model is loaded once on first call and reused for the entire
    application lifetime. This avoids the overhead of re-downloading and
    re-loading the model weights (~100 MB) on every embedding request.

    The model name is read from ``settings.EMBEDDING_MODEL`` (default:
    ``paraphrase-multilingual-MiniLM-L12-v2``). This particular model
    produces 384-dimensional vectors and supports 50+ languages, making
    it ideal for the bilingual (English / Arabic) knowledge base.

    Returns:
        HuggingFaceEmbeddings: A ready-to-use LangChain embeddings object.

    Raises:
        RuntimeError: If the model fails to load (e.g. network issue on
            first download, or invalid model name).
    """
    global _embedding_model

    if _embedding_model is not None:
        logger.debug("Returning cached embedding model.")
        return _embedding_model

    settings = get_settings()
    model_name = settings.EMBEDDING_MODEL

    logger.info(f"Loading embedding model: {model_name}")

    try:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embedding model '{model_name}' loaded successfully.")
        return _embedding_model
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        raise RuntimeError(
            f"Could not load embedding model '{model_name}'. "
            f"Ensure 'sentence-transformers' and 'langchain-huggingface' are installed and the model name is valid. "
            f"Original error: {e}"
        ) from e
