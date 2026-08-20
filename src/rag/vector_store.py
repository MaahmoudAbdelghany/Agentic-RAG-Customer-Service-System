"""Vector store manager using ChromaDB for document storage and similarity search.

Provides functions to initialize, ingest document chunks, perform similarity
searches, and reset the persistent ChromaDB collection for the RAG pipeline.
"""

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.settings import get_settings
from src.rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Module-level cached instance of Chroma vector store
_vector_store: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Return a cached singleton instance of the Chroma vector store.

    Initializes a persistent ChromaDB instance on first call using the
    embedding model provided by ``get_embedding_model()``. The vector store
    persists vectors to the directory specified by ``settings.CHROMA_PERSIST_DIR``.

    Returns:
        Chroma: A ready-to-use LangChain Chroma vector store instance.

    Raises:
        RuntimeError: If initialization fails.
    """
    global _vector_store

    if _vector_store is not None:
        logger.debug("Returning cached Chroma vector store instance.")
        return _vector_store

    settings = get_settings()
    persist_dir = settings.CHROMA_PERSIST_DIR
    collection_name = settings.CHROMA_COLLECTION_NAME

    logger.info(
        f"Initializing ChromaDB vector store: collection='{collection_name}', "
        f"persist_dir='{persist_dir}'"
    )

    try:
        os.makedirs(persist_dir, exist_ok=True)
        embedding_model = get_embedding_model()

        _vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )
        logger.info("ChromaDB vector store initialized successfully.")
        return _vector_store
    except Exception as e:
        logger.error(f"Failed to initialize Chroma vector store: {e}")
        raise RuntimeError(
            f"Could not initialize Chroma vector store at '{persist_dir}'. "
            f"Original error: {e}"
        ) from e


def add_documents(documents: List[Document]) -> List[str]:
    """Embed and insert document chunks into the Chroma vector store.

    Args:
        documents (List[Document]): List of LangChain Document objects with
            text content and metadata (e.g. source, language, chunk_index).

    Returns:
        List[str]: List of unique document IDs generated and stored in Chroma.

    Raises:
        ValueError: If the documents list is empty.
        RuntimeError: If document ingestion fails.
    """
    if not documents:
        raise ValueError("Cannot add empty document list to vector store.")

    logger.info(f"Adding {len(documents)} document chunk(s) to vector store...")

    try:
        store = get_vector_store()
        ids = store.add_documents(documents)
        logger.info(f"Successfully added {len(ids)} document chunk(s) to Chroma.")
        return ids
    except Exception as e:
        logger.error(f"Error adding documents to Chroma: {e}")
        raise RuntimeError(
            f"Failed to add documents to vector store. Original error: {e}"
        ) from e


def similarity_search(
    query: str,
    k: Optional[int] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Retrieve the top-k most relevant document chunks for a given query.

    Args:
        query (str): The search query text (customer question).
        k (Optional[int]): Number of top chunks to return. Defaults to
            ``settings.TOP_K_RESULTS`` (default: 5).
        filter_dict (Optional[Dict[str, Any]]): Optional metadata filter
            dictionary for Chroma (e.g. ``{"language": "ar"}``).

    Returns:
        List[Document]: List of relevant LangChain Document chunks.
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to similarity_search.")
        return []

    settings = get_settings()
    top_k = k if k is not None else settings.TOP_K_RESULTS

    logger.info(f"Performing similarity search for query: '{query[:60]}...' (k={top_k})")

    store = get_vector_store()
    results = store.similarity_search(
        query=query,
        k=top_k,
        filter=filter_dict,
    )
    logger.debug(f"Retrieved {len(results)} chunks for query.")
    return results


def similarity_search_with_score(
    query: str,
    k: Optional[int] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Document, float]]:
    """Retrieve the top-k most relevant chunks along with their similarity scores.

    Args:
        query (str): The search query text.
        k (Optional[int]): Number of top chunks to return. Defaults to
            ``settings.TOP_K_RESULTS``.
        filter_dict (Optional[Dict[str, Any]]): Optional metadata filter dict.

    Returns:
        List[Tuple[Document, float]]: List of (Document, score) tuples.
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to similarity_search_with_score.")
        return []

    settings = get_settings()
    top_k = k if k is not None else settings.TOP_K_RESULTS

    logger.info(f"Searching with scores for query: '{query[:60]}...' (k={top_k})")

    store = get_vector_store()
    results = store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=filter_dict,
    )
    logger.debug(f"Retrieved {len(results)} scored chunks for query.")
    return results


def reset_collection() -> None:
    """Clear and delete the existing Chroma collection on disk.

    Resets the module-level vector store cache singleton to ``None``.
    Useful when re-ingesting documents or refreshing the knowledge base.
    """
    global _vector_store

    settings = get_settings()
    persist_dir = settings.CHROMA_PERSIST_DIR
    collection_name = settings.CHROMA_COLLECTION_NAME

    logger.warning(
        f"Resetting ChromaDB collection '{collection_name}' in '{persist_dir}'..."
    )

    try:
        if _vector_store is not None:
            try:
                _vector_store.delete_collection()
            except Exception as e:
                logger.debug(f"delete_collection warning (can be ignored if empty): {e}")
            _vector_store = None

        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
            logger.info(f"Removed Chroma persistence directory: {persist_dir}")

        logger.info("ChromaDB collection reset completed successfully.")
    except Exception as e:
        logger.error(f"Failed to reset ChromaDB collection: {e}")
        raise RuntimeError(
            f"Could not reset Chroma vector store collection. Original error: {e}"
        ) from e
