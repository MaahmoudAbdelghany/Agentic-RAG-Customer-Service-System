"""Document chunking module for the RAG pipeline."""

import logging
from pathlib import Path
from typing import List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # Fallback for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split a list of LangChain documents into smaller chunks with metadata enrichment.

    Args:
        documents: List of LangChain Document objects to chunk.
        chunk_size: Optional custom chunk size (in characters). If None, uses settings.CHUNK_SIZE.
        chunk_overlap: Optional custom chunk overlap (in characters). If None, uses settings.CHUNK_OVERLAP.

    Returns:
        List[Document]: List of chunked Document objects with updated metadata.
    """
    if not documents:
        logger.warning("No documents provided to chunk.")
        return []

    settings = get_settings()
    effective_chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    effective_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    )

    logger.info(
        f"Splitting {len(documents)} documents (chunk_size={effective_chunk_size}, "
        f"chunk_overlap={effective_chunk_overlap})"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks: List[Document] = []

    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        total_chunks = len(doc_chunks)

        for idx, chunk in enumerate(doc_chunks):
            # Enrich metadata
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = total_chunks

            if "source" in chunk.metadata:
                chunk.metadata["source_file"] = Path(chunk.metadata["source"]).name

            all_chunks.append(chunk)

    logger.info(
        f"Successfully created {len(all_chunks)} chunks from {len(documents)} documents."
    )
    return all_chunks
