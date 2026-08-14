"""Document loader for the knowledge base."""

import logging
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def load_documents() -> List[Document]:
    """Load all markdown documents from the knowledge base directory.

    Returns:
        List[Document]: A list of loaded LangChain documents.
    """
    settings = get_settings()
    kb_dir = settings.KNOWLEDGE_BASE_DIR

    logger.info(f"Loading documents from {kb_dir}")

    loader = DirectoryLoader(
        kb_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    try:
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []
