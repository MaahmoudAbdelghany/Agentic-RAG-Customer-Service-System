# Agentic RAG Customer Service System

This project is an advanced Agentic Retrieval-Augmented Generation (RAG) Customer Service System. It leverages LangGraph and LangChain to create an autonomous agent capable of resolving customer queries by intelligently querying a knowledge base, retrieving simulated order information, and routing to human agents when necessary.

## Features

- **Agentic RAG**: A dynamic RAG pipeline powered by a LangGraph state machine.
- **Tools**: Includes specialized tools for knowledge retrieval, order tracking, and human handoff.
- **FastAPI Backend**: A robust REST API for integrating with external frontends.
- **Streamlit UI**: An interactive web interface for interacting with the agent and exploring analytics.
- **Guardrails AI**: Validates both the incoming prompts and the agent's responses to ensure safety and compliance.
- **ChromaDB**: Local vector store for document embeddings.

## Installation

This project uses `uv` for dependency management. To set up the environment, run:

```bash
uv sync
```

Alternatively, standard pip installation is supported via the `pyproject.toml` file.

## Configuration

1. Copy `.env.example` to `.env`.
2. Provide your API keys (e.g., `OPENAI_API_KEY`) and customize settings as needed.

## Usage

*Instructions for running the API and Streamlit UI will be added as those phases are implemented.*

## Architecture

1. **Phase 1-2**: Configuration and scaffolding.
2. **Phase 3-4**: Data ingestion and the core Vector/RAG pipeline.
3. **Phase 5-7**: LangGraph Agent construction, tooling, and guardrails.
4. **Phase 8-10**: Web endpoints (FastAPI) and UI (Streamlit) along with analytics tracking.
