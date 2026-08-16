# Agentic RAG Customer Service System — Tasks

## Phase 1: Project Scaffolding & Configuration
- [x] Create `pyproject.toml` with all dependencies (uv)
- [x] Create `.env.example`
- [x] Create `.gitignore`
- [x] Create `README.md`

## Phase 2: Configuration Module
- [x] Create `src/config/__init__.py`
- [x] Create `src/config/settings.py` (Pydantic Settings)

## Phase 3: Demo Knowledge Base (Synthetic Data)
- [x] Create `data/knowledge_base/return_policy.md`
- [x] Create `data/knowledge_base/shipping_policy.md`
- [x] Create `data/knowledge_base/warranty_policy.md`
- [x] Create `data/knowledge_base/faq.md`
- [x] Create `data/knowledge_base/product_catalog.md`
- [x] Create `data/mock_orders.json`

## Phase 4: RAG Pipeline
- [x] Create `src/rag/__init__.py`
- [x] Create `src/rag/document_loader.py`
- [x] Create `src/rag/chunker.py`
- [ ] Create `src/rag/embeddings.py`
- [ ] Create `src/rag/vector_store.py`
- [ ] Create `src/rag/ingest.py`

## Phase 5: Agent Tools
- [ ] Create `src/tools/__init__.py`
- [ ] Create `src/tools/knowledge_retrieval.py`
- [ ] Create `src/tools/order_tracking.py`
- [ ] Create `src/tools/human_handoff.py`

## Phase 6: LangGraph Agent
- [ ] Create `src/agent/__init__.py`
- [ ] Create `src/agent/prompts.py`
- [ ] Create `src/agent/state.py`
- [ ] Create `src/agent/graph.py`

## Phase 7: Guardrails Module
- [ ] Create `src/guardrails/__init__.py`
- [ ] Create `src/guardrails/validators.py`

## Phase 8: FastAPI Backend
- [ ] Create `src/api/__init__.py`
- [ ] Create `src/api/models.py`
- [ ] Create `src/api/main.py`

## Phase 9: Analytics & Logging
- [ ] Create `src/analytics/__init__.py`
- [ ] Create `src/analytics/logger.py`
- [ ] Create `src/analytics/metrics.py`

## Phase 10: Streamlit UI
- [ ] Create `src/ui/Home.py`
- [ ] Create `src/ui/pages/Analytics.py`
- [ ] Create `src/ui/pages/Knowledge_Base.py`

## Phase 11: Entry Points & Scripts
- [ ] Create `src/__init__.py`
- [ ] Create `scripts/run_api.py`
- [ ] Create `scripts/run_ui.py`
- [ ] Create `scripts/ingest_data.py`

## Verification
- [ ] Install dependencies with uv
- [ ] Run ingestion pipeline
- [ ] Test the chat API
- [ ] Test the Streamlit UI
