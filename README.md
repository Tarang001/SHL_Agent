# SHL Assessment Recommender Agent

## Architecture

```
User Query (POST /chat with full history)
        │
        ▼
Conversation State Extraction (app/state.py)
  - Seniority, language, industry, skills
  - Intent: recommend | compare | refine | clarify
  - Refusal check (legal, injection, out-of-scope) ← deterministic
        │
        ▼
Hybrid Retrieval (app/retriever.py)
  - BM25 keyword retrieval (rank-bm25)
  - Metadata filtering (job_level, language)
  - Heuristic boosting (executive→OPQ, safety→DSI, graduate→G+/SJT)
  → Top 25 candidates
        │
        ▼
LLM Generation (app/agent.py)
  - Grok (xAI) with retrieved candidates as grounding context
  - System prompt enforces schema, scope, and refusal behavior
  - Candidates injected directly = no hallucination surface
        │
        ▼
Programmatic Validation (app/main.py)
  - Schema check (reply/recommendations/end_of_conversation)
  - URL validation against catalog
  - Duplicate removal
  - Cap at 10
        │
        ▼
JSON Response
```

## Key Design Decisions

**BM25 over embeddings**: Avoids cold-start delay (30s timeout constraint). BM25 excels at exact technology names (Java, Docker, HIPAA) which are the primary query signals.

**Heuristic boosting**: Derived from gold conversation traces — e.g., executive → OPQ32r, safety-critical → DSI, graduate → G+/SJT. Hard-coded boosts ensure recall on the most common patterns.

**Stateless reconstruction**: Full conversation history processed every request. State extracted deterministically before any LLM call.

**Refusal before LLM**: Legal/injection/out-of-scope checks are regex-based and run before the LLM — no token waste, no risk of the model complying.

**Candidates-as-context**: Rather than giving the LLM the full 377-item catalog, we inject the top 25 retrieved candidates. This tightens the generation space and prevents hallucination.

## Quick Start

```bash
export XAI_API_KEY=your-key-here
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

## Evaluation

```bash
export XAI_API_KEY=your-key-here
python eval/harness.py
```

## Deploy to Render

1. Push to GitHub
2. New Web Service → connect repo
3. Set `XAI_API_KEY` env var
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## API

### GET /health
```json
{"status": "ok"}
```

### POST /chat
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a mid-level Java developer"}
  ]
}
```
Response:
```json
{
  "reply": "...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```
