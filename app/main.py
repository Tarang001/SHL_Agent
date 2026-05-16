"""
SHL Assessment Recommender Agent
Production-grade FastAPI service with hybrid retrieval + LLM generation
"""
import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.retriever import HybridRetriever
from app.agent import SHLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Globals — initialized once at startup
retriever: Optional[HybridRetriever] = None
agent: Optional[SHLAgent] = None


@app.on_event("startup")
async def startup():
    global retriever, agent
    catalog_path = os.path.join(os.path.dirname(__file__), "../data/catalog.json")
    retriever = HybridRetriever(catalog_path)
    agent = SHLAgent(retriever)
    logger.info(f"Loaded {len(retriever.catalog)} catalog items")


# ── Schema ──────────────────────────────────────────────────────────────────

class RecommendationInput(BaseModel):
    name: str
    url: str
    test_type: str = ""


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    recommendations: Optional[list[RecommendationInput]] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        if v[-1].role != "user":
            raise ValueError("last message must be from user")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    messages = []
    for m in request.messages:
        msg: dict = {"role": m.role, "content": m.content}
        if m.recommendations:
            msg["recommendations"] = [
                {"name": r.name, "url": r.url, "test_type": r.test_type}
                for r in m.recommendations
            ]
        messages.append(msg)
    result = await agent.respond(messages)
    
    # Programmatic validation — schema must always be correct; types from catalog only
    grounded = retriever.validate_recommendations_against_catalog(
        result.get("recommendations", [])
    )
    validated_recs = []
    for r in grounded:
        if not r.get("name") or not r.get("url"):
            continue
        if not retriever.is_valid_url(r["url"]):
            logger.warning(f"Rejected hallucinated URL: {r['url']}")
            continue
        validated_recs.append(Recommendation(
            name=r["name"],
            url=r["url"],
            test_type=r.get("test_type", ""),
        ))
    
    # Cap at 10
    validated_recs = validated_recs[:10]
    
    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=validated_recs,
        end_of_conversation=bool(result.get("end_of_conversation", False))
    )
