"""
FastAPI backend for the NBA analyst chatbot.

Run from repo root:
  uvicorn api.main:app --reload --app-dir src
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure `src` is on the path so `api` and `agent_chatbot` resolve.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

AGENT_CHATBOT_DIR = SRC_DIR / "agent_chatbot"
if str(AGENT_CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_CHATBOT_DIR))

from api.deps import app_state  # noqa: E402
from api.routers import chat, health, ml  # noqa: E402
from agent_service import create_agent  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the agent once at startup (embeddings + vector index)."""
    app_state.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    verbose = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
    app_state.agent = create_agent(ollama_model=app_state.ollama_model, verbose=verbose)
    yield
    app_state.agent = None


app = FastAPI(
    title="NBA Analyst API",
    description="Chat with the NBA analyst agent, train XGBoost models, and run matchup predictions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(ml.router, prefix="/api/v1")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def chat_ui():
    """Serve the chatbot UI."""
    if not INDEX_HTML.is_file():
        return {
            "error": "Frontend not found",
            "expected": str(INDEX_HTML),
            "hint": "Ensure src/api/static/index.html exists and restart uvicorn.",
        }
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/api")
async def api_info():
    return {
        "service": "NBA Analyst API",
        "ui": "/",
        "docs": "/docs",
        "health": "/api/v1/health",
        "chat": "/api/v1/chat",
    }
