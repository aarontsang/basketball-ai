"""
FastAPI dependencies and application state.
"""

from __future__ import annotations

import sys
from pathlib import Path

from llama_index.core.agent.workflow import ReActAgent

# Allow imports from agent_chatbot when the API is started from repo root.
AGENT_CHATBOT_DIR = Path(__file__).resolve().parents[1] / "agent_chatbot"
if str(AGENT_CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_CHATBOT_DIR))

from agent_service import create_agent  # noqa: E402


class AppState:
    agent: ReActAgent | None = None
    ollama_model: str = "llama3"


app_state = AppState()


def get_agent() -> ReActAgent:
    if app_state.agent is None:
        raise RuntimeError("Agent is not initialized yet")
    return app_state.agent
