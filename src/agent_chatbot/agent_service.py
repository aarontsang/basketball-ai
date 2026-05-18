"""
Shared NBA analyst agent setup for CLI and FastAPI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from agent_tools import (
    get_player_stats_tool,
    get_team_stats_tool,
    get_train_team_xgboost_tool,
    get_train_player_xgboost_tool,
    get_predict_team_matchup_tool,
    get_predict_player_matchup_tool,
)

AGENT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = AGENT_DIR / "storage"
CORPUS_PATH = AGENT_DIR / "nba_wikipedia_corpus.json"

AGENT_SYSTEM_PROMPT = """
You are a Senior NBA Data Analyst.

CRITICAL — use these EXACT tool names in the Action line (copy exactly):
- nba_history_tool
- player_nba_stats_tool
- team_nba_stats_tool
- train_team_xgboost_model
- train_player_xgboost_model
- predict_team_matchup
- predict_player_matchup

Never use made-up names like get_player_stats, get_team_stats, rag_nba_wiki, or train_xgboost_model.

ML training (ReAct format):
- Team / team-vs-team model → Action: train_team_xgboost_model / Action Input: {}
- Player / player-vs-player model → Action: train_player_xgboost_model / Action Input: {}

After a training tool returns results, give the user a clear summary. Do not call unrelated tools.

Stats & history:
- Vague "who is the best" questions → nba_history_tool first, then player_nba_stats_tool per player.
- Specific player stats → player_nba_stats_tool with JSON args, e.g. {{"player_name": "LeBron James", "scope": "season"}}.
- Team info → team_nba_stats_tool with {{"team_name": "Los Angeles Lakers"}}.

Predictions (only after the matching model is trained):
- predict_team_matchup → {{"home_team": "...", "away_team": "..."}}
- predict_player_matchup → {{"player_a": "...", "player_b": "..."}}
"""

DEFAULT_MAX_ITERATIONS = 25


def _configure_settings(ollama_model: str = "llama3", temperature: float = 0.1) -> None:
    Settings.llm = Ollama(model=ollama_model, temperature=temperature)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def _load_documents(corpus_path: Path = CORPUS_PATH) -> list[Document]:
    with open(corpus_path) as f:
        content = json.load(f)
    return [
        Document(text=text, metadata={"source": f"{corpus_path.name} - {title}"})
        for title, text in content.items()
    ]


def _load_or_build_index(documents: list[Document]) -> VectorStoreIndex:
    if STORAGE_DIR.exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        return load_index_from_storage(storage_context)

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    return index


def create_agent(
    *,
    ollama_model: str | None = None,
    verbose: bool = False,
) -> ReActAgent:
    """Build and return a configured ReAct agent."""
    model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3")
    _configure_settings(ollama_model=model)

    documents = _load_documents()
    index = _load_or_build_index(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)

    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="nba_history_tool",
            description=(
                "NBA history, narratives, and general basketball knowledge from documents. "
                "Not for live stats — use player_nba_stats_tool or team_nba_stats_tool instead."
            ),
        ),
    )

    return ReActAgent(
        tools=[
            rag_tool,
            get_player_stats_tool(),
            get_team_stats_tool(),
            get_train_team_xgboost_tool(),
            get_train_player_xgboost_tool(),
            get_predict_team_matchup_tool(),
            get_predict_player_matchup_tool(),
        ],
        llm=Settings.llm,
        system_prompt=AGENT_SYSTEM_PROMPT,
        verbose=verbose,
    )


async def chat(agent: ReActAgent, message: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> str:
    """Run one user turn through the agent."""
    result = await agent.run(user_msg=message, max_iterations=max_iterations)
    return str(result)
