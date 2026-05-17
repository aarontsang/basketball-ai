"""
This file defines the main chatbot agent that will interact with users. It sets up the language model
and the tools that the agent can use to answer queries. The agent follows a ReAct reasoning 
chain to determine which tools to use based on the user's question.The system prompt 
defines the agent's role, the tools it has access to, and the operational protocol it should 
follow when answering questions. The agent is designed to provide evidence-based comparisons 
and statistics about NBA players and teams.
"""

import json
import asyncio
import os
import traceback
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from agent_tools import (
    get_player_stats_tool,
    get_team_stats_tool,
    get_train_team_xgboost_tool,
    get_train_player_xgboost_tool,
    get_predict_team_matchup_tool,
    get_predict_player_matchup_tool,
)
from llama_index.llms.groq import Groq
from llama_index.core import StorageContext, load_index_from_storage


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

Settings.llm = Ollama(model="llama3", temperature=0.1)


Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



def create_document_from_json(file_path: str) -> Document:
    with open(file_path, "r") as f:
        content = json.load(f)

    documents = []
    for title, text in content.items():
        documents.append(Document(
            text=text,
            metadata={"source": f"{file_path} - {title}"}
        ))
    
    return documents


def create_index_from_documents(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index


async def run_chatbot(agent):

    print("NBA Analyst Agent is ready! (Type 'exit' to quit)")

    while True:
        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # 3. Use 'await agent.run()' and the parameter 'input'

            response = await agent.run(user_msg=query, max_iterations=25)

            print(f"\nResponse: {response}")

        except Exception as e:
            print(f"An error occurred: {e!r}")
            traceback.print_exc()


if __name__ == "__main__":
    documents = create_document_from_json("nba_wikipedia_corpus.json")
    document_index = create_index_from_documents(documents)
    query_engine = document_index.as_query_engine(similarity_top_k=2)

    if not os.path.exists("./storage"):
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="./storage")
    else:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)


    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="nba_history_tool",
            description="Use this for general knowledge, history, and narrative info." \
            "Use this provided documents to answer questions about NBA history, players, teams, and general basketball knowledge. " \
            "This tool is best for questions that require contextual understanding or historical information, rather than specific statistics." \
            "Answer queries based on the information in the provided documents. If the answer is not contained in the documents, say you don't know rather than trying to guess." \
            "If the question is about specific player or team statistics, use the appropriate stats tool instead of this one." \
            "Answer the user's queries in a sports analyst tone, providing clear and concise information based on the documents. " \
            "Respond to the user as you were a knowledgeable NBA analyst, using the information from the documents to provide accurate and insightful answers."
        )
    )

    player_stats_tool = get_player_stats_tool()
    team_stats_tool = get_team_stats_tool()
    train_team_tool = get_train_team_xgboost_tool()
    train_player_tool = get_train_player_xgboost_tool()
    predict_team_tool = get_predict_team_matchup_tool()
    predict_player_tool = get_predict_player_matchup_tool()

    agent = ReActAgent(
        tools=[
            rag_tool,
            player_stats_tool,
            team_stats_tool,
            train_team_tool,
            train_player_tool,
            predict_team_tool,
            predict_player_tool,
        ],
        llm=Settings.llm,
        system_prompt=AGENT_SYSTEM_PROMPT,
        verbose=True,
    )

    asyncio.run(run_chatbot(agent))
