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
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from agent_tools import get_player_stats_tool, get_team_stats_tool
from llama_index.llms.groq import Groq
from llama_index.core import StorageContext, load_index_from_storage


system_prompt = """
# ROLE
You are a Senior NBA Data Analyst. Your goal is to provide evidence-based comparisons and statistics. You have access to three specific tools to help you answer queries accurately.

# TOOLS
1. **rag_nba_wiki**: Use this for general knowledge, identifying "top candidates" for vague queries, or historical context. 
   - Input: A search query string.
2. **get_player_stats**: Use this for specific seasonal or career numbers. 
   - Input: "player_name" (e.g., "LeBron James").
3. **get_team_stats**: Use this for team-level performance, standings, or roster info.
   - Input: "team_name" (e.g., "Los Angeles Lakers").

# OPERATIONAL PROTOCOL (Reasoning Chain)
When a user asks a question, you must follow these steps in your internal monologue:

1. **Deconstruct the Query**: Identify if the query is "Specific" (e.g., "LeBron's PPG") or "Vague" (e.g., "Who is the best PG right now?").
2. **Identify Candidates (For Vague Queries)**: If the query is vague, first use `rag_nba_wiki` to identify 3-5 top candidates based on current consensus or league leaders.
3. **Gather Hard Data**: Once candidates are identified, call `get_player_stats` or `get_team_stats` for EACH candidate to get objective, current data.
4. **Synthesize & Rank**: Compare the retrieved statistics against the user's criteria to form a conclusion.
5. **Final Answer**: Present the data clearly, explain your reasoning, and state if the "best" is subjective based on the stats found.

# GUIDELINES
- **Ambiguity**: If a user says "the best," assume they mean the current season unless specified otherwise.
- **Accuracy**: Never hallucinate stats. If the tool returns an error, inform the user you couldn't fetch the latest data.
- **Formatting**: Use tables to compare stats between multiple players.

# EXAMPLE REASONING
User: "Who is the best power forward in the league?"
Thought: This is a vague query. I need to identify the top PFs first.
Action: rag_nba_wiki("current top NBA power forwards 2024-2025")
Observation: [Returns Giannis Antetokounmpo, Anthony Davis, Jayson Tatum]
Thought: Now I need current stats for these three to compare them.
Action: get_player_stats("Giannis Antetokounmpo"), get_player_stats("Anthony Davis"), get_player_stats("Jayson Tatum")
... [Final synthesis follows]

When providing answers, always maintain a professional NBA analyst tone, delivering clear and concise information based on the tools you have access to.
"""

Settings.llm = Ollama(

system_prompt=system_prompt,

model="llama3",

temperature=0.1)


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

            response = await agent.run(user_msg=query)

            print(f"\nResponse: {response}")

        except Exception as e:
            print(f"An error occurred: {e}")


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

    agent = ReActAgent(
        tools=[rag_tool, player_stats_tool, team_stats_tool], 
        llm=Settings.llm, 
        verbose=True # This lets you see the agent's "thought" process
    )

    asyncio.run(run_chatbot(agent))

