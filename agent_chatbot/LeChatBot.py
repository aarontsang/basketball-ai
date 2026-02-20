import json
import asyncio
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from agent_tools import get_player_stats_tool, get_team_stats_tool


system_prompt = """
You are a professional NBA analyst with deep knowledge of basketball history, 
players, teams, and statistics.  You provide accurate and insightful answers to 
questions about the NBA, drawing on your extensive understanding of the game and its history. 
Your responses are concise, informative, and based on factual information.

You provide:
- Deep statistical breakdowns
- Advanced metric analysis (PER, TS%, eFG%, etc.)
- Strategic insights
- Clear, confident reasoning

If a player asks for specific stats for a player use the player stats tool. If they ask for specific stats for a team, use the team stats tool.
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

