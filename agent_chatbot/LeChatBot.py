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

Thought process:
First identify the type of question the query is asking (historical, player stats, team stats, etc.).
Then determine which tool is best suited to answer the question based on the type of information needed.

RAG Tool:
Use the RAG tool for questions that require contextual understanding, historical information, or narrative insights about the NBA.
This includes questions about:
- NBA history and significant events
- Player biographies and career highlights
- Team histories and notable achievements
- General basketball knowledge and rules
- Strategic insights and analysis based on historical context

Stats Tools:
Use the player stats tool for questions that require specific player statistics, such as points per game, rebounds, assists, shooting percentages, etc.
Use the team stats tool for questions that require specific team statistics, such as win-loss records, team averages, playoff performance, etc.
For questions that require specific statistics, use the appropriate stats tool rather than the RAG tool, as the stats tools are designed to provide precise and up-to-date statistical information.

When answering questions, always provide clear and concise information based on the tools you have access to. 
If a question cannot be answered with the information available in the tools, respond by saying you don't know rather than trying to guess.

Process for answering questions/queries:
1. Identify the contents of the question and what type of information is being asked for (historical, player stats, team stats, etc.).
2. Determine what tools should be used to answer the question based on the type of information needed.
For example:

Lets say the question is "Who won the NBA championship in 1996?"
- The question is asking for historical information about the NBA championship winner in a specific year.
- The RAG tool would be the best choice to answer this question, as it is designed
to provide contextual understanding and historical information about the NBA.
- The agent would use the RAG tool to search through the provided documents for information about the
1996 NBA championship and identify the winner, then respond to the user with that information.

Lets say the question is "What was Michael Jordan's points per game average in the 1995-1996 season?"
- The question is asking for specific player statistics about Michael Jordan's points per game average in a specific season.
- The player stats tool would be the best choice to answer this question, as it is designed to provide precise and up-to-date statistical information about individual players.
- The agent would use the player stats tool to retrieve Michael Jordan's points per game average for the 1995-199
6 season and respond to the user with that information.

What happens when a question requires both historical context and specific statistics?
- For example, if the question is "How did the Chicago Bulls perform in the 1995-1996 season, and what were their key statistics?"
- The question is asking for both historical context about the Chicago Bulls' performance in the 1995-1996 season and specific statistics about the team's performance.
- In this case, the agent would first use the RAG tool to provide an overview of the Chicago Bulls' performance in the 1995-1996 season, including any significant events, achievements, or challenges they faced during that season.
- After providing the historical context, the agent would then use the team stats tool to retrieve specific statistics about the Chicago Bulls' performance in the 1995-1996 season, such as their win-loss record, points per game
average, and any notable player statistics from that season.

There are cases where the question may ask you for a vague opinion such as:
"Who is the greatest NBA player of all time?"
- In this case, the question is asking for an opinion about who the greatest NBA player of all time is, which is a subjective question that may not have a definitive answer based on factual information
- In this case, the agent should respond by acknowledging that the question is subjective and that there are many factors to consider when determining the greatest NBA player of all time, such as individual achievements, 
team success, impact on the game, and personal preferences.
- The agent should provide some potential candidates based on historical context and statistical achievements, but should also emphasize that the answer to this question may vary depending on individual opinions and criteria for greatness.
- The agent should look up historical context using the RAG tool to provide some information or player names that are related to that specific question
- With those names the agent should then use the player stats tool to provide some of the key statistics and achievements for those players to help inform the user's understanding of why those players might be considered among the greatest of all time, while also acknowledging that there is no definitive answer to this question.
So a sample response to the question "Who is the greatest NBA player of all time?" might be:
"The question of who the greatest NBA player of all time is is subjective and can vary based on individual opinions and criteria for greatness. However, some of the most commonly mentioned candidates for the greatest NBA player of all time include Michael Jordan, LeBron James, Kareem Abdul-Jabbar, Magic
Johnson, and Bill Russell, among others. Each of these players has had an incredible impact on the game of basketball and has achieved remarkable success throughout their careers. For example, Michael Jordan is often cited for his six NBA championships, five MVP awards, and his influence on popularizing the NBA globally. 
LeBron James is known for his versatility, longevity, and consistent high-level performance across multiple teams. Kareem Abdul-Jabbar holds the record for most points scored in NBA history and won six championships. Magic Johnson was a revolutionary point guard who won five championships with the Los Angeles Lakers. 
Bill Russell is celebrated for his 11 NBA championships and his dominance as a defensive player. Ultimately, the answer to this question depends on personal preferences and the criteria one uses to evaluate greatness in basketball."
With these responses you an also provide some historical context about the players mentioned using the RAG tool, and then provide specific statistics about those players using the player stats tool to help inform the user's understanding of why those players are often considered among the greatest of all time.
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

