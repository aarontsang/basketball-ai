from llama_index.core import Document
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
import json
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



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


def run_chatbot(query_engine):
    
    while True:
        query = input("Ask a question about the NBA (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        response = query_engine.query(query)
        print(f"Response: {response}\n")


def intent_router(user_query):
    prediction_keywords = ["predict", "forecast", "projection", "outcome"]
    stats_keywords = ["stats", "statistics", "data", "numbers"]
    if any(keyword in user_query.lower() for keyword in prediction_keywords):
        return "predict"
    elif any(keyword in user_query.lower() for keyword in stats_keywords):
        return "stats"
    else:
        return "general"


if __name__ == "__main__":
    documents = create_document_from_json("nba_wikipedia_corpus.json")
    document_index = create_index_from_documents(documents)
    query_engine = document_index.as_query_engine(similarity_top_k=2)
    run_chatbot(query_engine)