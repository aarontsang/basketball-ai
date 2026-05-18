"""
CLI entrypoint for the NBA analyst chatbot.
For the HTTP API, run: uvicorn api.main:app --reload --app-dir src
"""

import asyncio
import traceback

from agent_service import chat, create_agent


async def run_chatbot(agent):
    print("NBA Analyst Agent is ready! (Type 'exit' to quit)")

    while True:
        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        try:
            response = await chat(agent, query)
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"An error occurred: {e!r}")
            traceback.print_exc()


if __name__ == "__main__":
    agent = create_agent(verbose=True)
    asyncio.run(run_chatbot(agent))
