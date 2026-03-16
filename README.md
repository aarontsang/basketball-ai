# Basketball AI

## Project Breakdown

## Frameworks

## Steps to Run

## Libraries and External Packages
  * pandas (https://pandas.pydata.org/)
  * LlamaIndex (https://developers.llamaindex.ai/python/framework/)
  * nba_api  (https://github.com/swar/nba_api)
  * thefuzz (https://github.com/seatgeek/thefuzz)

## Code
  * LeChatBot.py Runs the entire chatbot, converts wikipedia pages into vectors, and initializes LLM and agents (156 lines)
  * agent_tools.py Agent tools for the chatbot such as API calling (63 lines)
  * question_generator.py Script for generating questions for testing (127 lines)
  * stats_query_handler.py Functions for the agent, used for calling nba_api for stats (112 lines)

### Data Collection
1. Ensure there is an out folder under the repo.
2. Set up the python environment as below:
   * ```python3 -m venv .venv```
   * ```source ./.venv/bin/activate```
   * ```pip install -r requirements.txt```
3. In the python environment, run ```python3 data.py```. Because there are a lot of player stats, it will take awhile.

### AI Model
