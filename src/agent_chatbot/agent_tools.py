"""
This file defines the tools that the agent can use to answer user queries.
The tools include:
- A RAG tool that uses a query engine to retrieve information from the document index.
- A player stats tool that retrieves quantitative NBA player statistics and provides analysis.
- A team stats tool that retrieves NBA team statistics.
"""

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from stats_query_handler import get_player_stats, get_team_info



def get_player_stats_tool():

    player_stats_tool = FunctionTool.from_defaults(
        fn=get_player_stats,
        name="player_nba_stats_tool",
        description="""
        CRITICAL: Use this tool when the user asks for quantative NBA player statistics. 
        including both traditional stats (points, rebounds, assists) and advanced metrics (PER, TS%, etc.).
        Extract the parameters below from the user's query and pass them to the function.

        Parameters:
        - player_name: full name of NBA player
        - scope: 'season' or 'career'
        - competition: 'regular', 'postseason', or 'allstar'
        - stat_view: 'per_game', 'totals', or 'advanced'
        - season: season in format '2023-24' (optional)


        After getting the stats, provide a concise analysis of the player's 
        performance based on the numbers. Highlight any notable strengths, weaknesses, 
        or trends in the data. For example, if a player's PER is particularly high,
        mention that they are performing well overall. If their TS% is low, note that 
        they may be struggling with shooting efficiency. Use the stats to provide 
        insights into the player's game and how they compare to league averages or
        other players at their position.
        """
    )
    return player_stats_tool



def get_team_stats_tool():
    team_stats_tool = FunctionTool.from_defaults(
    fn=get_team_info,
    name="team_nba_stats_tool",
    description="""
    Use this tool when the user asks for NBA team statistics. 
    Extract the parameters below from the user's query and pass them to the function.

    Parameters:
    - team_name: full name of NBA team
    - scope: 'season' or 'career'
    - competition: 'regular', 'postseason', or 'allstar'
    - stat_view: 'per_game', 'totals', or 'advanced'
    - season: season in format '2023-24' (optional)
    """
    )
    return team_stats_tool
    