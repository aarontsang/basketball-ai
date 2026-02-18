from llama_index.core.tools import FunctionTool
from stats_query_handler import get_player_career_stats, get_team_stats

stats_tool = FunctionTool.from_defaults(
    fn=get_player_career_stats,
    name="nba_stats_tool",
    description="Use this tool to get real NBA player or team statistics."
)

# prediction_tool = FunctionTool.from_defaults(
#     fn=predict_game,
#     name="xgboost_prediction_tool",
#     description="Use this tool when user asks to predict game outcomes."
# )

# rag_tool = FunctionTool.from_defaults(
#     fn=rag_query,
#     name="wikipedia_basketball_tool",
#     description="Use this tool for basketball history or contextual questions."
# )