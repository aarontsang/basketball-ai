"""
Tools for the NBA chatbot agent: RAG, stats lookup, XGBoost training, and matchup prediction.
"""

from llama_index.core.tools import FunctionTool
from stats_query_handler import get_player_stats, get_team_info
from ml_training_handler import (
    train_team_xgboost_model,
    train_player_xgboost_model,
    predict_team_matchup,
    predict_player_matchup,
)


def get_player_stats_tool():
    return FunctionTool.from_defaults(
        fn=get_player_stats,
        name="player_nba_stats_tool",
        description=(
            "Use for quantitative NBA player statistics (points, rebounds, assists, etc.). "
            "Parameters: player_name (required), scope ('season'|'career'), "
            "competition ('regular'|'postseason'|'allstar'), view ('totals'|'rankings'), "
            "season (optional, e.g. '2023-24')."
        ),
    )


def get_team_stats_tool():
    return FunctionTool.from_defaults(
        fn=get_team_info,
        name="team_nba_stats_tool",
        description=(
            "Use for NBA team info and roster details. "
            "Parameters: team_name (required, e.g. 'Los Angeles Lakers')."
        ),
    )


def get_train_team_xgboost_tool():
    return FunctionTool.from_defaults(
        fn=train_team_xgboost_model,
        name="train_team_xgboost_model",
        return_direct=True,
        description=(
            "Train the team-vs-team XGBoost model (predicts game winners). "
            "Use when the user asks to train/build/retrain a TEAM or team-vs-team model. "
            "No parameters — use Action Input: {}."
        ),
    )


def get_train_player_xgboost_tool():
    return FunctionTool.from_defaults(
        fn=train_player_xgboost_model,
        name="train_player_xgboost_model",
        return_direct=True,
        description=(
            "Train the player-vs-player XGBoost model (predicts who outperforms in a matchup). "
            "Use when the user asks to train/build/retrain a PLAYER or player-vs-player model. "
            "No parameters — use Action Input: {}."
        ),
    )


def get_predict_team_matchup_tool():
    return FunctionTool.from_defaults(
        fn=predict_team_matchup,
        name="predict_team_matchup",
        return_direct=True,
        description=(
            "Predict a game winner after train_team_xgboost_model has been run. "
            "Parameters: home_team, away_team (e.g. home_team='Los Angeles Lakers', "
            "away_team='Boston Celtics')."
        ),
    )


def get_predict_player_matchup_tool():
    return FunctionTool.from_defaults(
        fn=predict_player_matchup,
        name="predict_player_matchup",
        return_direct=True,
        description=(
            "Predict which player outperforms after train_player_xgboost_model has been run. "
            "Parameters: player_a, player_b (full names)."
        ),
    )
