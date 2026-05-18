"""
Chatbot-facing wrappers for XGBoost training and matchup prediction.
"""

from training_service import (
    predict_player_matchup as _predict_player_matchup,
    predict_team_matchup as _predict_team_matchup,
    train_xgboost_model as _train_xgboost_model,
)


def train_xgboost_model(model_type: str = "team") -> str:
    """
    Train an XGBoost model.

    model_type: 'team' / 'team_vs_team' for game-win prediction,
                'player' / 'player_vs_player' for head-to-head player performance.
    """
    try:
        return _train_xgboost_model(model_type=model_type)
    except Exception as exc:
        return f"Training failed: {exc}"


def train_team_xgboost_model() -> str:
    """Train the team vs team game-win XGBoost model. Takes no arguments."""
    return train_xgboost_model("team")


def train_player_xgboost_model() -> str:
    """Train the player vs player head-to-head XGBoost model. Takes no arguments."""
    return train_xgboost_model("player")


def predict_team_matchup(home_team: str, away_team: str) -> str:
    """Predict which team wins using the trained team model."""
    try:
        return _predict_team_matchup(home_team=home_team, away_team=away_team)
    except Exception as exc:
        return f"Team prediction failed: {exc}"


def predict_player_matchup(player_a: str, player_b: str) -> str:
    """Predict which player is likelier to have the better game."""
    try:
        return _predict_player_matchup(player_a=player_a, player_b=player_b)
    except Exception as exc:
        return f"Player prediction failed: {exc}"
