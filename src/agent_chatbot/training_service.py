"""
High-level API for training and using XGBoost matchup models.
Used by the NBA chatbot agent tools and FastAPI ML routes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
_XGBOOST_DIR = REPO_ROOT / "xgboost"
if str(_XGBOOST_DIR) not in sys.path:
    sys.path.insert(0, str(_XGBOOST_DIR))

from model_training_script import (  # noqa: E402
    PLAYER_H2H_FEATURES,
    TEAM_FEATURES,
    build_player_h2h_dataset,
    create_player_h2h_split,
    create_team_data_split,
    evaluate_model,
    load_player_game_data,
    load_season_game_data,
    modify_data_for_team_model,
    predict_player_h2h,
    predict_team_game,
    train_model,
)

DEFAULT_DATA_DIR = REPO_ROOT / "out"
DEFAULT_MODEL_DIR = REPO_ROOT / "xgboost" / "models"

MODEL_FILES = {
    "team": "team_matchup_model.joblib",
    "player": "player_matchup_model.joblib",
}

METADATA_FILES = {
    "team": "team_matchup_metadata.json",
    "player": "player_matchup_metadata.json",
}


def metadata_path(model_type: str, model_dir: str | Path | None = None) -> Path:
    kind = _normalize_model_type(model_type)
    return _resolve_model_dir(model_dir) / METADATA_FILES[kind]


def _normalize_model_type(model_type: str) -> str:
    normalized = model_type.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "team": "team",
        "teams": "team",
        "team_vs_team": "team",
        "team_v_team": "team",
        "player": "player",
        "players": "player",
        "player_vs_player": "player",
        "player_v_player": "player",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Use 'team' (team vs team) or 'player' (player vs player)."
        )
    return aliases[normalized]


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None:
        return DEFAULT_DATA_DIR
    path = Path(data_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_model_dir(model_dir: str | Path | None) -> Path:
    if model_dir is None:
        return DEFAULT_MODEL_DIR
    path = Path(model_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def train_xgboost_model(
    model_type: str = "team",
    data_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> str:
    """
    Train an XGBoost model for team vs team or player vs player matchups.
    Returns a human-readable summary for the chatbot to relay to the user.
    """
    kind = _normalize_model_type(model_type)
    data_path = _resolve_data_dir(data_dir)
    save_path = _resolve_model_dir(model_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if kind == "team":
        data = load_season_game_data(data_path)
        processed = modify_data_for_team_model(data)
        X_train, y_train, X_val, y_val, X_test, y_test = create_team_data_split(processed)
        features = TEAM_FEATURES
        label = "team game win"
    else:
        player_data = load_player_game_data(data_path)
        h2h = build_player_h2h_dataset(player_data)
        X_train, y_train, X_val, y_val, X_test, y_test = create_player_h2h_split(h2h)
        features = PLAYER_H2H_FEATURES
        label = "player outperforming opponent in head-to-head games"

    model = train_model(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)

    model_file = save_path / MODEL_FILES[kind]
    meta_file = save_path / METADATA_FILES[kind]
    joblib.dump(model, model_file)

    metadata = {
        "model_type": kind,
        "label": label,
        "features": features,
        "metrics": metrics,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "model_path": str(model_file),
    }
    meta_file.write_text(json.dumps(metadata, indent=2))

    return (
        f"Trained {kind} vs {kind} XGBoost model.\n"
        f"Target: {label}\n"
        f"Training rows: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}\n"
        f"Validation — accuracy: {metrics['val_accuracy']:.3f}, "
        f"AUC: {metrics['val_auc']:.3f}, log loss: {metrics['val_log_loss']:.3f}\n"
        f"Test — accuracy: {metrics['test_accuracy']:.3f}, "
        f"AUC: {metrics['test_auc']:.3f}, log loss: {metrics['test_log_loss']:.3f}\n"
        f"Model saved to: {model_file}"
    )


def _load_model_and_metadata(model_type: str, model_dir: str | Path | None = None):
    kind = _normalize_model_type(model_type)
    save_path = _resolve_model_dir(model_dir)
    model_file = save_path / MODEL_FILES[kind]
    meta_file = save_path / METADATA_FILES[kind]

    if not model_file.exists():
        raise FileNotFoundError(
            f"No trained {kind} model found at {model_file}. "
            f"Ask to train a {kind} model first."
        )

    model = joblib.load(model_file)
    metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return model, metadata, kind


def predict_team_matchup(
    home_team: str,
    away_team: str,
    data_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> str:
    """Predict win probability for the home team using the trained team model."""
    model, _, _ = _load_model_and_metadata("team", model_dir)
    data_path = _resolve_data_dir(data_dir)
    result = predict_team_game(model, home_team, away_team, data_path)
    return (
        f"Matchup: {home_team} (home) vs {away_team} (away)\n"
        f"Home win probability: {result['home_win_prob']:.1%}\n"
        f"Away win probability: {result['away_win_prob']:.1%}\n"
        f"Predicted winner: {result['predicted_winner']}\n"
        f"Based on latest rolling team stats through {result['as_of_date']}."
    )


def predict_player_matchup(
    player_a: str,
    player_b: str,
    data_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> str:
    """Predict which player is more likely to outperform in a head-to-head game."""
    model, _, _ = _load_model_and_metadata("player", model_dir)
    data_path = _resolve_data_dir(data_dir)
    result = predict_player_h2h(model, player_a, player_b, data_path)
    return (
        f"Head-to-head: {player_a} vs {player_b}\n"
        f"{player_a} outperforms probability: {result['player_a_prob']:.1%}\n"
        f"{player_b} outperforms probability: {result['player_b_prob']:.1%}\n"
        f"Predicted better game: {result['predicted_better']}\n"
        f"Historical H2H games in data: {result['h2h_games']}"
    )
