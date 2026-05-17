from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


TEAM_FEATURES = [
    "WL_ewm_10_diff",
    "PTS_ewm_10_diff",
    "EFG_ewm_10_diff",
    "TS_ewm_10_diff",
    "ast_tov_ewm_10_diff",
    "tov_rate_ewm_10_diff",
    "oreb_rate_ewm_10_diff",
    "stocks_ewm_10_diff",
    "pf_rate_ewm_10_diff",
    "PTS_ewm_10",
    "EFG_ewm_10",
    "TS_ewm_10",
    "ast_tov_ewm_10",
    "tov_rate_ewm_10",
    "oreb_rate_ewm_10",
    "stocks_ewm_10",
    "pf_rate_ewm_10",
    "IS_HOME",
]

PLAYER_H2H_FEATURES = [
    "PTS_ewm_10_diff",
    "REB_ewm_10_diff",
    "AST_ewm_10_diff",
    "game_score_ewm_10_diff",
    "MIN_ewm_10_diff",
]

MIN_PLAYER_MINUTES = 15


def is_home(matchup: str) -> int:
    return int("@" not in matchup)


def wl_to_binary(result: str) -> int:
    return int(str(result).upper() == "W")


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by="GAME_DATE")


def load_season_game_data(data_dir: Path) -> pd.DataFrame:
    frames = []
    for pattern in ("season_games_2024-25.csv", "season_games_2025-26.csv"):
        path = data_dir / pattern
        if path.exists():
            df = pd.read_csv(path)
            df["SEASON_LABEL"] = pattern.replace("season_games_", "").replace(".csv", "")
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No season game CSVs found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def load_player_game_data(data_dir: Path) -> pd.DataFrame:
    frames = []
    for pattern in ("player_stats_2024-25.csv", "player_stats_2025-26.csv"):
        path = data_dir / pattern
        if path.exists():
            df = pd.read_csv(path)
            df["SEASON_LABEL"] = pattern.replace("player_stats_", "").replace(".csv", "")
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No player stats CSVs found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def create_opponent_data(data: pd.DataFrame) -> pd.DataFrame:
    opp = data[
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID",
            "WL_ewm_10",
            "PTS_ewm_10",
            "EFG_ewm_10",
            "TS_ewm_10",
            "ast_tov_ewm_10",
            "tov_rate_ewm_10",
            "oreb_rate_ewm_10",
            "stocks_ewm_10",
            "pf_rate_ewm_10",
        ]
    ].copy()

    opp = opp.rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "WL_ewm_10": "OPP_WL_ewm_10",
            "PTS_ewm_10": "OPP_PTS_ewm_10",
            "EFG_ewm_10": "OPP_EFG_ewm_10",
            "TS_ewm_10": "OPP_TS_ewm_10",
            "ast_tov_ewm_10": "OPP_ast_tov_ewm_10",
            "tov_rate_ewm_10": "OPP_tov_rate_ewm_10",
            "oreb_rate_ewm_10": "OPP_oreb_rate_ewm_10",
            "stocks_ewm_10": "OPP_stocks_ewm_10",
            "pf_rate_ewm_10": "OPP_pf_rate_ewm_10",
        }
    )

    merged = data.merge(opp, on=["GAME_ID", "GAME_DATE"])
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]]
    return merged


def modify_data_for_team_model(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    stat_list = [
        "WL",
        "PTS",
        "EFG",
        "TS",
        "ast_tov",
        "tov_rate",
        "oreb_rate",
        "stocks",
        "pf_rate",
    ]

    data["IS_HOME"] = data["MATCHUP"].apply(is_home)
    data = data.sort_values(by=["TEAM_ID", "GAME_DATE"])
    data["WL"] = data["WL"].apply(wl_to_binary)

    data["EFG"] = (data["FGM"] + 0.5 * data["FG3M"]) / (data["FGA"] + 1e-8)
    data["TS"] = data["PTS"] / (2 * (data["FGA"] + 0.44 * data["FTA"]) + 1e-8)
    data["ast_tov"] = data["AST"] / (data["TOV"] + 1e-8)
    data["tov_rate"] = data["TOV"] / (data["FGA"] + 0.44 * data["FTA"] + 1e-8)
    data["oreb_rate"] = data["OREB"] / (data["OREB"] + data["DREB"] + 1e-8)
    data["stocks"] = data["STL"] + data["BLK"]
    data["pf_rate"] = data["PF"] / (data["MIN"].replace(0, np.nan) + 1e-8)

    for stat in stat_list:
        data[f"{stat}_ewm_10"] = (
            data.groupby("TEAM_ID")[stat]
            .shift(1)
            .ewm(span=10, min_periods=1)
            .mean()
        )

    data = create_opponent_data(data)

    subtract_stat_list = [
        "WL_ewm_10",
        "PTS_ewm_10",
        "EFG_ewm_10",
        "TS_ewm_10",
        "ast_tov_ewm_10",
        "tov_rate_ewm_10",
        "oreb_rate_ewm_10",
        "stocks_ewm_10",
        "pf_rate_ewm_10",
    ]
    for stat in subtract_stat_list:
        if stat == "WL_ewm_10":
            data[f"{stat}_diff"] = (data[stat] - data[f"OPP_{stat}"]).clip(-0.1, 0.1)
        else:
            data[f"{stat}_diff"] = data[stat] - data[f"OPP_{stat}"]

    return data.dropna(subset=TEAM_FEATURES + ["WL"])


def create_team_data_split(data: pd.DataFrame):
    train_mask = data["SEASON_LABEL"] == "2024-25"
    test_mask = data["SEASON_LABEL"] == "2025-26"
    target = "WL"

    X_train = data.loc[train_mask, TEAM_FEATURES]
    y_train = data.loc[train_mask, target]
    X_test_full = data.loc[test_mask, TEAM_FEATURES]
    y_test_full = data.loc[test_mask, target]

    mid = len(X_test_full) // 2
    X_val = X_test_full.iloc[:mid]
    y_val = y_test_full.iloc[:mid]
    X_test = X_test_full.iloc[mid:]
    y_test = y_test_full.iloc[mid:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def _player_game_score(row: pd.Series) -> float:
    return (
        row["PTS"]
        + 0.7 * row["REB"]
        + 0.7 * row["AST"]
        + row["STL"]
        + row["BLK"]
        - row["TOV"]
    )


def _add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_score"] = df.apply(_player_game_score, axis=1)
    df = df.sort_values(by=["PLAYER_ID", "GAME_DATE"])

    for stat in ("PTS", "REB", "AST", "MIN", "game_score"):
        df[f"{stat}_ewm_10"] = (
            df.groupby("PLAYER_ID")[stat]
            .shift(1)
            .ewm(span=10, min_periods=1)
            .mean()
        )
    return df


def build_player_h2h_dataset(player_data: pd.DataFrame) -> pd.DataFrame:
    """Build rows for player-vs-player: label=1 if player_a outscores player_b in the same game."""
    df = _add_player_rolling_features(player_data)
    df = df[df["MIN"] >= MIN_PLAYER_MINUTES]

    pairs = []
    for game_id, game_df in df.groupby("GAME_ID"):
        teams = game_df["TEAM_ID"].unique()
        if len(teams) != 2:
            continue

        team_a, team_b = teams
        side_a = game_df[game_df["TEAM_ID"] == team_a]
        side_b = game_df[game_df["TEAM_ID"] == team_b]

        top_a = side_a.nlargest(3, "MIN")
        top_b = side_b.nlargest(3, "MIN")

        for _, row_a in top_a.iterrows():
            for _, row_b in top_b.iterrows():
                if row_a["PLAYER_ID"] == row_b["PLAYER_ID"]:
                    continue
                label = int(row_a["game_score"] > row_b["game_score"])
                pairs.append(
                    {
                        "GAME_ID": game_id,
                        "GAME_DATE": row_a["GAME_DATE"],
                        "SEASON_LABEL": row_a["SEASON_LABEL"],
                        "PLAYER_A_ID": row_a["PLAYER_ID"],
                        "PLAYER_B_ID": row_b["PLAYER_ID"],
                        "PLAYER_A_NAME": row_a.get("PLAYER_NAME", row_a["PLAYER_ID"]),
                        "PLAYER_B_NAME": row_b.get("PLAYER_NAME", row_b["PLAYER_ID"]),
                        "PTS_ewm_10_diff": row_a["PTS_ewm_10"] - row_b["PTS_ewm_10"],
                        "REB_ewm_10_diff": row_a["REB_ewm_10"] - row_b["REB_ewm_10"],
                        "AST_ewm_10_diff": row_a["AST_ewm_10"] - row_b["AST_ewm_10"],
                        "game_score_ewm_10_diff": row_a["game_score_ewm_10"]
                        - row_b["game_score_ewm_10"],
                        "MIN_ewm_10_diff": row_a["MIN_ewm_10"] - row_b["MIN_ewm_10"],
                        "label": label,
                    }
                )

    h2h = pd.DataFrame(pairs)
    return h2h.dropna(subset=PLAYER_H2H_FEATURES + ["label"])


def create_player_h2h_split(h2h: pd.DataFrame):
    train_mask = h2h["SEASON_LABEL"] == "2024-25"
    test_mask = h2h["SEASON_LABEL"] == "2025-26"
    target = "label"

    X_train = h2h.loc[train_mask, PLAYER_H2H_FEATURES]
    y_train = h2h.loc[train_mask, target]
    X_test_full = h2h.loc[test_mask, PLAYER_H2H_FEATURES]
    y_test_full = h2h.loc[test_mask, target]

    mid = len(X_test_full) // 2
    X_val = X_test_full.iloc[:mid]
    y_val = y_test_full.iloc[:mid]
    X_test = X_test_full.iloc[mid:]
    y_test = y_test_full.iloc[mid:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> FrozenEstimator:
    model = XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # sklearn >= 1.6: wrap the fitted estimator in FrozenEstimator instead of cv="prefit"
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(model), method="isotonic"
    )
    calibrated_model.fit(X_val, y_val)
    return FrozenEstimator(calibrated_model)


def evaluate_model(model, X_val, y_val, X_test, y_test) -> dict[str, float]:
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    return {
        "val_accuracy": float(accuracy_score(y_val, val_probs >= 0.5)),
        "val_auc": float(roc_auc_score(y_val, val_probs)),
        "val_log_loss": float(log_loss(y_val, val_probs)),
        "test_accuracy": float(accuracy_score(y_test, test_probs >= 0.5)),
        "test_auc": float(roc_auc_score(y_test, test_probs)),
        "test_log_loss": float(log_loss(y_test, test_probs)),
    }


def _resolve_team_id(team_name: str, teams_csv: Path) -> int | None:
    teams = pd.read_csv(teams_csv)
    for col in ("full_name", "nickname", "abbreviation", "city"):
        if col not in teams.columns:
            continue
        match = teams[teams[col].str.lower() == team_name.lower()]
        if not match.empty:
            return int(match.iloc[0]["id"])
    partial = teams[teams["full_name"].str.lower().str.contains(team_name.lower(), na=False)]
    if not partial.empty:
        return int(partial.iloc[0]["id"])
    return None


def _resolve_player_id(player_name: str, players_csv: Path) -> int | None:
    players = pd.read_csv(players_csv)
    exact = players[players["full_name"].str.lower() == player_name.lower()]
    if not exact.empty:
        return int(exact.iloc[0]["id"])
    partial = players[players["full_name"].str.lower().str.contains(player_name.lower(), na=False)]
    if not partial.empty:
        return int(partial.iloc[0]["id"])
    return None


def predict_team_game(model, home_team: str, away_team: str, data_dir: Path) -> dict:
    teams_csv = data_dir / "teams.csv"
    home_id = _resolve_team_id(home_team, teams_csv)
    away_id = _resolve_team_id(away_team, teams_csv)
    if home_id is None or away_id is None:
        raise ValueError(f"Could not resolve teams: home={home_team!r}, away={away_team!r}")

    data = load_season_game_data(data_dir)
    processed = modify_data_for_team_model(data)

    home_row = processed[processed["TEAM_ID"] == home_id].sort_values("GAME_DATE").iloc[-1]
    away_row = processed[processed["TEAM_ID"] == away_id].sort_values("GAME_DATE").iloc[-1]

    features = {}
    diff_stats = [
        "WL_ewm_10",
        "PTS_ewm_10",
        "EFG_ewm_10",
        "TS_ewm_10",
        "ast_tov_ewm_10",
        "tov_rate_ewm_10",
        "oreb_rate_ewm_10",
        "stocks_ewm_10",
        "pf_rate_ewm_10",
    ]
    for stat in diff_stats:
        diff = home_row[stat] - away_row[stat]
        if stat == "WL_ewm_10":
            diff = np.clip(diff, -0.1, 0.1)
        features[f"{stat}_diff"] = diff

    for stat in [
        "PTS_ewm_10",
        "EFG_ewm_10",
        "TS_ewm_10",
        "ast_tov_ewm_10",
        "tov_rate_ewm_10",
        "oreb_rate_ewm_10",
        "stocks_ewm_10",
        "pf_rate_ewm_10",
    ]:
        features[stat] = home_row[stat]
    features["IS_HOME"] = 1

    X = pd.DataFrame([features])[TEAM_FEATURES]
    prob = float(model.predict_proba(X)[0, 1])

    return {
        "home_win_prob": prob,
        "away_win_prob": 1 - prob,
        "predicted_winner": home_team if prob >= 0.5 else away_team,
        "as_of_date": str(home_row["GAME_DATE"]),
    }


def predict_player_h2h(model, player_a: str, player_b: str, data_dir: Path) -> dict:
    players_csv = data_dir / "players.csv"
    id_a = _resolve_player_id(player_a, players_csv)
    id_b = _resolve_player_id(player_b, players_csv)
    if id_a is None or id_b is None:
        raise ValueError(f"Could not resolve players: {player_a!r}, {player_b!r}")

    player_data = load_player_game_data(data_dir)
    df = _add_player_rolling_features(player_data)

    row_a = df[df["PLAYER_ID"] == id_a].sort_values("GAME_DATE").iloc[-1]
    row_b = df[df["PLAYER_ID"] == id_b].sort_values("GAME_DATE").iloc[-1]

    features = {
        "PTS_ewm_10_diff": row_a["PTS_ewm_10"] - row_b["PTS_ewm_10"],
        "REB_ewm_10_diff": row_a["REB_ewm_10"] - row_b["REB_ewm_10"],
        "AST_ewm_10_diff": row_a["AST_ewm_10"] - row_b["AST_ewm_10"],
        "game_score_ewm_10_diff": row_a["game_score_ewm_10"] - row_b["game_score_ewm_10"],
        "MIN_ewm_10_diff": row_a["MIN_ewm_10"] - row_b["MIN_ewm_10"],
    }
    X = pd.DataFrame([features])
    prob_a = float(model.predict_proba(X)[0, 1])

    h2h_games = len(
        set(df[df["PLAYER_ID"] == id_a]["GAME_ID"])
        & set(df[df["PLAYER_ID"] == id_b]["GAME_ID"])
    )

    name_a = player_a
    name_b = player_b
    return {
        "player_a_prob": prob_a,
        "player_b_prob": 1 - prob_a,
        "predicted_better": name_a if prob_a >= 0.5 else name_b,
        "h2h_games": h2h_games,
    }


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    data = load_season_game_data(repo / "out")
    preprocessed = modify_data_for_team_model(data)
    X_train, y_train, X_val, y_val, X_test, y_test = create_team_data_split(preprocessed)
    trained = train_model(X_train, y_train, X_val, y_val)
    print(evaluate_model(trained, X_val, y_val, X_test, y_test))
