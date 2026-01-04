from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import shap


def is_home(matchup):
    return int("@" not in matchup)


def is_win(result):
    return int("L" not in result)


def sort_by_date(df):
    return df.sort_values(by='GAME_DATE')


def get_csv_info(directory: str) -> pd.DataFrame:
    nba_player_data_2025_26 = pd.read_csv('out/player_stats_2025-26.csv')
    nba_player_data_2024_25 = pd.read_csv('out/player_stats_2024-25.csv')
    nba_players = pd.read_csv('out/players.csv')
    nba_teams = pd.read_csv('out/teams.csv')
    nba_season_games_2024_25 = pd.read_csv('out/season_games_2024-25.csv')
    nba_season_games_2025_26 = pd.read_csv('out/season_games_2025-26.csv')


if __name__ == "__main__":
    pass