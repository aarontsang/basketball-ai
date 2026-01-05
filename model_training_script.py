from xgboost import XGBClassifier
import numpy as np
from pathlib import Path
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


def get_batch_csv_info(directory: str) -> list[pd.DataFrame]:
    csv_info_list = []
    for file in Path(directory).glob("*.csv"):
        csv_info_list.append(pd.read_csv(file))
    return csv_info_list


def create_opponent_data(data: pd.DataFrame) -> pd.DataFrame:
    opp = data[[
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
        "pf_rate_ewm_10"
    ]].copy()

    opp = opp.rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "WL_ewm_10": "OPP_WL_ewm_10",
        "PTS_ewm_10": "OPP_PTS_ewm_10",
        "EFG_ewm_10": "OPP_EFG_ewm_10",
        "TS_ewm_10": "OPP_TS_ewm_10",
        "ast_tov_ewm_10": "OPP_ast_tov_ewm_10",
        "tov_rate_ewm_10": "OPP_tov_rate_ewm_10",
        "oreb_rate_ewm_10": "OPP_oreb_rate_ewm_10",
        "stocks_ewm_10": "OPP_stocks_ewm_10",
        "pf_rate_ewm_10": "OPP_pf_rate_ewm_10"
    })

    merged = data.merge(
        opp,
        on=["GAME_ID", "GAME_DATE"]
    )


    # remove self-joins
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]]

    return merged


def modify_data_for_model(data: pd.DataFrame) -> pd.DataFrame:
    stat_list = ["WL", "PTS", "EFG", "TS", "ast_tov", "tov_rate", "oreb_rate", "stocks", "pf_rate"]
    # Example modification: Fill NaN values with 0
    data = data.fillna(0)
    # Add more feature engineering steps as needed

    # Add IS_HOME column
    data["IS_HOME"] = data["MATCHUP"].apply(is_home)

    # Sort by TEAM_ID and GAME_DATE
    sort_by_list = ["TEAM_ID", "GAME_DATE"]
    data = data.sort_values(by=sort_by_list, inplace=True)

    # Change WL to binary
    data["WL"] = data["MATCHUP"].apply(is_win)

    #Effective field goal percentage
    data["EFG"] = (data["FGM"] + 0.5 * data["FG3M"]) / data["FGA"]

    #True shooting percentage
    data["TS"] = data["PTS"] / (2 * (data["FGA"] + 0.44 * data["FTA"]))

    #Assist to turnover ratio
    data["ast_tov"] = data["AST"] / (data["TOV"] + 1e-8)  # Add small value to avoid division by zero)

    #Turnover rate
    data["tov_rate"] = data["TOV"] / (data["FGA"] + 0.44 * data["FTA"] + 1e-8) # Add small value to avoid division by zero

    #offensive rebound rate
    data["oreb_rate"] = data["OREB"] / (data["OREB"] + data["DREB"] + 1e-8) # Add small value to avoid division by zero

    #stealing and blocks
    data["stocks"] = data["STL"] + data["BLK"]

    #Personal fouls rate
    data["pf_rate"] = data["PF"] / (data["MIN"] + 1e-8) # Add small value to avoid division by zero

    #Get wins so far
    data["wins_so_far"] = (
        data.groupby("TEAM_ID")["WL"]
        .shift(1)
        .cumsum()
    )

    data["games_so_far"] = (
        data.groupby("TEAM_ID")["WL"]
        .cumcount()
    )

    #Get win percentage so far
    data["win_pct"] = data["wins_so_far"] / (data["games_so_far"] + 1e-8)  # Add small value to avoid division by zero

    #Rolling win percentages
    data["win_percentage_last_5"] = (
        data.groupby("TEAM_ID")["WL"]
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )

    data["win_percentage_last_10"] = (
        data.groupby("TEAM_ID")["WL"]
        .shift(1)
        .rolling(window=10, min_periods=1)
        .mean()
    )

    #Exponential weighted moving averages
    for stat in stat_list:
        data[f"{stat}_ewm_5"] = (
            data.groupby("TEAM_ID")[stat]
            .shift(1)
            .ewm(span=5, min_periods=1)
            .mean()
        )

        data[f"{stat}_ewm_10"] = (
            data.groupby("TEAM_ID")[stat]
            .shift(1)
            .ewm(span=10, min_periods=1)
            .mean()
        )

    #Winstreaks
    data["winstreak"] = (
        data.groupby("TEAM_ID")["WL"]
          .shift(1)
          .groupby(data["TEAM_ID"])
          .transform(lambda x: x.groupby((x != 1).cumsum()).cumcount())

    ).clip(-0.1,0.1) #make sure model doesn't interpret large winstreaks as more important than they are

    #Adds opponent data
    data = create_opponent_data(data)

    subtract_stat_list = ["WL_ewm_10", "PTS_ewm_10", "EFG_ewm_10", "TS_ewm_10", "ast_tov_ewm_10", "tov_rate_ewm_10", "oreb_rate_ewm_10", "stocks_ewm_10", "pf_rate_ewm_10"]
    for stat in subtract_stat_list:
        if stat == "WL_ewm_10":
            data[f"{stat}_diff"] = (data[stat] - data[f"OPP_{stat}"]).clip(-0.1,0.1)
        else:
            data[f"{stat}_diff"] = data[stat] - data[f"OPP_{stat}"]

    return data


def create_data_split(data: pd.DataFrame):
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    seaon_id_2024_25 = data["SEASON_ID"].unique()
    season_id_2025_26 = data["SEASON_ID"].unique()

    features = [
        "WL_ewm_10_diff",
        # "winstreak",
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
    target = "WL"

    train_mask = data["SEASON_ID"].isin(seaon_id_2024_25)
    test_mask  = data["SEASON_ID"].isin(season_id_2025_26)

    X_train_list.append(data.loc[train_mask, features])
    y_train_list.append(data.loc[train_mask, target])

    X_test_list.append(data.loc[test_mask, features])
    y_test_list.append(data.loc[test_mask, target])

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)

    mid = len(X_test_list) // 2

    X_test_list = X_test_list[mid:]
    y_test_list = y_test_list[mid:]

    X_val_list = X_test_list[:mid]
    y_val_list = y_test_list[:mid]

    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)

    X_val = pd.concat(X_val_list, ignore_index=True)
    y_val = pd.concat(y_val_list, ignore_index=True)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> FrozenEstimator:
    model = XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    calibrated_model = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val)

    frozen_model = FrozenEstimator(calibrated_model)

    return frozen_model


if __name__ == "__main__":
    pass