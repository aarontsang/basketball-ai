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


def modify_data_for_model(data: pd.DataFrame) -> pd.DataFrame:
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



if __name__ == "__main__":
    pass