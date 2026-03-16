"""
This file defines functions to query NBA player and team statistics using the nba_api 
library. The functions include:
- get_player_stats: Retrieves various statistics for a given NBA player 
based on parameters like scope, competition, and stat view.
- get_team_info: Retrieves information about a given NBA team.
The functions are designed to be used as tools within the chatbot agent, allowing it 
to provide quantitative insights about players and teams when users ask for specific stats.
"""

import pandas as pd
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import teaminfocommon
from thefuzz import process


def _get_player_id(player_name, player_csv='out/players.csv'):
    df = pd.read_csv(player_csv)
    row = df[df['full_name'] == player_name]

    if row.empty:
        return None

    return int(row['id'].values[0])


def normalize_player_name(user_input):
    """
    Converts 'Steph' -> 'Stephen Curry' or handles 'Lebron' -> 'LeBron James'.
    """
    # 1. Try the built-in NBA API search (Great for partial matches)
    nba_results = players.find_players_by_full_name(user_input)
    if nba_results:
        # Return the most relevant 'full_name' and its 'id'
        return nba_results[0]['full_name'], nba_results[0]['id']

    # 2. Fallback: Fuzzy matching against the entire league list
    # This catches typos like 'Stphen Curry'
    all_players = [p['full_name'] for p in players.get_players()]
    # extractOne returns (best_match, score)
    best_match, score = process.extractOne(user_input, all_players)
    
    # Only return if the match is confident (score > 80)
    if score > 80:
        match_data = players.find_players_by_full_name(best_match)[0]
        return match_data['full_name'], match_data['id']
    
    return None, None


def get_player_stats(
    player_name,
    scope="career",              # "career" | "season"
    competition="regular",       # "regular" | "postseason" | "allstar"
    view="totals",               # "totals" | "rankings"
    season=None,
    player_csv='out/players.csv'
):
    player_id = _get_player_id(player_name, player_csv)

    if player_id is None:
        return f"No player found with the name '{player_name}'."

    stats = playercareerstats.PlayerCareerStats(player_id=player_id)

    dataset_map = {
        ("career", "regular", "totals"): stats.career_totals_regular_season,
        ("career", "postseason", "totals"): stats.career_totals_post_season,
        ("career", "allstar", "totals"): stats.career_totals_all_star_season,
        ("season", "regular", "totals"): stats.season_totals_regular_season,
        ("season", "postseason", "totals"): stats.season_totals_post_season,
        ("season", "allstar", "totals"): stats.season_totals_all_star_season,
        ("season", "regular", "rankings"): stats.season_rankings_regular_season,
        ("season", "postseason", "rankings"): stats.season_rankings_post_season,
    }

    #Normalize input
    scope = scope.lower().strip()
    competition = competition.lower().strip()
    view = view.lower().strip()

    key = (scope, competition, view)

    if key not in dataset_map:
        return "Invalid stats configuration."

    df = dataset_map[key].get_data_frame()

    if scope == "season" and season:
        df = df[df["SEASON_ID"] == season]

    return df


def _get_team_id(team_name, team_csv='out/teams.csv'):
    df = pd.read_csv(team_csv)
    row = df[df['full_name'] == team_name]

    if row.empty:
        return None

    return int(row['id'].values[0])


def get_team_info(team_name, team_csv='out/teams.csv'):
    team_id = _get_team_id(team_name, team_csv)

    if not team_id:
        return f"No team found with the name '{team_name}'."

    return teaminfocommon.TeamInfoCommon(team_id=team_id).get_data_frames()[0]
