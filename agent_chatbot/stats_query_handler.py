import pandas as pd
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import teaminfocommon


def _get_player_id(player_name, player_csv='out/players.csv'):
    df = pd.read_csv(player_csv)
    row = df[df['full_name'] == player_name]

    if row.empty:
        return None

    return int(row['id'].values[0])


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