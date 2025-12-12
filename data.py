import pandas as pd

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.live.nba.endpoints import boxscore

def games_to_date(season_year: str, output_filepath: str) -> None:
    """Generates a CSV file containing all NBA games for a given season.

    Args:
        season_year (str): The NBA season year in 'YYYY-YY' format (e.g., '2022-23').
        output_filepath (str): The file path where the CSV will be saved.
    """
    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season_year)
    games_dict = game_finder.get_normalized_dict()
    games_df = pd.DataFrame(games_dict['LeagueGameFinderResults'])
    games_df.to_csv(output_filepath, index=False)

def get_teams():
    teams_list = teams.get_teams()

    df = pd.DataFrame(teams_list)
    df = df[['id', 'full_name', 'abbreviation']]
    df.to_csv('out/teams.csv', index=False)

def get_players():
    players_list = players.get_players()

    df = pd.DataFrame(players_list)
    df = df[['id', 'full_name', 'is_active']]
    df = df[df['is_active'] == True]
    df.to_csv('out/players.csv', index=False)

def active_player_stats(player_id: int, season_year: str) -> pd.DataFrame:
    game_finder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id, season_nullable=season_year)
    games_dict = game_finder.get_normalized_dict()
    games_df = pd.DataFrame(games_dict['LeagueGameFinderResults'])
    return games_df

def all_player_stats():
    players = pd.read_csv('out/players.csv')
    for id in players['id']:
        df = active_player_stats(id, '2024-25')
        df.to_csv(f'out/player_stats_2024_25.csv', index=False, mode='a', header=False)

        df = active_player_stats(id, '2025-26')
        df.to_csv(f'out/player_stats_2025_26.csv', index=False, mode='a', header=False)

def get_todays_games():
    sb = scoreboard.Scoreboard()
    games = sb.get_dict()['games']
    return games
    
if __name__ == "__main__":
    all_player_stats()
    get_players()
    get_teams()
    games_to_date('2025-26', 'out/season_games_2025_26.csv')