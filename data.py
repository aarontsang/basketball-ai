import os
from pathlib import Path
import pandas as pd
import time
import requests
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
    # Ensure parent directory exists
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)

    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season_year)
    games_dict = game_finder.get_normalized_dict()
    games_df = pd.DataFrame(games_dict['LeagueGameFinderResults'])
    games_df.to_csv(output_filepath, index=False)

def get_teams():
    teams_list = teams.get_teams()

    df = pd.DataFrame(teams_list)
    df = df[['id', 'full_name', 'abbreviation']]
    out_path = Path('out/teams.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def get_players():
    players_list = players.get_players()

    df = pd.DataFrame(players_list)
    df = df[['id', 'full_name', 'is_active']]
    df = df[df['is_active'] == True]
    out_path = Path('out/players.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def active_player_stats(player_id: int, season_year: str) -> pd.DataFrame:
    game_finder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id, season_nullable=season_year)
    games_dict = game_finder.get_normalized_dict()
    games_df = pd.DataFrame(games_dict['LeagueGameFinderResults'])
    return games_df

def all_player_stats():
    players_path = Path('out/players.csv')
    # If the players list doesn't exist yet, generate it
    if not players_path.exists():
        players_path.parent.mkdir(parents=True, exist_ok=True)
        get_players()


    
    players = pd.read_csv(players_path)

    for player_id in players['id']:
        for season in ['2024-25', '2025-26']:
            out_file = Path(f'out/player_stats_{season}.csv')
            
            for attempt in range(5):
                try:
                    df = active_player_stats(player_id, season)
                    df['PLAYER_ID'] = player_id

                    if not df.empty:
                        out_file.parent.mkdir(parents=True, exist_ok=True)
                        if not out_file.exists():
                            df.to_csv(out_file, index=False)
                        else:
                            df.to_csv(out_file, index=False, mode='a', header=False)

                    break  # success, stop retrying
                except (TimeoutError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                    # Exponential backoff
                    wait_time = 5 * 2 ** attempt
                    print(f"Error fetching player {player_id} for {season}: {e}")
                    print(f"Retry {attempt+1}/5 in {wait_time} seconds...")
                    time.sleep(wait_time)
            




if __name__ == "__main__":
    # games_to_date('2024-25', 'out/season_games_2024_25.csv')
    # games_to_date('2025-26', 'out/season_games_2025_26.csv')
    # get_players()
    # get_teams()
    all_player_stats()
    
