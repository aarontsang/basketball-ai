import pandas as pd
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import teaminfocommon



def get_player_career_stats(player_name, player_csv = 'out/players.csv'):
    # Find the player ID based on the name
    
    player_information = pd.read_csv(player_csv)
    player_id = player_information[player_information['full_name'] == player_name]['id'].values
    
    player_career = playercareerstats.PlayerCareerStats(player_id)

    if not player_career: 
        return f"No player found with the name '{player_name}'."
    
        
    # Get the player's career stats
    career_stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
    
    return career_stats


def get_team_stats(team_name, team_csv = 'out/teams.csv'):
    team_information = pd.read_csv(team_csv)
    team_info = team_information[team_information['full_name'] == team_name]
    
    if team_info.empty:
        return f"No team found with the name '{team_name}'."
    
    team_id = team_info['id'].values[0]
    
    team_info = teaminfocommon.TeamInfoCommon(team_id=team_id).get_data_frames()[0]
    return team_info