from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd

pd.set_option('display.max_rows', None)

# Pull all team game logs
logs = LeagueGameLog(season='2025-26', season_type_all_star='Regular Season')
df = logs.get_data_frames()[0]
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values('GAME_DATE')

def get_record_before(df_team, date):
    past_games = df_team[df_team['GAME_DATE'] < date]
    wins = (past_games['WL'] == 'W').sum()
    losses = (past_games['WL'] == 'L').sum()
    return wins, losses

games = []

for game_id, group in df.groupby('GAME_ID'):
    # Try to find the home row
    home_row = group[group['MATCHUP'].str.contains(" vs. ")]
    
    if not home_row.empty:
        home_row = home_row.iloc[0]
        home_team = home_row['TEAM_ABBREVIATION']
        away_team = home_row['MATCHUP'].split(" vs. ")[1]
    else:
        # If " vs. " is missing, fallback to away row and infer home team
        away_row = group[group['MATCHUP'].str.contains(" @ ")].iloc[0]
        away_team = away_row['TEAM_ABBREVIATION']
        home_team = away_row['MATCHUP'].split(" @ ")[1]
        home_row = away_row  # treat this row as reference

    game_date = home_row['GAME_DATE']
    home_pts = home_row['PTS']
    
    # Get the away points from the other row
    away_pts = group[group['TEAM_ABBREVIATION'] == away_team]['PTS'].values[0]

    home_margin = home_pts - away_pts
    home_result = "W" if home_margin > 0 else "L"

    # Compute records
    home_logs = df[df['TEAM_ABBREVIATION'] == home_team]
    away_logs = df[df['TEAM_ABBREVIATION'] == away_team]

    home_w, home_l = get_record_before(home_logs, game_date)
    away_w, away_l = get_record_before(away_logs, game_date)

    games.append({
        "GAME_DATE": game_date,
        "HOME_TEAM": home_team,
        "AWAY_TEAM": away_team,
        "HOME_RESULT": home_result,
        "HOME_WINS_BEFORE": home_w,
        "HOME_LOSSES_BEFORE": home_l,
        "AWAY_WINS_BEFORE": away_w,
        "AWAY_LOSSES_BEFORE": away_l,
        "HOME_MARGIN": home_margin
    })

# Convert to DataFrame
final_df = pd.DataFrame(games).sort_values("GAME_DATE")

# Save CSV
output_path = "nba_games_home_away_unique_safe.csv"
final_df.to_csv(output_path, index=False)

print(f"Saved file to {output_path}")
print(final_df.head())
