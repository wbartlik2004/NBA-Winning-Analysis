import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

# =========================================
# SETTINGS
# =========================================
SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19",
    "2021-22", "2022-23", "2023-24", "2024-25"
]

LAST_N_GAMES = 10

# Team timezone offsets
TEAM_TZ = {
    "ATL": 0, "BOS": 0, "BKN": 0, "NJN": 0, "CHA": 0, "CHO": 0, "CLE": 0,
    "DET": 0, "IND": 0, "MIA": 0, "MIL": 0, "NYK": 0,
    "ORL": 0, "PHI": 0, "TOR": 0, "WAS": 0,
    "CHI": -1, "DAL": -1, "HOU": -1, "MEM": -1, "MIN": -1, 
    "NOP": -1, "NOK": -1, "NOH": -1, "OKC": -1, "SAS": -1,
    "DEN": -2, "UTA": -2, "PHX": -2,
    "GSW": -3, "LAC": -3, "LAL": -3, "POR": -3, "SAC": -3
}

def rest_bucket(days):
    if pd.isna(days) or days >= 5:
        return "5+"
    return str(int(days))

def record_bucket(wins):
    if wins >= 9: return "9+"
    elif wins >= 7: return "7+"
    elif wins >= 5: return "5+"
    elif wins >= 3: return "3+"
    else: return "0-2"

# =========================================
# LOAD AND PREP DATA
# =========================================
print("Loading data...")
dfs = []
for season in SEASONS:
    print(f"  {season}...")
    logs = LeagueGameLog(season=season, season_type_all_star="Regular Season")
    df = logs.get_data_frames()[0]
    df["SEASON"] = season
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("\nProcessing...")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

df["Location"] = df["MATCHUP"].apply(lambda x: "Home" if "vs." in x else "Away")

# Initialize metrics
df["REST_DAYS"] = None
df["LAST10_W"] = 0
df["LAST10_L"] = 0

for team, team_df in df.groupby("TEAM_ABBREVIATION"):
    team_df = team_df.sort_values("GAME_DATE")
    idxs = team_df.index.to_list()
    dates = team_df["GAME_DATE"].tolist()
    
    for i in range(len(team_df)):
        idx = idxs[i]
        
        # Rest days
        if i > 0:
            rest_days = (dates[i] - dates[i-1]).days - 1
            df.loc[idx, "REST_DAYS"] = max(0, rest_days)
        else:
            df.loc[idx, "REST_DAYS"] = 5
        
        # Last 10 record
        recent = team_df.iloc[max(0, i - LAST_N_GAMES):i]
        df.loc[idx, "LAST10_W"] = (recent["WL"] == "W").sum()
        df.loc[idx, "LAST10_L"] = (recent["WL"] == "L").sum()

# Create buckets
df["REST_BUCKET"] = df["REST_DAYS"].apply(rest_bucket)
df["RECORD_BUCKET"] = df["LAST10_W"].apply(record_bucket)

# =========================================
# CREATE GAME-LEVEL DATA
# =========================================
print("\nCreating game-level data...")
all_games = []

for game_id, game_df in df.groupby("GAME_ID"):
    if len(game_df) != 2:
        continue
    
    home_rows = game_df[game_df["Location"] == "Home"]
    away_rows = game_df[game_df["Location"] == "Away"]
    
    if home_rows.empty or away_rows.empty:
        continue
    
    home = home_rows.iloc[0]
    away = away_rows.iloc[0]
    
    # Skip if incomplete last 10
    if (home["LAST10_W"] + home["LAST10_L"] < 10 or 
        away["LAST10_W"] + away["LAST10_L"] < 10):
        continue
    
    all_games.append({
        "HOME_REST": home["REST_BUCKET"],
        "HOME_RECORD": home["RECORD_BUCKET"],
        "AWAY_REST": away["REST_BUCKET"],
        "AWAY_RECORD": away["RECORD_BUCKET"],
        "HOME_WON": 1 if home["WL"] == "W" else 0
    })

games_df = pd.DataFrame(all_games)

print(f"Total games analyzed: {len(games_df):,}")

# =========================================
# CREATE AGGREGATED TABLE
# =========================================
print("\nCreating aggregated table...")

# Aggregate by all four dimensions
summary_table = (
    games_df
    .groupby([
        "HOME_REST",
        "HOME_RECORD",
        "AWAY_REST",
        "AWAY_RECORD"
    ])
    .agg(
        Games=("HOME_WON", "count"),
        HomeWins=("HOME_WON", "sum")
    )
    .reset_index()
)

summary_table["HomeWinPct"] = (summary_table["HomeWins"] / summary_table["Games"]).round(3)
summary_table["AwayWins"] = summary_table["Games"] - summary_table["HomeWins"]
summary_table["AwayWinPct"] = (summary_table["AwayWins"] / summary_table["Games"]).round(3)

# Reorder columns for clarity
summary_table = summary_table[[
    "HOME_REST", "HOME_RECORD",
    "AWAY_REST", "AWAY_RECORD",
    "Games", "HomeWins", "AwayWins", "HomeWinPct", "AwayWinPct"
]]

# Sort by number of games (most common first)
summary_table = summary_table.sort_values("Games", ascending=False).reset_index(drop=True)

# Create filtered versions
high_confidence = summary_table[summary_table["Games"] >= 20].copy()
extreme_scenarios = summary_table[
    (summary_table["Games"] >= 10) & 
    ((summary_table["HomeWinPct"] >= 0.60) | (summary_table["HomeWinPct"] <= 0.40))
].copy().sort_values("HomeWinPct", ascending=False).reset_index(drop=True)

# =========================================
# SAVE TO EXCEL
# =========================================
output_file = "nba_rest_record_analysis.xlsx"

print("\nSaving to Excel...")
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # All combinations
    summary_table.to_excel(writer, sheet_name='All_Combinations', index=False)
    
    # High confidence (20+ games)
    high_confidence.to_excel(writer, sheet_name='High_Confidence_20plus', index=False)
    
    # Extreme scenarios (potential betting edges)
    extreme_scenarios.to_excel(writer, sheet_name='Extreme_Scenarios', index=False)

print(f"\n✅ Saved to: {output_file}")

# =========================================
# DISPLAY KEY INSIGHTS
# =========================================
print("\n" + "="*80)
print("REST + RECORD ANALYSIS")
print("="*80)

print("\nTOP 10 MOST COMMON SCENARIOS:")
print(summary_table[["HOME_REST", "HOME_RECORD", "AWAY_REST", "AWAY_RECORD", 
                     "Games", "HomeWinPct"]].head(10).to_string(index=False))

print("\n" + "="*80)
print("TOP 10 HOME ADVANTAGES (60%+ win rate, 10+ games):")
top_home = extreme_scenarios[extreme_scenarios["HomeWinPct"] >= 0.60].head(10)
if len(top_home) > 0:
    print(top_home[["HOME_REST", "HOME_RECORD", "AWAY_REST", "AWAY_RECORD",
                    "Games", "HomeWinPct"]].to_string(index=False))
else:
    print("None found")

print("\n" + "="*80)
print("TOP 10 AWAY OPPORTUNITIES (60%+ away win rate, 10+ games):")
top_away = extreme_scenarios[extreme_scenarios["AwayWinPct"] >= 0.60].head(10)
if len(top_away) > 0:
    print(top_away[["HOME_REST", "HOME_RECORD", "AWAY_REST", "AWAY_RECORD",
                    "Games", "AwayWinPct"]].to_string(index=False))
else:
    print("None found")

print("\n" + "="*80)
print("BEST HOME SCENARIOS BY REST:")
for rest in ["5+", "4", "3", "2", "1", "0"]:
    best = summary_table[
        (summary_table["HOME_REST"] == rest) & 
        (summary_table["Games"] >= 10)
    ].nlargest(1, "HomeWinPct")
    
    if len(best) > 0:
        row = best.iloc[0]
        print(f"  {rest} days rest: {row['HOME_RECORD']} vs {row['AWAY_RECORD']} ({row['AWAY_REST']} rest) = {row['HomeWinPct']:.1%} in {int(row['Games'])} games")

print("\n" + "="*80)
print("SUMMARY STATISTICS:")
print(f"Total combinations: {len(summary_table):,}")
print(f"High confidence (20+ games): {len(high_confidence):,}")
print(f"Extreme scenarios (>60% or <40%, 10+ games): {len(extreme_scenarios):,}")
print(f"Average home win rate: {summary_table['HomeWinPct'].mean():.1%}")
print(f"Total games analyzed: {summary_table['Games'].sum():,}")

# Show rest advantage breakdown
print("\n" + "="*80)
print("HOME WIN % BY REST ADVANTAGE:")
print("="*80)
for home_rest in ["5+", "3", "1", "0"]:
    for away_rest in ["5+", "3", "1", "0"]:
        subset = summary_table[
            (summary_table["HOME_REST"] == home_rest) & 
            (summary_table["AWAY_REST"] == away_rest) &
            (summary_table["Games"] >= 5)
        ]
        if len(subset) > 0:
            avg_win = subset["HomeWinPct"].mean()
            total_games = subset["Games"].sum()
            print(f"  Home {home_rest} vs Away {away_rest}: {avg_win:.1%} ({int(total_games)} games)")

print("="*80)
