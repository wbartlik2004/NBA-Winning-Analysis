import pandas as pd

# === Load NBA stats ===
nba = pd.read_csv("nba_games_home_away_unique_safe.csv")

# Convert GAME_DATE → text BEFORE renaming
nba["GAME_DATE"] = pd.to_datetime(nba["GAME_DATE"]).dt.strftime("%Y-%m-%d")

# Rename columns to final Excel names
nba = nba.rename(columns={
    "GAME_DATE": "Date",
    "HOME_TEAM": "Home_Team",
    "AWAY_TEAM": "Away_Team",
    "HOME_RESULT": "Result",
    "HOME_MARGIN": "Margin of Victory"
})

# Ensure numeric columns are numeric
numeric_cols = [
    "Margin of Victory",
    "HOME_WINS_BEFORE",
    "HOME_LOSSES_BEFORE",
    "AWAY_WINS_BEFORE",
    "AWAY_LOSSES_BEFORE"
]

for col in numeric_cols:
    nba[col] = pd.to_numeric(nba[col], errors="coerce")

# === Load Kalshi odds ===
kalshi = pd.read_csv("kalshi_odds.csv")

kalshi["date"] = pd.to_datetime(kalshi["date"]).dt.strftime("%Y-%m-%d")

# Ensure Kalshi numbers are numeric
kalshi["home_team_spread"] = pd.to_numeric(kalshi["home_team_spread"], errors="coerce")
kalshi["home_team_odds"] = pd.to_numeric(kalshi["home_team_odds"], errors="coerce")

# Team name normalization (abbrev → full name)
team_map = {
    "OKC": "Oklahoma City Thunder",
    "HOU": "Houston Rockets",
    "LAL": "Los Angeles Lakers",
    "GSW": "Golden State Warriors",
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

# Append full names for merging with Kalshi
nba["Home_Team_Full"] = nba["Home_Team"].map(team_map)
nba["Away_Team_Full"] = nba["Away_Team"].map(team_map)

# === MERGE ===
merged = pd.merge(
    nba,
    kalshi,
    left_on=["Date", "Home_Team_Full", "Away_Team_Full"],
    right_on=["date", "home_team", "away_team"],
    how="left"
)

# === Final Columns ===
final = merged[[
    "Date",
    "Home_Team",
    "Away_Team",
    "Result",
    "Margin of Victory",
    "home_team_spread",
    "home_team_odds",
    "HOME_WINS_BEFORE",
    "HOME_LOSSES_BEFORE",
    "AWAY_WINS_BEFORE",
    "AWAY_LOSSES_BEFORE"
]]

# Rename to Excel column names
final = final.rename(columns={
    "home_team_spread": "Spread",
    "home_team_odds": "Odds"
})

# === Save to Excel in your folder ===
# Save to Excel
final.to_excel("Kalshi Charts.xlsx", index=False)

print("Created Kalshi Charts.xlsx")
