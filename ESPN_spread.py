import requests
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import time

INPUT_CSV = "nba_games_ids_teams.csv"
OUTPUT_CSV = "nba_spreads_time.csv"

def get_game_info(game_id):
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    r = requests.get(url).json()

    # ----- Start Time (ISO) -----
    iso_time = r.get("header", {}).get("competitions", [{}])[0].get("date")

    if iso_time:
        # Convert ISO to datetime
        dt_utc = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))

        # Convert UTC → EST/EDT (America/New_York)
        dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))

        # yyyy-mm-dd
        game_date = dt_est.strftime("%Y-%m-%d")

        # EST time, e.g., 7:30 PM
        est_time = dt_est.strftime("%I:%M %p").lstrip("0")
    else:
        game_date = "NOT FOUND"
        est_time = "NOT FOUND"

    # ----- Competitors -----
    comps = r.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
    away = next((c for c in comps if c["homeAway"] == "away"), None)
    home = next((c for c in comps if c["homeAway"] == "home"), None)

    away_team = away["team"]["displayName"] if away else "NOT FOUND"
    home_team = home["team"]["displayName"] if home else "NOT FOUND"

    # ----- Odds / Spread -----
    odds = r.get("pickcenter", [])
    spread = odds[0].get("spread", "NOT FOUND") if odds else "NOT FOUND"

    return away_team, home_team, game_date, est_time, spread

# ----- Main CSV Loop -----
with open(INPUT_CSV, "r") as infile, open(OUTPUT_CSV, "w", newline="") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(
        outfile,
        fieldnames=["game_id", "away_team", "home_team", "date", "start_time_est", "away_team_spread"]
    )
    writer.writeheader()

    for row in reader:
        game_id = row["game_id"]

        away_team, home_team, game_date, est_time, spread = get_game_info(game_id)

        writer.writerow({
            "game_id": game_id,
            "away_team": away_team,
            "home_team": home_team,
            "date": game_date,
            "start_time_est": est_time,
            "away_team_spread": spread
        })

        print(f"{game_id} | {away_team} @ {home_team} | {game_date} {est_time} | {spread}")
        time.sleep(0.3)
