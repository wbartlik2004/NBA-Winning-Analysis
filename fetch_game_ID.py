import requests
import csv
import datetime
import time

def fetch_games_for_date(date_str):
    """
    Fetch games from ESPN scoreboard API for a given date (YYYYMMDD).
    Returns list of tuples (game_id, away_team, home_team).
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    params = {
        "dates": date_str
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    games = []
    for event in data.get("events", []):
        game_id = event.get("id")
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        teams = comp.get("competitors", [])
        away = home = None
        for t in teams:
            if t.get("homeAway") == "away":
                away = t["team"]["displayName"]
            elif t.get("homeAway") == "home":
                home = t["team"]["displayName"]
        if game_id and away and home:
            games.append((game_id, away, home))
    return games

def iterate_season(start_date, end_date, output_csv="nba_games_ids_teams.csv"):
    """
    Iterate from start_date to end_date (datetime.date objects),
    fetch games each day and write to CSV.
    """
    current = start_date
    all_rows = []
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        try:
            games = fetch_games_for_date(date_str)
            print(f"{date_str}: found {len(games)} games")
            all_rows.extend([(date_str,) + g for g in games])
        except Exception as e:
            print(f"Error on {date_str}: {e}")
        current += datetime.timedelta(days=1)
        time.sleep(1)  # polite rate‑limit pause
    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf‑8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "game_id", "away_team", "home_team"])
        writer.writerows(all_rows)
    print(f"Finished. Wrote {len(all_rows)} rows to {output_csv}")

if __name__ == "__main__":
    season_start = datetime.date(2025, 10, 21)  
    today = datetime.date.today()
    iterate_season(season_start, today)
