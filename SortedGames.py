import requests
import csv
import datetime
import time
import os

# Configuration
SEASONS = [
    (2015, 10, 2016, 6), (2016, 10, 2017, 6), (2017, 10, 2018, 6), 
    (2018, 10, 2019, 6), (2021, 10, 2022, 6), (2022, 10, 2023, 6), 
    (2023, 10, 2024, 6), (2024, 10, 2025, 6)
]
BASE_DIR = "SortedGames"

def fetch_and_process_games(date_str):
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    resp = requests.get(url, params={"dates": date_str})
    resp.raise_for_status()
    data = resp.json()
    
    games_data = []
    for event in data.get("events", []):
        if event.get("season", {}).get("type") != 2:
            continue
            
        comp = event.get("competitions", [])[0]
        date_time = event.get("date")
        
        home = next(t for t in comp["competitors"] if t["homeAway"] == "home")
        away = next(t for t in comp["competitors"] if t["homeAway"] == "away")
        
        game_info = {
            "date": date_time,
            "home_name": home["team"]["displayName"],
            "away_name": away["team"]["displayName"],
            "home_pts": home.get("score"),
            "away_pts": away.get("score"),
            "home_win": home.get("winner")
        }
        games_data.append(game_info)
    return games_data

def save_game_to_folders(game, season_str):
    for role in ["home", "away"]:
        team_name = game[f"{role}_name"].replace(" ", "_")
        opp_name = game["away_name"] if role == "home" else game["home_name"]
        
        # New Column Value: Capitalize the role (Home/Away)
        location = role.capitalize() 
        
        our_pts = game[f"{role}_pts"]
        opp_pts = game["away_pts"] if role == "home" else game["home_pts"]
        
        # Determine W/L
        if game["home_win"] is True:
            wl = "W" if role == "home" else "L"
        else:
            wl = "L" if role == "home" else "W"

        team_dir = os.path.join(BASE_DIR, team_name)
        os.makedirs(team_dir, exist_ok=True)
        
        file_path = os.path.join(team_dir, f"{season_str.replace('-', '_')}_Regular_Season.csv")
        
        file_exists = os.path.isfile(file_path)
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # UPDATED HEADER: Added 'Location'
            if not file_exists:
                writer.writerow(["Date", "Against Who?", "Location", "WL?", "OurPoints", "OppPoints"])
            
            # UPDATED ROW: Added 'location'
            writer.writerow([game["date"], opp_name, location, wl, our_pts, opp_pts])

if __name__ == "__main__":
    for s_year, s_month, e_year, e_month in SEASONS:
        start_date = datetime.date(s_year, s_month, 1)
        # Handle month end safely
        end_date = datetime.date(e_year, e_month, 28) + datetime.timedelta(days=4)
        end_date = end_date - datetime.timedelta(days=end_date.day)
        
        season_label = f"{s_year}-{str(e_year)[2:]}"
        print(f"Starting Season: {season_label}")
        
        current = start_date
        while current <= end_date and current <= datetime.date.today():
            date_str = current.strftime("%Y%m%d")
            try:
                games = fetch_and_process_games(date_str)
                for g in games:
                    save_game_to_folders(g, season_label)
            except Exception as e:
                print(f"Error on {date_str}: {e}")
            
            current += datetime.timedelta(days=1)
            time.sleep(0.1)