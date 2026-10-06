import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = "SortedGames"
SEASONS = ['2015_16', '2016_17', '2017_18', '2018_19', '2021_22', '2022_23', '2023_24', '2024_25']

def prepare_league_data():
    all_rows = []
    
    # 1. Load Data
    for team in os.listdir(BASE_DIR):
        team_path = os.path.join(BASE_DIR, team)
        if not os.path.isdir(team_path): continue
        
        for s in SEASONS:
            f_path = os.path.join(team_path, f"{s}_Regular_Season.csv")
            if os.path.exists(f_path):
                df = pd.read_csv(f_path)
                df['Season'] = s
                df['Team'] = team
                all_rows.append(df)
    
    full_df = pd.concat(all_rows).reset_index(drop=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df = full_df.sort_values(['Team', 'Date'])
    
    # 2. Convert W/L to Binary (Target Variable)
    full_df['Target'] = full_df['WL?'].map({'W': 1, 'L': 0})
    
    # 3. Feature Engineering: Fatigue (Games in last 15 Days)
    # We use a 15-day rolling count, shifting by 1 so we don't count the current game
    full_df['Fatigue_15D'] = full_df.groupby('Team').apply(
        lambda x: x.rolling('15D', on='Date').count()['Target'].shift(1)
    ).reset_index(level=0, drop=True).fillna(0)
    
    # 4. Feature Engineering: Performance (Last 10 Games Win %)
    full_df['WinRate_10G'] = full_df.groupby('Team')['Target'].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )
    
    return full_df.dropna()

def run_seasonal_validation(df):
    features = ['Fatigue_15D', 'WinRate_10G']
    results = []

    print(f"{'Test Season':<15} | {'Accuracy':<10} | {'Fatigue Coeff':<15} | {'WinRate Coeff'}")
    print("-" * 70)

    # Expanding Window: Train on all seasons prior to 'i', test on season 'i'
    for i in range(1, len(SEASONS)):
        train_seasons = SEASONS[:i]
        test_season = SEASONS[i]
        
        train_data = df[df['Season'].isin(train_seasons)]
        test_data = df[df['Season'] == test_season]
        
        if train_data.empty or test_data.empty: continue

        # Model: Logistic Regression (ISLP Chapter 4)
        model = LogisticRegression()
        model.fit(train_data[features], train_data['Target'])
        
        # Evaluation
        preds = model.predict(test_data[features])
        acc = accuracy_score(test_data['Target'], preds)
        coeffs = model.coef_[0]
        
        print(f"{test_season:<15} | {acc:<10.2%} | {coeffs[0]:<15.4f} | {coeffs[1]:.4f}")
        
        results.append({
            'Season': test_season,
            'Accuracy': acc,
            'Fatigue_Weight': coeffs[0],
            'Performance_Weight': coeffs[1]
        })
    
    return results

if __name__ == "__main__":
    print("Gathering league data and calculating features...")
    master_df = prepare_league_data()
    
    print("\nStarting Walking Forward Seasonal Validation...\n")
    final_results = run_seasonal_validation(master_df)