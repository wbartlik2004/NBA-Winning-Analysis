import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BASE_DIR = "SortedGames"
TRAIN_SEASONS = ['2015-16', '2016-17', '2017-18', '2018-19', '2021-22', '2022-23']
TEST_SEASONS = ['2023-24', '2024-25']

# Actual NBA Regular Season Start Dates for Time Decay Calculation
SEASON_STARTS = {
    '2015-16': '2015-10-27', '2016-17': '2016-10-25', '2017-18': '2017-10-17',
    '2018-19': '2018-10-16', '2021-22': '2021-10-19', '2022-23': '2022-10-18',
    '2023-24': '2023-10-24', '2024-25': '2024-10-22'
}

# --- 1. DATA AGGREGATION ---
print("Aggregating data and standardizing timestamps...")
all_games = []

if not os.path.exists(BASE_DIR):
    print(f"Error: {BASE_DIR} folder not found.")
    exit()

for team_name in os.listdir(BASE_DIR):
    team_path = os.path.join(BASE_DIR, team_name)
    if os.path.isdir(team_path):
        for file in os.listdir(team_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(team_path, file))
                df['Team'] = team_name
                df['Season'] = file.replace('_Regular_Season.csv', '').replace('_', '-')
                all_games.append(df)

master_df = pd.concat(all_games)

# CRITICAL: Strip timezones immediately to avoid comparison errors later
master_df['Date'] = pd.to_datetime(master_df['Date']).dt.tz_localize(None)
master_df = master_df.sort_values(['Team', 'Date'])

# --- 2. FEATURE ENGINEERING ---
master_df['MOV'] = master_df['OurPoints'] - master_df['OppPoints']
master_df['Capped_MOV'] = master_df['MOV'].clip(-20, 20)
master_df['Win_Target'] = (master_df['WL?'] == 'W').astype(int)

# Create 20 Rolling Windows
feature_cols = []
for n in range(1, 21):
    col_name = f'avg_mov_{n}'
    master_df[col_name] = master_df.groupby('Team')['Capped_MOV'].transform(
        lambda x: x.rolling(window=n).mean().shift(1)
    )
    feature_cols.append(col_name)

# --- 3. CONTINUOUS TIME DECAY WEIGHTING ---
def get_time_decay_weight(row):
    season = row['Season']
    game_date = row['Date']
    
    if season in SEASON_STARTS:
        start_dt = pd.to_datetime(SEASON_STARTS[season])
        days_elapsed = (game_date - start_dt).days
        
        # Cap at 180 days (standard season length)
        days_elapsed = max(0, min(days_elapsed, 180))
        
        # Weight scales linearly from 1.0 (Day 0) to 2.0 (Day 180)
        return 1.0 + (days_elapsed / 180.0)
    return 1.0

master_df['sample_weight'] = master_df.apply(get_time_decay_weight, axis=1)

# --- 4. PREPARE SETS ---
final_df = master_df.dropna(subset=feature_cols)
train_df = final_df[final_df['Season'].isin(TRAIN_SEASONS)]
test_df = final_df[final_df['Season'].isin(TEST_SEASONS)]

# --- 5. TRAIN & EVALUATE ---
results = []
print(f"\n{'n':<4} | {'Log-Loss':<10} | {'Accuracy':<10} | {'Status'}")
print("-" * 42)

for n in range(1, 21):
    feature = f'avg_mov_{n}'
    
    # Using the n-window and HomeGame (if available)
    current_features = [feature]
    if 'HomeGame' in train_df.columns:
        current_features.append('HomeGame')

    X_train = train_df[current_features]
    y_train = train_df['Win_Target']
    w_train = train_df['sample_weight']
    
    X_test = test_df[current_features]
    y_test = test_df['Win_Target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with sample weights
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    probs = model.predict_proba(X_test_scaled)[:, 1]
    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    
    results.append({'n': n, 'Log_Loss': loss, 'Accuracy': acc})
    print(f"{n:<4} | {loss:<10.4f} | {acc:<10.4f} | Done ✓")

# --- 6. FINAL RANKING ---
print("\n" + "="*35)
print("  WEIGHTED MODEL RANKINGS ")
print("="*35)
results_df = pd.DataFrame(results).set_index('n')
print(results_df.sort_values('Log_Loss'))

best_n = results_df['Log_Loss'].idxmin()
print(f"\nWinning Model: Last {best_n} games (with Time Decay).")