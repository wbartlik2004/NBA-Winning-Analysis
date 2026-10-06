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

SEASON_STARTS = {
    '2015-16': '2015-10-27', '2016-17': '2016-10-25', '2017-18': '2017-10-17',
    '2018-19': '2018-10-16', '2021-22': '2021-10-19', '2022-23': '2022-10-18',
    '2023-24': '2023-10-24', '2024-25': '2024-10-22'
}

ASG_DATES = {
    '2015-16': '2016-02-14', '2016-17': '2017-02-19', '2017-18': '2018-02-18',
    '2018-19': '2019-02-17', '2021-22': '2022-02-20', '2022-23': '2023-02-19',
    '2023-24': '2024-02-18', '2024-25': '2025-02-16'
}

# --- 1. DATA AGGREGATION ---
all_games = []
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
# Standardize dates immediately
master_df['Date'] = pd.to_datetime(master_df['Date']).dt.tz_localize(None)
master_df = master_df.sort_values(['Team', 'Date'])

# --- 2. FEATURE ENGINEERING ---
master_df['Capped_MOV'] = (master_df['OurPoints'] - master_df['OppPoints']).clip(-20, 20)
master_df['Win_Target'] = (master_df['WL?'] == 'W').astype(int)

feature_cols = []
for n in range(1, 21):
    col = f'avg_mov_{n}'
    master_df[col] = master_df.groupby('Team')['Capped_MOV'].transform(lambda x: x.rolling(n).mean().shift(1))
    feature_cols.append(col)

# --- 3. HYBRID WEIGHTING LOGIC ---
def get_hybrid_weight(row):
    game_date = row['Date']
    season = row['Season']
    
    # 1. Step Weights (Jan 1st and Post-ASB)
    if game_date.month in [1, 2, 3, 4, 5, 6]:
        # Check if it's also after the All-Star Game
        if season in ASG_DATES:
            asg_dt = pd.to_datetime(ASG_DATES[season])
            if game_date > asg_dt:
                return 1.5 # Post All-Star
        return 1.2 # Jan 1st to All-Star
    
    # 2. Linear Decay for Early Season (Oct - Dec)
    if season in SEASON_STARTS:
        start_dt = pd.to_datetime(SEASON_STARTS[season])
        jan_1st = pd.to_datetime(f"{start_dt.year + 1}-01-01")
        
        total_early_days = (jan_1st - start_dt).days
        days_elapsed = (game_date - start_dt).days
        days_elapsed = max(0, days_elapsed)
        
        # Scale weight from 0.5 (Opening Night) to 1.0 (New Year's Eve)
        decay_weight = 0.8 + (0.2 * (days_elapsed / total_early_days))
        return decay_weight

    return 1.0

master_df['sample_weight'] = master_df.apply(get_hybrid_weight, axis=1)

# --- 4. PREPARE SETS ---
final_df = master_df.dropna(subset=feature_cols)
train_df = final_df[final_df['Season'].isin(TRAIN_SEASONS)]
test_df = final_df[final_df['Season'].isin(TEST_SEASONS)]

# --- 5. TRAIN & EVALUATE ---
results = []
print(f"\n{'n':<4} | {'Log-Loss':<10} | {'Accuracy':<10}")
print("-" * 35)

for n in range(1, 21):
    feature = f'avg_mov_{n}'
    features = [feature]
    if 'HomeGame' in train_df.columns: features.append('HomeGame')

    X_train, y_train, w_train = train_df[features], train_df['Win_Target'], train_df['sample_weight']
    X_test, y_test = test_df[features], test_df['Win_Target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression().fit(X_train_scaled, y_train, sample_weight=w_train)
    
    probs = model.predict_proba(X_test_scaled)[:, 1]
    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    
    results.append({'n': n, 'Log_Loss': loss, 'Accuracy': acc})
    print(f"{n:<4} | {loss:<10.4f} | {acc:<10.4f} | Done ✓")

# --- 6. FINAL RANKING ---
results_df = pd.DataFrame(results).set_index('n')
print("\n" + "="*40)
print("  HYBRID DECAY + STEP RANKINGS ")
print("="*40)
print(results_df.sort_values('Log_Loss'))