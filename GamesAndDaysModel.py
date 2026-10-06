import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BASE_DIR = "SortedGames"
ALL_SEASONS = ['2015-16', '2016-17', '2017-18', '2018-19', '2021-22', '2022-23', '2023-24', '2024-25']

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

# --- HYBRID WEIGHTING FUNCTION ---
def get_hybrid_weight(row):
    game_date = row['Date']
    season = row['Season']
    
    if game_date.month in [1, 2, 3, 4, 5, 6]:
        if season in ASG_DATES:
            asg_dt = pd.to_datetime(ASG_DATES[season])
            if game_date > asg_dt:
                return 1.5 
        return 1.2
    
    if season in SEASON_STARTS:
        start_dt = pd.to_datetime(SEASON_STARTS[season])
        jan_1st = pd.to_datetime(f"{start_dt.year + 1}-01-01")
        total_early_days = (jan_1st - start_dt).days
        days_elapsed = max(0, (game_date - start_dt).days)
        return 0.8 + (0.2 * (days_elapsed / total_early_days))

    return 1.0

# --- 1. DATA AGGREGATION ---
print("Aggregating league data...")
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
master_df['Date'] = pd.to_datetime(master_df['Date']).dt.tz_localize(None)
master_df = master_df.sort_values(['Team', 'Date'])

# --- 2. FEATURE ENGINEERING ---
master_df['Capped_MOV'] = (master_df['OurPoints'] - master_df['OppPoints']).clip(-20, 20)
master_df['Win_Target'] = (master_df['WL?'] == 'W').astype(int)

# Feature 1: Performance (15-Game Window with Seasonal Reset)
master_df['MOV_Last_15_Games'] = master_df.groupby(['Team', 'Season'])['Capped_MOV'].transform(
    lambda x: x.shift(1).rolling(window=20).mean()
)

# Feature 2: Fatigue (Games in last 15 Days)
master_df['Fatigue_15D'] = master_df.groupby('Team').apply(
    lambda x: x.rolling('7D', on='Date').count()['Win_Target'].shift(1),
    include_groups=False 
).reset_index(level=0, drop=True).fillna(0)

# Weights calculation
master_df['sample_weight'] = master_df.apply(get_hybrid_weight, axis=1)

# Drop rows where we don't have enough history yet
final_df = master_df.dropna(subset=['MOV_Last_15_Games'])

# --- 3. EXPANDING WINDOW VALIDATION ---
print(f"\n{'Test Season':<12} | {'Log-Loss':<10} | {'Accuracy':<10} | {'Fatigue Coeff'}")
print("-" * 65)

seasonal_accuracies = []

for i in range(1, len(ALL_SEASONS)):
    train_df = final_df[final_df['Season'].isin(ALL_SEASONS[:i])]
    test_df = final_df[final_df['Season'] == ALL_SEASONS[i]]
    
    if train_df.empty or test_df.empty: continue

    features = ['MOV_Last_15_Games', 'Fatigue_15D']
    X_train, y_train, w_train = train_df[features], train_df['Win_Target'], train_df['sample_weight']
    X_test, y_test = test_df[features], test_df['Win_Target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression().fit(X_train_scaled, y_train, sample_weight=w_train)
    
    probs = model.predict_proba(X_test_scaled)[:, 1]
    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    fatigue_b = model.coef_[0][1]
    
    print(f"{ALL_SEASONS[i]:<12} | {loss:<10.4f} | {acc:<10.2%} | {fatigue_b:.4f}")
    seasonal_accuracies.append(acc)

# --- 4. FINAL AGGREGATE METRIC ---
if seasonal_accuracies:
    print("-" * 65)
    print(f"OVERALL MEAN ACCURACY: {np.mean(seasonal_accuracies):.2%}")
    print("-" * 65)