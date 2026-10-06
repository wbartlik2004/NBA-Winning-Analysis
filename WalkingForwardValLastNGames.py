import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BASE_DIR = "SortedGames"
# Ordered list to ensure the walking-forward logic respects time
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
    
    # 1. Step Weights (Jan 1st and Post-ASB)
    if game_date.month in [1, 2, 3, 4, 5, 6]:
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
        days_elapsed = max(0, (game_date - start_dt).days)
        
        # Scale weight from 0.8 (Opening Night) to 1.0 (New Year's Eve)
        return 0.8 + (0.2 * (days_elapsed / total_early_days))

    return 1.0

# --- 1. DATA AGGREGATION ---
print("Aggregating league data from SortedGames...")
all_games = []
for team_name in os.listdir(BASE_DIR):
    team_path = os.path.join(BASE_DIR, team_name)
    if os.path.isdir(team_path):
        for file in os.listdir(team_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(team_path, file))
                df['Team'] = team_name
                # Normalize season name: 2015_16 -> 2015-16
                s_label = file.replace('_Regular_Season.csv', '').replace('_', '-')
                df['Season'] = s_label
                all_games.append(df)

master_df = pd.concat(all_games)
master_df['Date'] = pd.to_datetime(master_df['Date']).dt.tz_localize(None)
master_df = master_df.sort_values(['Team', 'Date'])

# --- 2. FEATURE ENGINEERING (WITH SEASONAL RESET) ---
master_df['Capped_MOV'] = (master_df['OurPoints'] - master_df['OppPoints']).clip(-20, 20)
master_df['Win_Target'] = (master_df['WL?'] == 'W').astype(int)

# Group by BOTH Team and Season to prevent lookback across off-seasons
master_df['MOV_Last_15_Games'] = master_df.groupby(['Team', 'Season'])['Capped_MOV'].transform(
    lambda x: x.shift(1).rolling(window=15).mean()
)

# Apply Weighting Logic
master_df['sample_weight'] = master_df.apply(get_hybrid_weight, axis=1)

# Drop rows where 15-game history doesn't exist yet (first 15 games of every season)
master_df = master_df.dropna(subset=['MOV_Last_15_Games'])

# --- 3. EXPANDING WINDOW VALIDATION ---
print(f"\n{'Test Season':<12} | {'Train Count':<12} | {'Log-Loss':<10} | {'Accuracy'}")
print("-" * 60)

seasonal_accuracies = []

for i in range(1, len(ALL_SEASONS)):
    train_seasons = ALL_SEASONS[:i]
    test_season = ALL_SEASONS[i]
    
    train_df = master_df[master_df['Season'].isin(train_seasons)]
    test_df = master_df[master_df['Season'] == test_season]
    
    if train_df.empty or test_df.empty:
        continue

    # Prepare features
    X_train = train_df[['MOV_Last_15_Games']]
    y_train = train_df['Win_Target']
    w_train = train_df['sample_weight']

    X_test = test_df[['MOV_Last_15_Games']]
    y_test = test_df['Win_Target']

    # Scale data (StandardScaler is standard practice in ISLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression with seasonal weights
    model = LogisticRegression().fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # Evaluate
    probs = model.predict_proba(X_test_scaled)[:, 1]
    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    
    print(f"{test_season:<12} | {len(train_df):<12} | {loss:<10.4f} | {acc:.2%}")
    seasonal_accuracies.append(acc)

print("-" * 60)
print(f"Mean Accuracy across all seasons: {np.mean(seasonal_accuracies):.2%}")