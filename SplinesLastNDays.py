import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# --- 1. CONFIGURATION ---
BASE_DIR = "SortedGames"
TRAIN_SEASONS = ['2015-16', '2016-17', '2017-18', '2018-19', '2021-22', '2022-23']
TEST_SEASONS = ['2023-24', '2024-25']

# Critical dates for weighting logic
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

# --- 2. DATA AGGREGATION & CLEANING ---
print("Aggregating Team CSVs...")
all_games = []

if not os.path.exists(BASE_DIR):
    print(f"Error: {BASE_DIR} not found. Please run your scraper first.")
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
master_df['Date'] = pd.to_datetime(master_df['Date']).dt.tz_localize(None)
master_df = master_df.sort_values(['Team', 'Date'])

# --- 3. FEATURE ENGINEERING ---
# Define Target and Core Features
master_df['Capped_MOV'] = (master_df['OurPoints'] - master_df['OppPoints']).clip(-20, 20)
master_df['Win_Target'] = (master_df['WL?'] == 'W').astype(int)
master_df['HomeGame'] = (master_df['Location'] == 'Home').astype(int)

# Set Date index for time-based rolling windows
master_df = master_df.set_index('Date', drop=False)

print("Calculating 7-21 Day Time Windows...")
feature_cols = []
for d in range(7, 22):
    col_name = f'avg_mov_{d}d'
    # Rolling window looks back 'd' days. shift(1) ensures we don't include today's result.
    master_df[col_name] = master_df.groupby('Team')['Capped_MOV'].transform(
        lambda x: x.rolling(window=f'{d}D').mean().shift(1)
    )
    feature_cols.append(col_name)

# --- 4. HYBRID WEIGHTING LOGIC ---
def get_hybrid_weight(row):
    game_date = row['Date']
    season = row['Season']
    
    # Late Season Plateau Weights
    if game_date.month in [1, 2, 3, 4, 5, 6]:
        if season in ASG_DATES:
            asg_dt = pd.to_datetime(ASG_DATES[season])
            if game_date > asg_dt:
                return 4.0  # Post All-Star
        return 3.0  # Jan 1st to All-Star
    
    # Early Season Linear Decay (Oct - Dec)
    if season in SEASON_STARTS:
        start_dt = pd.to_datetime(SEASON_STARTS[season])
        jan_1st = pd.to_datetime(f"{start_dt.year + 1}-01-01")
        total_early_days = (jan_1st - start_dt).days
        days_elapsed = max(0, (game_date - start_dt).days)
        # Scale 0.5 to 1.0
        return 0.5 + (0.5 * (days_elapsed / total_early_days))
    
    return 1.0

master_df['sample_weight'] = master_df.apply(get_hybrid_weight, axis=1)

# Drop rows missing the rolling features (start of seasons)
final_df = master_df.dropna(subset=feature_cols)

# --- 5. MODEL TRAINING LOOP (NON-LINEAR POLYNOMIAL) ---
results = []
print(f"\n{'Days':<5} | {'Log-Loss':<10} | {'Accuracy':<10}")
print("-" * 35)

for d in range(7, 22):
    feature = f'avg_mov_{d}d'
    
    # Filter Train/Test sets
    train_df = final_df[final_df['Season'].isin(TRAIN_SEASONS)]
    test_df = final_df[final_df['Season'].isin(TEST_SEASONS)]

    X_train_raw = train_df[[feature, 'HomeGame']]
    y_train = train_df['Win_Target']
    w_train = train_df['sample_weight']
    
    X_test_raw = test_df[[feature, 'HomeGame']]
    y_test = test_df['Win_Target']

    # Apply Polynomial Bending (Degree 3)
    # This turns [MOV, Home] into [MOV, Home, MOV^2, MOV*Home, Home^2, MOV^3, ...]
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_raw)
    X_test_poly = poly.transform(X_test_raw)

    # Scale for Logistic Regression Stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # Evaluate
    probs = model.predict_proba(X_test_scaled)[:, 1]
    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    
    results.append({'days': d, 'Log_Loss': loss, 'Accuracy': acc})
    print(f"{d:<5} | {loss:<10.4f} | {acc:<10.4f} | Done ✓")

# --- 6. FINAL RESULTS & RANKING ---
results_df = pd.DataFrame(results).set_index('days')
print("\n" + "="*45)
print("  FINAL RANKING: POLYNOMIAL LOGISTIC MODEL ")
print("="*45)
print(results_df.sort_values('Log_Loss'))

best_days = results_df['Log_Loss'].idxmin()
print(f"\nWINNING MODEL: {best_days}-Day Time Window")
print(f"Minimum Log-Loss achieved: {results_df.loc[best_days, 'Log_Loss']:.4f}")