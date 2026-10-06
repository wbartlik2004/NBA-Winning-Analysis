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

SEASON_STARTS = {'2015-16': '2015-10-27', '2016-17': '2016-10-25', '2017-18': '2017-10-17', '2018-19': '2018-10-16', '2021-22': '2021-10-19', '2022-23': '2022-10-18', '2023-24': '2023-10-24', '2024-25': '2024-10-22'}
ASG_DATES = {'2015-16': '2016-02-14', '2016-17': '2017-02-19', '2017-18': '2018-02-18', '2018-19': '2019-02-17', '2021-22': '2022-02-20', '2022-23': '2023-02-19', '2023-24': '2024-02-18', '2024-25': '2025-02-16'}

# --- 1. DATA AGGREGATION ---
print("Aggregating data...")
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

# --- 2. FEATURE ENGINEERING (DAY-BASED ROLLING) ---
master_df['Capped_MOV'] = (master_df['OurPoints'] - master_df['OppPoints']).clip(-20, 20)
master_df['Win_Target'] = (master_df['WL?'] == 'W').astype(int)
master_df['HomeGame'] = (master_df['Location'] == 'Home').astype(int)

# Day-based rolling requires the Date to be the index
master_df = master_df.set_index('Date', drop=False)

feature_cols = []
for d in range(7, 22): # 7 to 21 days
    col_name = f'avg_mov_{d}d'
    # .rolling('7D') looks back at the last 7 days of dates
    master_df[col_name] = master_df.groupby('Team')['Capped_MOV'].transform(
        lambda x: x.rolling(window=f'{d}D').mean().shift(1, freq=f'{d}D')
    )
    # Note: Using shift with freq is safer for time-series rolling
    master_df[col_name] = master_df.groupby('Team')['Capped_MOV'].transform(
        lambda x: x.rolling(window=f'{d}D').mean().shift(1)
    )
    feature_cols.append(col_name)

# --- 3. HYBRID WEIGHTING ---
def get_hybrid_weight(row):
    game_date = row['Date']
    season = row['Season']
    if game_date.month in [1, 2, 3, 4, 5, 6]:
        if season in ASG_DATES and game_date > pd.to_datetime(ASG_DATES[season]):
            return 1.0
        return 1.0
    if season in SEASON_STARTS:
        start_dt = pd.to_datetime(SEASON_STARTS[season])
        total_early_days = (pd.to_datetime(f"{start_dt.year + 1}-01-01") - start_dt).days
        days_elapsed = max(0, (game_date - start_dt).days)
        return 1 + (0 * (days_elapsed / total_early_days))
    return 1.0

master_df['sample_weight'] = master_df.apply(get_hybrid_weight, axis=1)

# --- 4. TRAIN & EVALUATE ---
final_df = master_df.dropna(subset=feature_cols)
train_df = final_df[final_df['Season'].isin(TRAIN_SEASONS)]
test_df = final_df[final_df['Season'].isin(TEST_SEASONS)]

results = []
print(f"\n{'Days':<5} | {'Log-Loss':<10} | {'Accuracy':<10}")
print("-" * 35)

for d in range(7, 22):
    feature = f'avg_mov_{d}d'
    X_train, y_train, w_train = train_df[[feature, 'HomeGame']], train_df['Win_Target'], train_df['sample_weight']
    X_test, y_test = test_df[[feature, 'HomeGame']], test_df['Win_Target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression().fit(X_train_scaled, y_train, sample_weight=w_train)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    results.append({'days': d, 'Log_Loss': loss, 'Accuracy': acc})
    print(f"{d:<5} | {loss:<10.4f} | {acc:<10.4f} | Done")

# --- 5. RANKING ---
results_df = pd.DataFrame(results).set_index('days')
print("\n" + "="*40)
print("  RANKING BY LOG-LOSS (Time Windows) ")
print("="*40)
print(results_df.sort_values('Log_Loss'))
print(average_log_loss := results_df['Log_Loss'].mean())