import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. AGGREGATE DATA
BASE_DIR = "SortedGames"
all_games = []

for team_name in os.listdir(BASE_DIR):
    team_path = os.path.join(BASE_DIR, team_name)
    if os.path.isdir(team_path):
        for file in os.listdir(team_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(team_path, file))
                # Add metadata from folder/file names
                df['Team'] = team_name
                df['Season'] = file.replace('_Regular_Season.csv', '').replace('_', '-')
                all_games.append(df)

master_df = pd.concat(all_games)
master_df['Date'] = pd.to_datetime(master_df['Date'])
master_df = master_df.sort_values(['Team', 'Date'])

# 2. FEATURE ENGINEERING (CAPPED MOV)
master_df['MOV'] = master_df['OurPoints'] - master_df['OppPoints']
master_df['Capped_MOV'] = master_df['MOV'].clip(-20, 20)
master_df['Win_Target'] = (master_df['WL?'] == 'W').astype(int)

# Create 20 rolling windows
feature_cols = []
for n in range(1, 21):
    col_name = f'avg_mov_{n}'
    master_df[col_name] = master_df.groupby('Team')['Capped_MOV'].transform(
        lambda x: x.rolling(window=n).mean().shift(1)
    )
    feature_cols.append(col_name)

# 3. SPLIT DATA BY SEASON
train_seasons = ['2015-16', '2016-17', '2017-18', '2018-19', '2021-22', '2022-23']
test_seasons = ['2023-24', '2024-25']

# Drop NaNs (rows where we don't have enough history for the 20-game window)
final_df = master_df.dropna(subset=feature_cols)

train_df = final_df[final_df['Season'].isin(train_seasons)]
test_df = final_df[final_df['Season'].isin(test_seasons)]

# 4. TRAIN AND EVALUATE 20 MODELS
results = []

print(f"{'n':<4} | {'Log-Loss':<10} | {'Accuracy':<10} | {'Status'}")
print("-" * 40)

for n in range(1, 21):
    feature = f'avg_mov_{n}'
    
    # Data Setup
    X_train = train_df[[feature]]
    y_train = train_df['Win_Target']
    X_test = test_df[[feature]]
    y_test = test_df['Win_Target']
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    probs = model.predict_proba(X_test_scaled)[:, 1]
    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    
    # Append to results list
    results.append({
        'n': n,
        'Log_Loss': round(loss, 4),
        'Accuracy': round(acc, 4)
    })
    
    # Real-time printout
    print(f"{n:<4} | {loss:<10.4f} | {acc:<10.4f} | Done ✓")

# Final Summary Table
print("\n" + "="*30)
print(" FINAL MODEL RANKINGS ")
print("="*30)
results_df = pd.DataFrame(results).set_index('n')
print(results_df.sort_values('Log_Loss')) # Shows best models at the top

best_n = results_df['Log_Loss'].idxmin()
print(f"\nWinner: n={best_n} is your most significant predictive window.")