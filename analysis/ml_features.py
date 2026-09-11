"""
Machine Learning Feature Engineering for Lotofácil.
Ensures absolutely zero data leakage by computing features for draw N using ONLY draws up to N-1.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from .stats import PRIME_NUMBERS, FIBONACCI_NUMBERS, MULTIPLES_OF_3, MULTIPLES_OF_5


def build_feature_dataset(balls_list: List[List[int]]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transforms a raw list of historical draws (balls_list) into a dataset (X, y)
    for predicting the probability of each number (1-25) appearing in draw N,
    using only information from draws 0 to N-1.

    Each draw expands to 25 rows (one for each number).
    So N draws -> N * 25 rows.
    """
    # 1. Convert to binary matrix (shape: num_draws, 25)
    num_draws = len(balls_list)
    draws_bin = np.zeros((num_draws, 25), dtype=np.int8)
    for i, balls in enumerate(balls_list):
        for b in balls:
            if 1 <= b <= 25:
                draws_bin[i, b - 1] = 1

    df_bin = pd.DataFrame(draws_bin, columns=[f'n_{n:02d}' for n in range(1, 26)])

    # We cannot use data from row `i` to predict row `i`.
    # All features for row `i` must be built from `df_bin.iloc[:i]`.
    # A simple way is to build features for the whole DataFrame, and then SHIFT them by 1.

    # 2. Build local features per number (Freq, Delays, Lags)
    features_per_number = {}
    for num in range(1, 26):
        col = f'n_{num:02d}'
        s = df_bin[col]

        # Shifted series represents the history available AT the time of prediction
        # e.g. at draw 5, we only know up to draw 4. So we shift by 1.
        hist = s.shift(1)

        f_dict = {
            'target': s,  # The actual outcome we want to predict
            'lag_1': hist,
            'lag_2': s.shift(2),
            'lag_3': s.shift(3),
            'lag_4': s.shift(4),
            'lag_5': s.shift(5),
            # Frequencies
            'freq_5': hist.rolling(5, min_periods=1).mean(),
            'freq_10': hist.rolling(10, min_periods=1).mean(),
            'freq_20': hist.rolling(20, min_periods=1).mean(),
            'freq_50': hist.rolling(50, min_periods=1).mean(),
            'freq_100': hist.rolling(100, min_periods=1).mean(),
            'freq_all': hist.expanding(min_periods=1).mean(),
        }

        # Calculate current delay (how many draws since last 1)
        # Using a cumulative sum of 1s to group periods, then counting 0s
        # Cumsum increments every time the number hits
        hits = hist.fillna(0).astype(int)
        cumsum_hits = hits.cumsum()
        
        # We want to count consecutive zeros.
        # Group by the cumsum_hits, and within each group, cumulative count.
        # e.g. hits = [0, 0, 1, 0, 1, 0, 0]
        # cumsum = [0, 0, 1, 1, 2, 2, 2]
        # delay  = [1, 2, 0, 1, 0, 1, 2]
        delay = hist.groupby(cumsum_hits).cumcount()
        # when hist == 1, delay should be 0.
        delay = delay.where(hist == 0, 0)
        # However, for the very first group (cumsum=0), the delay starts at 1,2,3..
        f_dict['delay_current'] = delay

        # Moving Averages (Trend)
        f_dict['ma_short'] = f_dict['freq_10']
        f_dict['ma_long'] = f_dict['freq_50']
        f_dict['ma_diff'] = f_dict['ma_short'] - f_dict['ma_long']

        features_per_number[num] = pd.DataFrame(f_dict)

    # 3. Build global features of the past draws
    # Global features must also be shifted by 1 so we only use past info.
    past_bin = df_bin.shift(1)
    
    global_features = pd.DataFrame(index=df_bin.index)
    global_features['g_sum'] = past_bin.apply(lambda row: sum((i+1)*v for i, v in enumerate(row) if v==1), axis=1)
    global_features['g_evens'] = past_bin.apply(lambda row: sum(v for i, v in enumerate(row) if (i+1)%2==0 and v==1), axis=1)
    global_features['g_primes'] = past_bin.apply(lambda row: sum(v for i, v in enumerate(row) if (i+1) in PRIME_NUMBERS and v==1), axis=1)

    # Combine everything into a flat dataset where each row is (draw_id, number)
    all_rows = []
    
    # We skip the first few draws (e.g., skip first 10 draws) because they don't have enough history
    START_DRAW = 10
    
    for num in range(1, 26):
        df_num = features_per_number[num].copy()
        
        # Add global features
        for col in global_features.columns:
            df_num[col] = global_features[col]
            
        # Add static number features
        df_num['is_even'] = 1 if num % 2 == 0 else 0
        df_num['is_prime'] = 1 if num in PRIME_NUMBERS else 0
        df_num['is_fibonacci'] = 1 if num in FIBONACCI_NUMBERS else 0
        df_num['is_mult3'] = 1 if num in MULTIPLES_OF_3 else 0
        df_num['number_id'] = num
        df_num['draw_id'] = df_num.index
        
        # Drop the first draws where we have NaNs
        df_num = df_num.iloc[START_DRAW:].copy()
        all_rows.append(df_num)

    # Concatenate all numbers
    final_df = pd.concat(all_rows, ignore_index=True)
    
    # Sort by draw_id to maintain temporal order during cross validation
    final_df = final_df.sort_values(by=['draw_id', 'number_id']).reset_index(drop=True)
    
    # Fill any remaining NaNs (e.g., from rolling means with min_periods if there was absolutely no data, which shouldn't happen, but just in case)
    final_df = final_df.fillna(0)
    
    y = final_df['target']
    X = final_df.drop(columns=['target', 'draw_id'])
    
    # Assert no future leakage
    # We shouldn't use target in X
    assert 'target' not in X.columns
    
    return X, y, final_df


def extract_latest_features(balls_list: List[List[int]]) -> pd.DataFrame:
    """
    Given a history of draws, extract the features for the NEXT prediction (draw N+1).
    This computes the features using the entire balls_list as history, 
    so we don't shift by 1. The 'last row' of the computed features represents the current state.
    """
    # 1. Convert to binary matrix
    num_draws = len(balls_list)
    draws_bin = np.zeros((num_draws, 25), dtype=np.int8)
    for i, balls in enumerate(balls_list):
        for b in balls:
            if 1 <= b <= 25:
                draws_bin[i, b - 1] = 1

    df_bin = pd.DataFrame(draws_bin, columns=[f'n_{n:02d}' for n in range(1, 26)])

    # 2. Extract features based on ALL history (no shift, because we want to predict the next draw)
    features_per_number = {}
    for num in range(1, 26):
        col = f'n_{num:02d}'
        hist = df_bin[col]

        # Calculate delay
        hits = hist.fillna(0).astype(int)
        cumsum_hits = hits.cumsum()
        delay = hist.groupby(cumsum_hits).cumcount()
        delay = delay.where(hist == 0, 0)
        
        # We only care about the very last row (the state right now)
        f_dict = {
            'lag_1': hist.iloc[-1],
            'lag_2': hist.iloc[-2] if num_draws >= 2 else 0,
            'lag_3': hist.iloc[-3] if num_draws >= 3 else 0,
            'lag_4': hist.iloc[-4] if num_draws >= 4 else 0,
            'lag_5': hist.iloc[-5] if num_draws >= 5 else 0,
            'freq_5': hist.tail(5).mean(),
            'freq_10': hist.tail(10).mean(),
            'freq_20': hist.tail(20).mean(),
            'freq_50': hist.tail(50).mean(),
            'freq_100': hist.tail(100).mean(),
            'freq_all': hist.mean(),
            'delay_current': delay.iloc[-1],
        }
        
        f_dict['ma_short'] = f_dict['freq_10']
        f_dict['ma_long'] = f_dict['freq_50']
        f_dict['ma_diff'] = f_dict['ma_short'] - f_dict['ma_long']
        
        features_per_number[num] = f_dict
        
    # Global features (current state)
    last_row = df_bin.iloc[-1]
    g_sum = sum((i+1)*v for i, v in enumerate(last_row) if v==1)
    g_evens = sum(v for i, v in enumerate(last_row) if (i+1)%2==0 and v==1)
    g_primes = sum(v for i, v in enumerate(last_row) if (i+1) in PRIME_NUMBERS and v==1)
    
    rows = []
    for num in range(1, 26):
        d = features_per_number[num]
        d['g_sum'] = g_sum
        d['g_evens'] = g_evens
        d['g_primes'] = g_primes
        d['is_even'] = 1 if num % 2 == 0 else 0
        d['is_prime'] = 1 if num in PRIME_NUMBERS else 0
        d['is_fibonacci'] = 1 if num in FIBONACCI_NUMBERS else 0
        d['is_mult3'] = 1 if num in MULTIPLES_OF_3 else 0
        d['number_id'] = num
        
        rows.append(d)
        
    df_next = pd.DataFrame(rows)
    # Reorder columns to exactly match X from training
    expected_cols = [
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
        'freq_5', 'freq_10', 'freq_20', 'freq_50', 'freq_100', 'freq_all', 
        'delay_current', 'ma_short', 'ma_long', 'ma_diff', 
        'g_sum', 'g_evens', 'g_primes', 
        'is_even', 'is_prime', 'is_fibonacci', 'is_mult3', 'number_id'
    ]
    return df_next[expected_cols]
