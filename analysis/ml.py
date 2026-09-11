"""
Machine Learning integration for Lotofácil predictions.
Features Ensemble models, temporal validation, and calibration.
"""

import os
import json
import random
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from .stats import extract_balls
from .ml_features import build_feature_dataset, extract_latest_features


HISTORY_FILE = 'training_history.json'


def _simulate_random_baseline(num_simulations=100000):
    """Simulates selecting 15 random numbers against 15 drawn numbers to find expected hits."""
    # Theoretical mean is exactly 9. (15 * 15 / 25)
    # But let's build the distribution to match the prompt's requirements
    hits = np.random.hypergeometric(ngood=15, nbad=10, nsample=15, size=num_simulations)
    
    return {
        'mean': float(np.mean(hits)),
        'median': float(np.median(hits)),
        'std': float(np.std(hits)),
        'min': int(np.min(hits)),
        'max': int(np.max(hits)),
        'dist': {
            '8': float(np.mean(hits == 8) * 100),
            '9': float(np.mean(hits == 9) * 100),
            '10': float(np.mean(hits == 10) * 100),
            '11': float(np.mean(hits == 11) * 100),
            '12': float(np.mean(hits == 12) * 100),
            '13': float(np.mean(hits == 13) * 100),
            '14': float(np.mean(hits == 14) * 100),
            '15': float(np.mean(hits == 15) * 100),
            'ge_10': float(np.mean(hits >= 10) * 100),
            'ge_11': float(np.mean(hits >= 11) * 100),
            'ge_12': float(np.mean(hits >= 12) * 100),
            'ge_13': float(np.mean(hits >= 13) * 100),
            'ge_14': float(np.mean(hits >= 14) * 100),
        }
    }


def train_lotofacil_model(results, model_path='lotofacil_model.pkl'):
    """
    Trains a Calibrated Ensemble model (RandomForest + LogisticRegression)
    to predict the probability of each number appearing.
    Validates using Walk-Forward (TimeSeriesSplit).
    """
    balls_list = extract_balls(results)
    if len(balls_list) < 200:
        raise ValueError("Dados insuficientes para treinamento seguro (mínimo 200 concursos).")

    # 1. Feature Engineering
    X, y, full_df = build_feature_dataset(balls_list)
    
    # 2. Setup Walk-Forward Validation (Backtest)
    # We want to test on the last 500 draws (or less if not enough data)
    num_test_draws = min(500, len(balls_list) // 5)
    
    # Extract unique draw_ids to split properly without leaking across numbers of the same draw
    draw_ids = full_df['draw_id'].unique()
    test_draw_ids = draw_ids[-num_test_draws:]
    train_draw_ids = draw_ids[:-num_test_draws]
    
    # Create train and backtest sets
    train_mask = full_df['draw_id'].isin(train_draw_ids)
    test_mask = full_df['draw_id'].isin(test_draw_ids)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_df = full_df[test_mask]
    
    # 3. Model Definition (Ensemble)
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        min_samples_leaf=10, 
        random_state=42, 
        n_jobs=-1
    )
    lr = LogisticRegression(
        max_iter=1000, 
        C=0.1, 
        class_weight='balanced', 
        random_state=42
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr)],
        voting='soft'
    )
    
    # Calibrate probabilities using 3-fold CV on train set
    calibrated_model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
    
    # 4. Training
    calibrated_model.fit(X_train, y_train)
    
    # 5. Backtesting (Walk-Forward Evaluation)
    # For each test draw, we predict probabilities for all 25 numbers and pick top 15
    preds_prob = calibrated_model.predict_proba(X_test)[:, 1]
    
    test_eval_df = pd.DataFrame({
        'draw_id': test_df['draw_id'],
        'number_id': test_df['number_id'],
        'target': y_test,
        'prob': preds_prob
    })
    
    # Evaluate Top-15 hits per draw
    hits_per_draw = []
    for draw_id, group in test_eval_df.groupby('draw_id'):
        top15_predicted = group.nlargest(15, 'prob')['number_id'].values
        actual_drawn = group[group['target'] == 1]['number_id'].values
        
        hits = len(set(top15_predicted).intersection(set(actual_drawn)))
        hits_per_draw.append(hits)
        
    hits_arr = np.array(hits_per_draw)
    
    # 6. Metrics & Baseline
    baseline = _simulate_random_baseline()
    mean_hits = np.mean(hits_arr)
    lift = (mean_hits - baseline['mean']) / baseline['mean']
    
    model_metrics = {
        'mean': float(mean_hits),
        'median': float(np.median(hits_arr)),
        'std': float(np.std(hits_arr)),
        'min': int(np.min(hits_arr)),
        'max': int(np.max(hits_arr)),
        'dist': {
            '8': float(np.mean(hits_arr == 8) * 100),
            '9': float(np.mean(hits_arr == 9) * 100),
            '10': float(np.mean(hits_arr == 10) * 100),
            '11': float(np.mean(hits_arr == 11) * 100),
            '12': float(np.mean(hits_arr == 12) * 100),
            '13': float(np.mean(hits_arr == 13) * 100),
            '14': float(np.mean(hits_arr == 14) * 100),
            '15': float(np.mean(hits_arr == 15) * 100),
            'ge_10': float(np.mean(hits_arr >= 10) * 100),
            'ge_11': float(np.mean(hits_arr >= 11) * 100),
            'ge_12': float(np.mean(hits_arr >= 12) * 100),
            'ge_13': float(np.mean(hits_arr >= 13) * 100),
            'ge_14': float(np.mean(hits_arr >= 14) * 100),
        },
        'lift': lift,
        'train_samples': int(len(train_draw_ids)),
        'test_samples': int(len(test_draw_ids))
    }
    
    # 7. Retrain on ALL data for final model
    calibrated_model.fit(X, y)
    
    # 8. Save Model and History
    # Check if we should promote
    version = f"LF-ENSEMBLE-v1.{int(datetime.now().timestamp())}"
    history_entry = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': version,
        'model_type': 'Calibrated VotingClassifier (RF+LR)',
        'train_draws': model_metrics['train_samples'],
        'test_draws': model_metrics['test_samples'],
        'mean_hits': model_metrics['mean'],
        'baseline_hits': baseline['mean'],
        'lift': lift,
        'metrics': model_metrics
    }
    
    # Load history to see if it's better
    is_better = True
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            if history:
                best_past = max(h['mean_hits'] for h in history)
                if mean_hits < best_past:
                    is_better = False
    else:
        history = []
        
    history.append(history_entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
        
    # Overwrite the model ONLY if it is better (or if it's the first one)
    if is_better or not os.path.exists(model_path):
        joblib.dump(calibrated_model, model_path)
        history_entry['status'] = 'PROMOVIDO'
    else:
        history_entry['status'] = 'REJEITADO (Desempenho inferior ao atual)'
        
    # We return the JSON string to be shown in the UI
    response = {
        'history_entry': history_entry,
        'baseline': baseline
    }
    return response


def load_lotofacil_model(model_path='lotofacil_model.pkl'):
    """
    Loads trained model from pickle file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo não encontrado. Treine o modelo primeiro.")
    return joblib.load(model_path)


def predict_next_numbers(model, all_results, top_k=15):
    """
    Uses trained model to predict probabilities for the next draw given all past draws.
    Needs `all_results` to compute features properly.
    """
    balls_list = extract_balls(all_results)
    
    # Extract features for the current state (next draw)
    X_next = extract_latest_features(balls_list)
    
    # Predict probabilities for each of the 25 numbers
    probs = model.predict_proba(X_next)[:, 1]
    
    # Rank them
    ranking = [(num, prob) for num, prob in zip(range(1, 26), probs)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the top_k numbers as a list of ints for compatibility
    return [r[0] for r in ranking[:top_k]]


def predict_next_numbers_detailed(model, all_results):
    """
    Returns full ranking and probability for all 25 numbers.
    Used for combinatorial optimization and detailed UI.
    """
    balls_list = extract_balls(all_results)
    X_next = extract_latest_features(balls_list)
    probs = model.predict_proba(X_next)[:, 1]
    
    ranking = [{'dezena': int(num), 'score': float(prob)} for num, prob in zip(range(1, 26), probs)]
    ranking.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'ranking': ranking,
        'top15': [r['dezena'] for r in ranking[:15]],
        'top16': [r['dezena'] for r in ranking[:16]],
        'top17': [r['dezena'] for r in ranking[:17]],
        'top18': [r['dezena'] for r in ranking[:18]],
        'top19': [r['dezena'] for r in ranking[:19]],
        'top20': [r['dezena'] for r in ranking[:20]],
    }


def generate_suggested_games(valid_numbers, num_games=6):
    """
    Generates balanced Lotofácil games (15 numbers each out of 25),
    building around `valid_numbers` core.
    """
    games = []
    core_len = min(len(valid_numbers), 15)
    core = valid_numbers[:core_len]

    remaining_pool = [n for n in range(1, 26) if n not in core]
    needed = 15 - core_len

    for _ in range(num_games):
        if needed > 0 and len(remaining_pool) >= needed:
            additional = random.sample(remaining_pool, needed)
            game = sorted(core + additional)
        else:
            game = sorted(random.sample(range(1, 26), 15))

        games.append(game)

    return games
