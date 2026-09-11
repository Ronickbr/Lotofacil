"""
Machine Learning integration for Lotofácil predictions.
Features Ensemble models, temporal validation, calibration and safe fallbacks.
"""

import os
import json
import random
import tempfile
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from .stats import extract_balls
from .ml_features import build_feature_dataset, extract_latest_features


HISTORY_FILE = 'training_history.json'


def _simulate_random_baseline(num_simulations=50000):
    """Simula uma seleção aleatória de 15 dezenas contra um sorteio de 15 dezenas."""
    hits = np.random.hypergeometric(
        ngood=15,
        nbad=10,
        nsample=15,
        size=num_simulations,
    )
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
        },
    }


def _load_training_history():
    """Carrega histórico tolerando arquivo inexistente, vazio ou legado/corrompido."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []


def _save_model_atomic(model, model_path):
    """Salva o modelo sem deixar um pickle parcial em caso de interrupção."""
    model_dir = os.path.dirname(os.path.abspath(model_path)) or '.'
    fd, tmp_path = tempfile.mkstemp(prefix='lotofacil_model_', suffix='.pkl', dir=model_dir)
    os.close(fd)
    try:
        joblib.dump(model, tmp_path)
        os.replace(tmp_path, model_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _frequency_fallback(all_results):
    """Ranking estatístico seguro usado quando o modelo treinado está indisponível/incompatível."""
    balls_list = extract_balls(all_results)
    if not balls_list:
        return [(n, 0.0) for n in range(1, 26)]

    recent = balls_list[-50:]
    full = balls_list
    scores = []
    for n in range(1, 26):
        recent_freq = sum(n in draw for draw in recent) / max(len(recent), 1)
        full_freq = sum(n in draw for draw in full) / max(len(full), 1)
        score = 0.7 * recent_freq + 0.3 * full_freq
        scores.append((n, float(score)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def train_lotofacil_model(results, model_path='lotofacil_model.pkl'):
    """
    Treina um ensemble calibrado (RandomForest + LogisticRegression).
    Mantém validação temporal e limita paralelismo para reduzir risco de travamento local.
    """
    balls_list = extract_balls(results)
    if len(balls_list) < 200:
        raise ValueError('Dados insuficientes para treinamento seguro (mínimo 200 concursos).')

    X, y, full_df = build_feature_dataset(balls_list)

    draw_ids = full_df['draw_id'].unique()
    num_test_draws = min(200, max(20, len(draw_ids) // 5))
    if len(draw_ids) <= num_test_draws:
        raise ValueError('Quantidade de concursos insuficiente após a preparação das features.')

    test_draw_ids = draw_ids[-num_test_draws:]
    train_draw_ids = draw_ids[:-num_test_draws]

    train_mask = full_df['draw_id'].isin(train_draw_ids)
    test_mask = full_df['draw_id'].isin(test_draw_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_df = full_df[test_mask]

    rf = RandomForestClassifier(
        n_estimators=60,
        max_depth=7,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=1,
    )
    lr = LogisticRegression(
        max_iter=700,
        C=0.1,
        class_weight='balanced',
        random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr)],
        voting='soft',
        n_jobs=1,
    )
    calibrated_model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)

    calibrated_model.fit(X_train, y_train)

    preds_prob = calibrated_model.predict_proba(X_test)[:, 1]
    test_eval_df = pd.DataFrame({
        'draw_id': test_df['draw_id'].to_numpy(),
        'number_id': test_df['number_id'].to_numpy(),
        'target': np.asarray(y_test),
        'prob': preds_prob,
    })

    hits_per_draw = []
    for _, group in test_eval_df.groupby('draw_id'):
        top15_predicted = group.nlargest(15, 'prob')['number_id'].values
        actual_drawn = group[group['target'] == 1]['number_id'].values
        hits_per_draw.append(len(set(top15_predicted).intersection(set(actual_drawn))))

    if not hits_per_draw:
        raise ValueError('O backtest não produziu amostras válidas.')

    hits_arr = np.array(hits_per_draw)
    baseline = _simulate_random_baseline()
    mean_hits = float(np.mean(hits_arr))
    lift = float((mean_hits - baseline['mean']) / baseline['mean'])

    model_metrics = {
        'mean': mean_hits,
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
        'test_samples': int(len(test_draw_ids)),
    }

    # Treino final usando todos os dados.
    calibrated_model.fit(X, y)

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
        'metrics': model_metrics,
    }

    history = _load_training_history()
    past_scores = [h.get('mean_hits') for h in history if isinstance(h, dict) and isinstance(h.get('mean_hits'), (int, float))]
    best_past = max(past_scores) if past_scores else None
    is_better = best_past is None or mean_hits >= best_past

    if is_better or not os.path.exists(model_path):
        _save_model_atomic(calibrated_model, model_path)
        history_entry['status'] = 'PROMOVIDO'
    else:
        history_entry['status'] = 'REJEITADO (Desempenho inferior ao atual)'

    history.append(history_entry)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

    return {
        'history_entry': history_entry,
        'baseline': baseline,
    }


def load_lotofacil_model(model_path='lotofacil_model.pkl'):
    """Carrega o modelo e converte incompatibilidade/corrupção em erro tratável pela interface."""
    if not os.path.exists(model_path):
        raise FileNotFoundError('Modelo não encontrado. Treine o modelo primeiro.')
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise FileNotFoundError(
            'O modelo salvo está incompatível ou corrompido. Execute um novo treinamento em IA / Modelo.'
        ) from exc


def predict_next_numbers(model, all_results, top_k=15):
    """Retorna as dezenas mais prováveis; usa fallback estatístico se o modelo falhar."""
    balls_list = extract_balls(all_results)
    if not balls_list:
        return []

    try:
        X_next = extract_latest_features(balls_list)
        probs = model.predict_proba(X_next)[:, 1]
        ranking = [(num, float(prob)) for num, prob in zip(range(1, 26), probs)]
        ranking.sort(key=lambda x: x[1], reverse=True)
    except Exception:
        ranking = _frequency_fallback(all_results)

    return [r[0] for r in ranking[:top_k]]


def predict_next_numbers_detailed(model, all_results):
    """Retorna ranking completo; usa fallback estatístico se o modelo não puder inferir."""
    balls_list = extract_balls(all_results)
    if not balls_list:
        ranking_pairs = [(n, 0.0) for n in range(1, 26)]
    else:
        try:
            X_next = extract_latest_features(balls_list)
            probs = model.predict_proba(X_next)[:, 1]
            ranking_pairs = [(num, float(prob)) for num, prob in zip(range(1, 26), probs)]
            ranking_pairs.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            ranking_pairs = _frequency_fallback(all_results)

    ranking = [{'dezena': int(num), 'score': float(prob)} for num, prob in ranking_pairs]
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
    """Gera jogos balanceados de 15 dezenas usando o ranking como núcleo."""
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
