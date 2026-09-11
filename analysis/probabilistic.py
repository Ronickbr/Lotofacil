"""
Probabilistic analysis for Lotofácil:
- Shannon Entropy
- Hypergeometric Distribution
- Mean Regression Detection
"""

import numpy as np
from math import comb, log2
from collections import Counter
from .stats import extract_balls


def calculate_shannon_entropy(results, windows=None):
    """
    Calculates Shannon entropy of number frequencies across time windows.
    H = -Σ p_i * log2(p_i)

    Maximum entropy for 25 equally-likely numbers = log2(25) ≈ 4.644

    Returns entropy for each window (how uniform the distribution is).
    """
    if windows is None:
        windows = [20, 100, 'total']

    balls_list = extract_balls(results)
    total = len(balls_list)
    max_entropy = log2(25)

    result = {}
    for w in windows:
        if w == 'total':
            subset = balls_list
            label = 'Histórico Completo'
        else:
            subset = balls_list[:min(w, total)]
            label = f'Últimos {w} concursos'

        if not subset:
            result[str(w)] = {'entropy': 0, 'max_entropy': max_entropy, 'ratio': 0, 'label': label}
            continue

        counts = Counter()
        for draw in subset:
            counts.update(draw)

        total_balls = sum(counts.values())
        if total_balls == 0:
            result[str(w)] = {'entropy': 0, 'max_entropy': max_entropy, 'ratio': 0, 'label': label}
            continue

        entropy = 0.0
        for num in range(1, 26):
            p = counts.get(num, 0) / total_balls
            if p > 0:
                entropy -= p * log2(p)

        result[str(w)] = {
            'entropy': round(entropy, 4),
            'max_entropy': round(max_entropy, 4),
            'ratio': round(entropy / max_entropy * 100, 2),
            'label': label,
            'interpretation': _interpret_entropy(entropy / max_entropy),
        }

    return result


def _interpret_entropy(ratio):
    """Interprets entropy ratio (0-1)."""
    if ratio > 0.99:
        return 'Distribuição extremamente uniforme (próxima do ideal aleatório)'
    elif ratio > 0.97:
        return 'Distribuição muito uniforme'
    elif ratio > 0.95:
        return 'Distribuição razoavelmente uniforme'
    elif ratio > 0.90:
        return 'Leve concentração detectada'
    else:
        return 'Concentração significativa (possível viés ou amostra pequena)'


def calculate_hypergeometric(chosen_group_size, target_hits, total_pool=25, draw_size=15):
    """
    Hypergeometric probability:
    P(X=k) = C(K, k) * C(N-K, n-k) / C(N, n)

    Where:
    - N = total_pool (25)
    - K = chosen_group_size (e.g., 18)
    - n = draw_size (15)
    - k = target_hits (e.g., 13)

    Returns probability for each k from max(0, n-(N-K)) to min(K, n).
    """
    N = total_pool
    K = chosen_group_size
    n = draw_size
    total_comb = comb(N, n)

    k_min = max(0, n - (N - K))
    k_max = min(K, n)

    probabilities = {}
    for k in range(k_min, k_max + 1):
        if K >= k and (N - K) >= (n - k):
            prob = comb(K, k) * comb(N - K, n - k) / total_comb
        else:
            prob = 0.0
        probabilities[k] = {
            'probability': round(prob, 8),
            'percentage': round(prob * 100, 4),
            'odds': f'1 em {round(1/prob):,}' if prob > 0 else 'Impossível',
        }

    return {
        'group_size': K,
        'draw_size': n,
        'total_pool': N,
        'probabilities': probabilities,
    }


def calculate_hypergeometric_table():
    """
    Precomputed table for common group sizes (15-22) and target hits (11-15).
    Useful for the UI to display a quick reference.
    """
    table = {}
    for group in range(15, 23):
        row = {}
        for hits in range(max(0, 15 - (25 - group)), min(group, 15) + 1):
            if hits >= 11:
                prob = comb(group, hits) * comb(25 - group, 15 - hits) / comb(25, 15)
                row[hits] = {
                    'probability': round(prob, 8),
                    'percentage': round(prob * 100, 4),
                }
        table[group] = row
    return table


def detect_mean_regression(results, recent_window=20):
    """
    Detects numbers whose recent frequency deviates significantly from historical average.
    Flags numbers that may be experiencing inflation (abnormally hot) or deflation (abnormally cold).

    Uses z-score: z = (observed - expected) / std_error
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total < recent_window + 10:
        return {}

    recent = balls_list[:recent_window]
    historical = balls_list[recent_window:]

    # Historical baseline
    hist_counts = Counter()
    for draw in historical:
        hist_counts.update(draw)
    hist_total = len(historical)

    # Recent counts
    recent_counts = Counter()
    for draw in recent:
        recent_counts.update(draw)

    expected_per_draw = 15 / 25  # 0.60

    result = {}
    for num in range(1, 26):
        hist_freq = hist_counts.get(num, 0) / hist_total if hist_total > 0 else expected_per_draw
        recent_freq = recent_counts.get(num, 0) / recent_window

        # Standard error for binomial proportion
        std_error = np.sqrt(hist_freq * (1 - hist_freq) / recent_window) if hist_freq > 0 and hist_freq < 1 else 0.1

        z_score = (recent_freq - hist_freq) / std_error if std_error > 0 else 0

        if z_score > 1.96:
            status = 'inflated'
            status_label = 'Inflacionado'
        elif z_score < -1.96:
            status = 'deflated'
            status_label = 'Deflacionado'
        elif z_score > 1.0:
            status = 'slightly_hot'
            status_label = 'Levemente Quente'
        elif z_score < -1.0:
            status = 'slightly_cold'
            status_label = 'Levemente Frio'
        else:
            status = 'normal'
            status_label = 'Normal'

        result[num] = {
            'historical_freq': round(hist_freq * 100, 2),
            'recent_freq': round(recent_freq * 100, 2),
            'z_score': round(z_score, 3),
            'status': status,
            'status_label': status_label,
            'tendency': 'Tende a cair' if status == 'inflated' else ('Tende a subir' if status == 'deflated' else 'Estável'),
        }

    return result
