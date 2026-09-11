"""
Statistical analysis tools for Lotofácil draw history.
Includes: frequency, delays, hot/cold, parity, sum, primes, fibonacci, multiples, windowed analysis.
"""

from collections import Counter, defaultdict
from math import comb, sqrt
import numpy as np

PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23}
FIBONACCI_NUMBERS = {1, 2, 3, 5, 8, 13, 21}
MULTIPLES_OF_3 = {3, 6, 9, 12, 15, 18, 21, 24}
MULTIPLES_OF_5 = {5, 10, 15, 20, 25}
PERFECT_SQUARES = {1, 4, 9, 16, 25}


def extract_balls(results):
    """
    Extracts a list of integer ball lists (15 numbers each) from database rows.
    Handles tuples/lists where the last column or last item might be data_sorteio.
    """
    import pandas as pd
    balls_list = []
    for row in results:
        if isinstance(row, (dict, pd.Series)):
            balls = [int(row[f'bola{i}']) for i in range(1, 16)]
        else:
            # Slices first 15 columns as ball numbers
            balls = [int(b) for b in row[:15]]
        balls_list.append(balls)
    return balls_list


def calculate_basic_stats(results):
    """
    Calculates fundamental descriptive statistics:
    - Number frequencies and percentages
    - Parity (even/odd) averages
    - Prime numbers averages
    - Total sum averages
    """
    balls_list = extract_balls(results)
    total_games = len(balls_list)
    if total_games == 0:
        return {
            'total_games': 0,
            'avg_even': 0.0,
            'avg_odd': 0.0,
            'avg_primes': 0.0,
            'avg_sum': 0.0,
            'number_frequencies': {},
        }

    all_numbers = []
    total_even = 0
    total_odd = 0
    total_primes = 0
    total_sum = 0

    for draw in balls_list:
        all_numbers.extend(draw)
        even_in_draw = sum(1 for n in draw if n % 2 == 0)
        odd_in_draw = sum(1 for n in draw if n % 2 != 0)
        primes_in_draw = sum(1 for n in draw if n in PRIME_NUMBERS)
        sum_in_draw = sum(draw)

        total_even += even_in_draw
        total_odd += odd_in_draw
        total_primes += primes_in_draw
        total_sum += sum_in_draw

    freq_counter = Counter(all_numbers)
    number_frequencies = {
        num: {
            'count': freq_counter.get(num, 0),
            'percentage': (freq_counter.get(num, 0) / total_games) * 100,
        }
        for num in range(1, 26)
    }

    return {
        'total_games': total_games,
        'avg_even': total_even / total_games,
        'avg_odd': total_odd / total_games,
        'avg_primes': total_primes / total_games,
        'avg_sum': total_sum / total_games,
        'number_frequencies': number_frequencies,
    }


def calculate_delays(results):
    """
    Calculates delay ('atraso') for each number (1-25).
    Delay is the number of consecutive draws since the number last appeared.
    Results are assumed to be ordered ASC (oldest draw first).
    """
    balls_list = extract_balls(results)
    delays = {}

    for num in range(1, 26):
        delay = 0
        found = False
        for draw in reversed(balls_list):
            if num in draw:
                found = True
                break
            delay += 1
        delays[num] = delay if found else len(balls_list)

    return delays


def calculate_full_delays(results):
    """
    Comprehensive delay analysis for each number (1-25):
    - current: atraso atual (quantos concursos desde a última aparição)
    - mean: atraso médio histórico
    - max: atraso máximo histórico
    - std: desvio padrão dos atrasos
    - percentile: percentil do atraso atual em relação ao histórico
    - all_delays: lista de todos os atrasos (para gráficos)
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total == 0:
        return {num: {'current': 0, 'mean': 0, 'max': 0, 'std': 0, 'percentile': 0} for num in range(1, 26)}

    result = {}
    for num in range(1, 26):
        occurrences = [index for index, draw in enumerate(balls_list) if num in draw]
        if occurrences:
            result_current = total - 1 - occurrences[-1]
            delays = [occurrences[0]]
            delays.extend(
                current - previous - 1
                for previous, current in zip(occurrences, occurrences[1:])
            )
        else:
            delays = [total]
            result_current = total

        arr = np.array(delays)
        pct = float(np.sum(arr <= result_current) / len(arr) * 100) if len(arr) > 0 else 0

        result[num] = {
            'current': result_current,
            'mean': round(float(np.mean(arr)), 2),
            'max': int(np.max(arr)),
            'std': round(float(np.std(arr)), 2),
            'percentile': round(pct, 1),
        }

    return result


def calculate_hot_cold(results, recent_window=10):
    """
    Categorizes numbers as hot ('quente'), neutral ('neutro'), or cold ('frio')
    based on their occurrences in the last `recent_window` draws compared to expected count.
    Expected occurrences in N draws = N * (15 / 25) = N * 0.6.
    """
    balls_list = extract_balls(results)
    recent_draws = balls_list[-recent_window:] if balls_list else []
    actual_window = len(recent_draws)
    expected = actual_window * 0.6

    counts = Counter()
    for draw in recent_draws:
        counts.update(draw)

    hot_cold = {}
    for num in range(1, 26):
        cnt = counts.get(num, 0)
        if cnt > expected + 1.0:
            status = 'quente'
        elif cnt < expected - 1.0:
            status = 'frio'
        else:
            status = 'neutro'

        hot_cold[num] = {
            'count': cnt,
            'expected': expected,
            'status': status,
        }

    return hot_cold


def calculate_windowed_frequency(results, windows=None):
    """
    Frequency analysis across multiple time windows.
    Score(n) = w1*F_10 + w2*F_50 + w3*F_100 + w4*F_total

    Returns dict per number with frequency in each window + weighted score.
    """
    if windows is None:
        windows = [10, 20, 50, 100, 500]

    balls_list = extract_balls(results)
    total = len(balls_list)

    if total == 0:
        return {num: {'windows': {}, 'score': 0.0} for num in range(1, 26)}

    # Weights: more recent windows get higher weight
    weight_map = {10: 0.30, 20: 0.25, 50: 0.20, 100: 0.15, 500: 0.05}
    weight_total = 0.05

    result = {}
    for num in range(1, 26):
        window_freqs = {}
        weighted_score = 0.0

        for w in windows:
            subset = balls_list[-min(w, total):]
            count = sum(1 for draw in subset if num in draw)
            freq = count / len(subset) if subset else 0
            window_freqs[w] = {
                'count': count,
                'total': len(subset),
                'frequency': round(freq * 100, 2),
            }
            weighted_score += weight_map.get(w, 0.1) * freq

        # Total frequency
        total_count = sum(1 for draw in balls_list if num in draw)
        total_freq = total_count / total
        window_freqs['total'] = {
            'count': total_count,
            'total': total,
            'frequency': round(total_freq * 100, 2),
        }
        weighted_score += weight_total * total_freq

        result[num] = {
            'windows': window_freqs,
            'score': round(weighted_score, 6),
        }

    return result


def calculate_parity_distribution(results):
    """
    Analyzes even/odd distribution per draw.
    Returns histogram of (n_even, n_odd) configurations with:
    - observed count
    - observed percentage
    - theoretical probability using combinatorics: C(12,even)*C(13,odd) / C(25,15)
    (12 even numbers and 13 odd numbers in 1-25)
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total == 0:
        return {'distribution': {}, 'total': 0}

    # In 1-25: even = {2,4,6,8,10,12,14,16,18,20,22,24} = 12 numbers
    # odd = {1,3,5,7,9,11,13,15,17,19,21,23,25} = 13 numbers
    n_even_pool = 12
    n_odd_pool = 13
    total_comb = comb(25, 15)

    config_counts = Counter()
    for draw in balls_list:
        evens = sum(1 for n in draw if n % 2 == 0)
        config_counts[evens] += 1

    distribution = {}
    for n_even in range(0, 16):
        n_odd = 15 - n_even
        if n_even > n_even_pool or n_odd > n_odd_pool:
            theoretical = 0.0
        else:
            theoretical = comb(n_even_pool, n_even) * comb(n_odd_pool, n_odd) / total_comb

        observed = config_counts.get(n_even, 0)
        distribution[n_even] = {
            'even': n_even,
            'odd': n_odd,
            'observed': observed,
            'observed_pct': round(observed / total * 100, 2) if total else 0,
            'theoretical_pct': round(theoretical * 100, 2),
        }

    return {'distribution': distribution, 'total': total}


def calculate_sum_analysis(results):
    """
    Analyzes the sum of drawn numbers per game:
    - mean, median, std, min, max
    - percentile bands (10th, 25th, 50th, 75th, 90th)
    - histogram buckets for charting
    """
    balls_list = extract_balls(results)
    if not balls_list:
        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'percentiles': {}, 'histogram': []}

    sums = [sum(draw) for draw in balls_list]
    arr = np.array(sums)

    # Histogram: buckets of 5
    hist_min = int(np.min(arr)) // 5 * 5
    hist_max = (int(np.max(arr)) // 5 + 1) * 5
    buckets = list(range(hist_min, hist_max + 1, 5))
    histogram = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = int(np.sum((arr >= lo) & (arr < hi)))
        histogram.append({'range': f'{lo}-{hi-1}', 'count': count})

    return {
        'mean': round(float(np.mean(arr)), 2),
        'median': round(float(np.median(arr)), 2),
        'std': round(float(np.std(arr)), 2),
        'min': int(np.min(arr)),
        'max': int(np.max(arr)),
        'percentiles': {
            '10': round(float(np.percentile(arr, 10)), 1),
            '25': round(float(np.percentile(arr, 25)), 1),
            '50': round(float(np.percentile(arr, 50)), 1),
            '75': round(float(np.percentile(arr, 75)), 1),
            '90': round(float(np.percentile(arr, 90)), 1),
        },
        'histogram': histogram,
        'total': len(sums),
    }


def calculate_number_classes(results):
    """
    Distribution of special number classes per draw:
    - primes, fibonacci, multiples of 3, multiples of 5, perfect squares
    Returns average per draw and distribution histogram for each class.
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total == 0:
        return {}

    classes = {
        'primes': PRIME_NUMBERS,
        'fibonacci': FIBONACCI_NUMBERS,
        'multiples_3': MULTIPLES_OF_3,
        'multiples_5': MULTIPLES_OF_5,
        'perfect_squares': PERFECT_SQUARES,
    }
    labels = {
        'primes': 'Primos',
        'fibonacci': 'Fibonacci',
        'multiples_3': 'Múltiplos de 3',
        'multiples_5': 'Múltiplos de 5',
        'perfect_squares': 'Quadrados Perfeitos',
    }

    result = {}
    for cls_key, cls_set in classes.items():
        counts_per_draw = []
        distribution = Counter()
        for draw in balls_list:
            cnt = sum(1 for n in draw if n in cls_set)
            counts_per_draw.append(cnt)
            distribution[cnt] += 1

        arr = np.array(counts_per_draw)
        pool_size = len(cls_set)

        result[cls_key] = {
            'label': labels[cls_key],
            'pool_size': pool_size,
            'pool_numbers': sorted(cls_set),
            'avg': round(float(np.mean(arr)), 2),
            'std': round(float(np.std(arr)), 2),
            'distribution': {k: v for k, v in sorted(distribution.items())},
        }

    return result
