"""
Statistical analysis tools for Lotofácil draw history.
"""

from collections import Counter, defaultdict
import pandas as pd

PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23}


def extract_balls(results):
    """
    Extracts a list of integer ball lists (15 numbers each) from database rows.
    Handles tuples/lists where the last column or last item might be data_sorteio.
    """
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
    Results are assumed to be ordered DESC (most recent draw first).
    """
    balls_list = extract_balls(results)
    delays = {}

    for num in range(1, 26):
        delay = 0
        found = False
        for draw in balls_list:
            if num in draw:
                found = True
                break
            delay += 1
        delays[num] = delay if found else len(balls_list)

    return delays


def calculate_hot_cold(results, recent_window=10):
    """
    Categorizes numbers as hot ('quente'), neutral ('neutro'), or cold ('frio')
    based on their occurrences in the last `recent_window` draws compared to expected count.
    Expected occurrences in N draws = N * (15 / 25) = N * 0.6.
    """
    balls_list = extract_balls(results)
    recent_draws = balls_list[:recent_window] if balls_list else []
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
