"""
Pattern and temporal/seasonal analysis for Lotofácil draw history.
"""

from collections import Counter, defaultdict
from scipy import stats as scipy_stats
from .stats import extract_balls


def analyze_consecutive_repeats(results):
    """
    Analyzes repetition between consecutive draws.
    Counts how many times each number repeats in consecutive games.
    """
    balls_list = extract_balls(results)
    patterns = Counter()

    for i in range(len(balls_list) - 1):
        current_draw = set(balls_list[i])
        next_draw = set(balls_list[i + 1])
        repeated = current_draw & next_draw
        patterns.update(repeated)

    return patterns


def chi_square_test(observed, expected):
    """
    Performs chi-square goodness of fit test.
    Returns (chi2_statistic, p_value).
    """
    if len(observed) != len(expected):
        return 0.0, 1.0

    observed_array = list(observed)
    expected_array = list(expected)

    total_obs = sum(observed_array)
    if total_obs == 0:
        return 0.0, 1.0

    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed_array, expected_array) if e > 0)
    df = len(observed_array) - 1

    try:
        p_val = 1.0 - scipy_stats.chi2.cdf(chi2, df) if df > 0 else 1.0
    except Exception:
        p_val = 1.0

    return chi2, p_val


def analyze_temporal_patterns(results):
    """
    Groups frequencies by day of week (dow), month, and day of month.
    Expects results rows to contain date as the last element.
    """
    patterns = defaultdict(Counter)
    for result in results:
        balls = [int(b) for b in result[:15]]
        date = result[-1]

        if hasattr(date, 'weekday'):
            dow = date.weekday()
            month = date.month
            day = date.day

            for num in balls:
                patterns[('dow', dow)][num] += 1
                patterns[('month', month)][num] += 1
                patterns[('day', day)][num] += 1

    return patterns


def detect_seasonal_patterns(results):
    """
    Detects seasonal trends (monthly and day-of-week distributions)
    and evaluates statistical significance via Chi-Square test.
    """
    seasonal_patterns = {}

    monthly_data = defaultdict(Counter)
    dow_data = defaultdict(Counter)

    for result in results:
        balls = [int(b) for b in result[:15]]
        date = result[-1]

        if hasattr(date, 'weekday'):
            month = date.month
            dow = date.weekday()

            for num in balls:
                monthly_data[month][num] += 1
                dow_data[dow][num] += 1

    # Monthly patterns
    for month in range(1, 13):
        counts = monthly_data[month]
        total = sum(counts.values())
        if total >= 10:
            expected = total / 25.0
            obs_list = [counts.get(n, 0) for n in range(1, 26)]
            exp_list = [expected] * 25

            chi2, p_value = chi_square_test(obs_list, exp_list)
            significant = p_value < 0.05
            seasonal_patterns[f'month_{month}'] = {
                'numbers': sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10],
                'total': total,
                'chi2': chi2,
                'p_value': p_value,
                'significant': significant,
            }

    # Day of week patterns
    days_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    for dow in range(7):
        counts = dow_data[dow]
        total = sum(counts.values())
        if total >= 10:
            expected = total / 25.0
            obs_list = [counts.get(n, 0) for n in range(1, 26)]
            exp_list = [expected] * 25

            chi2, p_value = chi_square_test(obs_list, exp_list)
            significant = p_value < 0.05
            seasonal_patterns[f'dow_{dow}'] = {
                'day_name': days_names[dow],
                'numbers': sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10],
                'total': total,
                'chi2': chi2,
                'p_value': p_value,
                'significant': significant,
            }

    return seasonal_patterns
