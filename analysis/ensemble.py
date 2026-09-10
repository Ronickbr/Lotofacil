"""
Ensemble analysis for combining multiple numerical analysis models.
"""

from collections import Counter, defaultdict
from .stats import extract_balls, calculate_basic_stats
from .bayes import calculate_bayes_probabilities
from .patterns import analyze_consecutive_repeats


def _normalize_dict(score_dict):
    """
    Normalizes dictionary values (numbers 1-25) so they sum to 1.0.
    If total is 0, assigns uniform distribution 1/25 = 0.04.
    """
    total = sum(score_dict.get(num, 0) for num in range(1, 26))
    if total <= 0:
        return {num: 1.0 / 25.0 for num in range(1, 26)}
    return {num: score_dict.get(num, 0) / float(total) for num in range(1, 26)}


def combine_analysis_methods(results, weights=None):
    """
    Combines Frequency, Consecutive Patterns, and Bayesian probabilities into a unified score.
    Each method's outputs are normalized to a common scale before weighting.

    Default weights:
        - 40% Frequency Analysis
        - 30% Consecutive Repeat Patterns
        - 30% Bayesian Posterior Probabilities

    Returns:
        dict: {number: combined_normalized_score}
    """
    if weights is None:
        weights = {'freq': 0.4, 'patterns': 0.3, 'bayes': 0.3}

    # Normalize weight vector
    total_weight = sum(weights.values())
    if total_weight <= 0:
        norm_weights = {'freq': 0.4, 'patterns': 0.3, 'bayes': 0.3}
    else:
        norm_weights = {k: v / total_weight for k, v in weights.items()}

    balls_list = extract_balls(results)
    if not balls_list:
        return {num: 1.0 / 25.0 for num in range(1, 26)}

    # 1. Frequency scores
    basic_stats = calculate_basic_stats(results)
    freq_dict = {
        num: data['count']
        for num, data in basic_stats['number_frequencies'].items()
    }
    norm_freq = _normalize_dict(freq_dict)

    # 2. Pattern scores (consecutive repeats)
    patterns = analyze_consecutive_repeats(results)
    norm_patterns = _normalize_dict(dict(patterns))

    # 3. Bayes probabilities
    bayes_probs = calculate_bayes_probabilities(results)
    norm_bayes = _normalize_dict(bayes_probs)

    # Combine with weights
    combined_scores = {}
    for num in range(1, 26):
        score = (
            norm_weights['freq'] * norm_freq[num] +
            norm_weights['patterns'] * norm_patterns[num] +
            norm_weights['bayes'] * norm_bayes[num]
        )
        combined_scores[num] = score

    return combined_scores
