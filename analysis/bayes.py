"""
Bayesian probability analysis module for Lotofácil.
"""

from collections import Counter
from .stats import extract_balls


def calculate_bayes_probabilities(results, prior_weight=1.0):
    """
    Calculates Bayesian posterior probability for each number (1-25) using Beta-Binomial conjugate updating.

    In Lotofácil, 15 balls are drawn out of 25 per game.
    Prior expectation for any number: p_0 = 15 / 25 = 0.60.
    With prior weight `prior_weight`, prior pseudo-counts are:
        alpha = 15 * prior_weight
        beta = 10 * prior_weight

    Given N games and S_i occurrences of number i:
        Posterior Mean P(number i is drawn) = (S_i + alpha) / (N + alpha + beta)

    Returns:
        dict: {number: posterior_probability}
    """
    balls_list = extract_balls(results)
    total_games = len(balls_list)

    if total_games == 0:
        return {num: 0.60 for num in range(1, 26)}

    alpha = 15.0 * prior_weight
    beta = 10.0 * prior_weight

    counts = Counter()
    for draw in balls_list:
        counts.update(draw)

    posterior_probs = {}
    for num in range(1, 26):
        successes = counts.get(num, 0)
        prob = (successes + alpha) / (total_games + alpha + beta)
        posterior_probs[num] = prob

    return posterior_probs
