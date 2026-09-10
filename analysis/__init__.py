"""
Analysis package for Lotofácil numerical analysis and statistics.
"""

from .stats import (
    extract_balls,
    calculate_basic_stats,
    calculate_delays,
    calculate_hot_cold,
    PRIME_NUMBERS,
)
from .bayes import calculate_bayes_probabilities
from .patterns import (
    analyze_consecutive_repeats,
    chi_square_test,
    analyze_temporal_patterns,
    detect_seasonal_patterns,
)
from .ensemble import combine_analysis_methods
from .ml import train_lotofacil_model, predict_next_numbers, generate_suggested_games

__all__ = [
    'extract_balls',
    'calculate_basic_stats',
    'calculate_delays',
    'calculate_hot_cold',
    'PRIME_NUMBERS',
    'calculate_bayes_probabilities',
    'analyze_consecutive_repeats',
    'chi_square_test',
    'analyze_temporal_patterns',
    'detect_seasonal_patterns',
    'combine_analysis_methods',
    'train_lotofacil_model',
    'predict_next_numbers',
    'generate_suggested_games',
]
