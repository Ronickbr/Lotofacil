"""
Analysis package for Lotofácil numerical analysis and statistics.
"""

from .stats import (
    extract_balls,
    calculate_basic_stats,
    calculate_delays,
    calculate_full_delays,
    calculate_hot_cold,
    calculate_windowed_frequency,
    calculate_parity_distribution,
    calculate_sum_analysis,
    calculate_number_classes,
    PRIME_NUMBERS,
    FIBONACCI_NUMBERS,
    MULTIPLES_OF_3,
    MULTIPLES_OF_5,
    PERFECT_SQUARES,
)
from .bayes import calculate_bayes_probabilities
from .patterns import (
    analyze_consecutive_repeats,
    chi_square_test,
    analyze_temporal_patterns,
    detect_seasonal_patterns,
)
from .spatial import (
    analyze_spatial_distribution,
    analyze_consecutive_sequences,
)
from .correlation import (
    calculate_correlation_matrix,
    find_top_pairs,
    find_top_triples,
)
from .probabilistic import (
    calculate_shannon_entropy,
    calculate_hypergeometric,
    calculate_hypergeometric_table,
    detect_mean_regression,
)
from .generators import (
    monte_carlo_generate,
    genetic_algorithm_generate,
    hamming_distance_optimize,
    generate_frequency_based,
    generate_delay_based,
    generate_repetition_based,
    explain_game,
)
from .scoring import (
    calculate_game_score,
    rank_games,
    select_diverse_top_games,
)
from .combinatorics import (
    generate_combinations,
    generate_reduced_closure,
    get_game_hash
)

from .ensemble import combine_analysis_methods
from .ml import (
    train_lotofacil_model,
    load_lotofacil_model,
    predict_next_numbers,
    predict_next_numbers_detailed,
    generate_suggested_games,
)

__all__ = [
    # Stats
    'extract_balls',
    'calculate_basic_stats',
    'calculate_delays',
    'calculate_full_delays',
    'calculate_hot_cold',
    'calculate_windowed_frequency',
    'calculate_parity_distribution',
    'calculate_sum_analysis',
    'calculate_number_classes',
    'PRIME_NUMBERS',
    'FIBONACCI_NUMBERS',
    'MULTIPLES_OF_3',
    'MULTIPLES_OF_5',
    'PERFECT_SQUARES',
    # Bayes
    'calculate_bayes_probabilities',
    # Patterns
    'analyze_consecutive_repeats',
    'chi_square_test',
    'analyze_temporal_patterns',
    'detect_seasonal_patterns',
    # Spatial
    'analyze_spatial_distribution',
    'analyze_consecutive_sequences',
    # Correlation
    'calculate_correlation_matrix',
    'find_top_pairs',
    'find_top_triples',
    # Probabilistic
    'calculate_shannon_entropy',
    'calculate_hypergeometric',
    'calculate_hypergeometric_table',
    'detect_mean_regression',
    # Generators
    'monte_carlo_generate',
    'genetic_algorithm_generate',
    'hamming_distance_optimize',
    # Scoring
    'calculate_game_score',
    'rank_games',
    'select_diverse_top_games',
    # Ensemble
    'combine_analysis_methods',
    # ML
    'train_lotofacil_model',
    'load_lotofacil_model',
    'predict_next_numbers',
    'predict_next_numbers_detailed',
    'generate_suggested_games',
]
