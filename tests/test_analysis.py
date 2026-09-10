"""
Unit tests for Lotofácil numerical analysis modules.
"""

import unittest
import numpy as np
import datetime
from analysis import (
    extract_balls,
    calculate_basic_stats,
    calculate_delays,
    calculate_hot_cold,
    calculate_bayes_probabilities,
    analyze_consecutive_repeats,
    chi_square_test,
    analyze_temporal_patterns,
    detect_seasonal_patterns,
    combine_analysis_methods,
    train_lotofacil_model,
    predict_next_numbers,
    generate_suggested_games,
)


class TestAnalysisModules(unittest.TestCase):

    def setUp(self):
        # Sample draws (15 numbers each from 1-25)
        self.sample_results = [
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, datetime.date(2023, 10, 1)),
            (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, datetime.date(2023, 10, 2)),
            (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, datetime.date(2023, 10, 3)),
            (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, datetime.date(2023, 10, 4)),
            (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, datetime.date(2023, 10, 5)),
        ]

    def test_extract_balls(self):
        balls = extract_balls(self.sample_results)
        self.assertEqual(len(balls), 5)
        self.assertEqual(len(balls[0]), 15)
        self.assertEqual(balls[0], list(range(1, 16)))

    def test_basic_stats(self):
        stats = calculate_basic_stats(self.sample_results)
        self.assertEqual(stats['total_games'], 5)
        self.assertIn('number_frequencies', stats)
        self.assertGreater(stats['avg_even'], 0)
        self.assertGreater(stats['avg_odd'], 0)
        self.assertGreater(stats['avg_primes'], 0)
        self.assertGreater(stats['avg_sum'], 0)

        # Check frequency of number 5 (present in all 5 draws)
        self.assertEqual(stats['number_frequencies'][5]['count'], 5)
        self.assertEqual(stats['number_frequencies'][5]['percentage'], 100.0)

    def test_calculate_delays(self):
        delays = calculate_delays(self.sample_results)
        # Most recent draw is the first item in sample_results: (1..15)
        # So number 1 is in draw 0 (delay = 0)
        self.assertEqual(delays[1], 0)
        self.assertEqual(delays[15], 0)
        # Number 19 is in draw 4 (fifth item, so 4 draws away from index 0)
        self.assertEqual(delays[19], 4)

    def test_calculate_hot_cold(self):
        hot_cold = calculate_hot_cold(self.sample_results, recent_window=5)
        self.assertIn(1, hot_cold)
        self.assertIn('status', hot_cold[1])
        self.assertIn(hot_cold[1]['status'], ['quente', 'frio', 'neutro'])

    def test_bayes_probabilities(self):
        probs = calculate_bayes_probabilities(self.sample_results)
        self.assertEqual(len(probs), 25)
        # Sum of probabilities across 25 numbers should equal 15.0
        total_expected_drawn = sum(probs.values())
        self.assertAlmostEqual(total_expected_drawn, 15.0, places=4)
        # All probabilities between 0 and 1
        for num, p in probs.items():
            self.assertTrue(0.0 <= p <= 1.0)

    def test_consecutive_repeats(self):
        repeats = analyze_consecutive_repeats(self.sample_results)
        # Draw 0 (1..15) & Draw 1 (2..16) -> repeat (2..15) = 14 numbers
        self.assertIn(2, repeats)

    def test_chi_square_test(self):
        chi2, p_val = chi_square_test([10, 10, 10], [10, 10, 10])
        self.assertEqual(chi2, 0.0)
        self.assertEqual(p_val, 1.0)

    def test_combine_analysis_methods(self):
        combined = combine_analysis_methods(self.sample_results)
        self.assertEqual(len(combined), 25)
        # Combined scores should sum to 1.0
        self.assertAlmostEqual(sum(combined.values()), 1.0, places=4)

    def test_temporal_and_seasonal_patterns(self):
        patterns = analyze_temporal_patterns(self.sample_results)
        self.assertIsNotNone(patterns)
        seasonal = detect_seasonal_patterns(self.sample_results)
        self.assertIsInstance(seasonal, dict)

    def test_ml_model_and_predictions(self):
        # Generate 15 fake draws to train model
        extended_results = []
        for i in range(20):
            draw = sorted(np.random.choice(range(1, 26), 15, replace=False).tolist())
            draw.append(datetime.date(2023, 1, 1))
            extended_results.append(draw)

        model, mean_acc = train_lotofacil_model(extended_results, 'test_model.pkl')
        self.assertIsNotNone(model)
        self.assertTrue(0.0 <= mean_acc <= 1.0)

        preds = predict_next_numbers(model, extended_results[0], top_k=10)
        self.assertLessEqual(len(preds), 10)

        games = generate_suggested_games(preds, num_games=6)
        self.assertEqual(len(games), 6)
        for g in games:
            self.assertEqual(len(g), 15)
            self.assertEqual(len(set(g)), 15)


if __name__ == '__main__':
    unittest.main()
