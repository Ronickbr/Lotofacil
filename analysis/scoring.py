"""
Multidimensional scoring system for Lotofácil games.
Optimized for high performance by precomputing historical states.
"""

import numpy as np
from collections import Counter
from .stats import extract_balls, PRIME_NUMBERS
from .spatial import NUM_TO_ROW


class GameScorer:
    """Precomputes historical data once to score thousands of games instantly."""
    
    def __init__(self, results, weights=None):
        self.weights = weights or {
            'frequency': 0.20,
            'delay': 0.15,
            'repetition': 0.10,
            'parity': 0.10,
            'sum': 0.15,
            'spatial': 0.10,
            'correlation': 0.10,
            'primes': 0.10,
        }
        self.balls_list = extract_balls(results)
        self.total = len(self.balls_list)
        
        if self.total == 0:
            return

        # Precompute frequencies
        freq_counter = Counter()
        for draw in self.balls_list:
            freq_counter.update(draw)
        self.freq_probs = {n: freq_counter.get(n, 0) / self.total for n in range(1, 26)}
        
        # Precompute delays
        self.delays = {}
        for num in range(1, 26):
            for i, draw in enumerate(self.balls_list):
                if num in draw:
                    self.delays[num] = i
                    break
            else:
                self.delays[num] = self.total
        self.max_delay = max(self.delays.values()) if self.delays else 1
        
        # Precompute repeats
        self.last_draw = set(self.balls_list[0]) if self.balls_list else set()
        
        # Precompute sums
        sums_hist = [sum(d) for d in self.balls_list]
        self.mean_sum = np.mean(sums_hist)
        self.std_sum = max(np.std(sums_hist), 1)
        
        # Precompute correlations (pair counts)
        self.pair_counts = Counter()
        for draw in self.balls_list:
            ds = sorted(set(draw))
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    self.pair_counts[(ds[i], ds[j])] += 1
                    
        self.expected_pair_freq = 13 * 14 / (25 * 24)

    def score_game(self, game):
        if self.total == 0:
            return {'total_score': 0.5, 'components': {}, 'game_sum': sum(game), 'evens': 0, 'odds': 0, 'primes': 0}
            
        game_set = set(game)
        
        # --- Component 1: Frequency ---
        f_score = sum(self.freq_probs[n] for n in game) / 15.0
        f_score = min(1.0, f_score / 0.75)
        
        # --- Component 2: Delay ---
        a_score = sum(1.0 - self.delays.get(n, self.total) / max(self.max_delay, 1) for n in game) / 15.0
        
        # --- Component 3: Repetition ---
        repeats = len(game_set & self.last_draw)
        r_score = max(0, 1.0 - abs(repeats - 9.5) / 7.5)
        
        # --- Component 4: Parity ---
        evens = sum(1 for n in game if n % 2 == 0)
        p_score = 1.0 - abs(evens - 7.5) / 7.5
        
        # --- Component 5: Sum ---
        game_sum = sum(game)
        s_score = max(0, 1.0 - abs(game_sum - self.mean_sum) / (2 * self.std_sum))
        
        # --- Component 6: Spatial ---
        rows = [0] * 5
        for n in game:
            rows[NUM_TO_ROW[n]] += 1
        d_score = max(0, 1.0 - np.std(rows) / 2.0)
        
        # --- Component 7: Correlation ---
        game_sorted = sorted(game)
        pair_score_sum = 0
        pair_count = 0
        for i in range(len(game_sorted)):
            for j in range(i + 1, len(game_sorted)):
                pair_score_sum += self.pair_counts.get((game_sorted[i], game_sorted[j]), 0)
                pair_count += 1
        
        avg_pair_freq = pair_score_sum / max(pair_count, 1) / max(self.total, 1)
        c_score = min(1.0, avg_pair_freq / max(self.expected_pair_freq, 0.001))
        
        # --- Component 8: Primes ---
        primes_in_game = sum(1 for n in game if n in PRIME_NUMBERS)
        h_score = max(0, 1.0 - abs(primes_in_game - 5.4) / 5.4)
        
        components = {
            'frequency': round(f_score, 4),
            'delay': round(a_score, 4),
            'repetition': round(r_score, 4),
            'parity': round(p_score, 4),
            'sum': round(s_score, 4),
            'spatial': round(d_score, 4),
            'correlation': round(c_score, 4),
            'primes': round(h_score, 4),
        }
        total_score = sum(self.weights[k] * components[k] for k in self.weights)
        
        return {
            'total_score': round(total_score, 4),
            'components': components,
            'game_sum': game_sum,
            'evens': evens,
            'odds': 15 - evens,
            'primes': primes_in_game,
        }


def calculate_game_score(game, results, weights=None):
    """Fallback function for scoring a single game (used minimally now)."""
    scorer = GameScorer(results, weights)
    return scorer.score_game(game)


def rank_games(games, results, weights=None):
    """Scores and ranks a list of games optimally."""
    scorer = GameScorer(results, weights)
    scored = []
    for game in games:
        score_data = scorer.score_game(game)
        scored.append({
            'numbers': sorted(game),
            **score_data,
        })
    scored.sort(key=lambda x: x['total_score'], reverse=True)
    return scored


def select_diverse_top_games(games, results, num_select=6, weights=None):
    """
    From a pool of games:
    1. Score all games optimally
    2. Sort by score
    3. Greedily pick top `num_select` games maximizing Hamming distance diversity
    """
    ranked = rank_games(games, results, weights)

    if len(ranked) <= num_select:
        return ranked

    def hamming(a, b):
        return len(set(a).symmetric_difference(set(b)))

    selected = [ranked[0]]

    for candidate in ranked[1:]:
        if len(selected) >= num_select:
            break
        min_dist = min(hamming(candidate['numbers'], sel['numbers']) for sel in selected)
        if min_dist >= 4:
            selected.append(candidate)

    if len(selected) < num_select:
        for candidate in ranked:
            if len(selected) >= num_select:
                break
            if candidate not in selected:
                selected.append(candidate)

    return selected[:num_select]
