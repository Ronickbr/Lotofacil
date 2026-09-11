"""
Advanced game generators for Lotofácil:
- Monte Carlo simulation with statistical filters
- Genetic Algorithm optimization
- Hamming Distance diversification
"""

import random
import numpy as np
from math import comb
from collections import Counter
from .stats import extract_balls, PRIME_NUMBERS


def monte_carlo_generate(results, num_games=6, num_simulations=100000, filters=None):
    """
    Monte Carlo generator:
    1. Simulates `num_simulations` random draws of 15 from 25
    2. Applies statistical filters (parity, sum range, consecutive limits, etc.)
    3. Scores each surviving game by how "typical" it is relative to historical data
    4. Returns top `num_games` most typical games

    Filters dict can contain:
    - min_even / max_even: parity bounds
    - min_sum / max_sum: sum bounds
    - max_consecutive: max length of consecutive sequence
    - min_repeat: minimum numbers repeated from last draw
    """
    if filters is None:
        filters = {}

    balls_list = extract_balls(results)
    total = len(balls_list)

    # Compute historical baselines
    hist_freq = Counter()
    for draw in balls_list:
        hist_freq.update(draw)

    if total > 0:
        freq_scores = {n: hist_freq.get(n, 0) / total for n in range(1, 26)}
        sums = [sum(d) for d in balls_list]
        hist_mean_sum = np.mean(sums)
        hist_std_sum = max(np.std(sums), 1)
    else:
        freq_scores = {n: 0.6 for n in range(1, 26)}
        hist_mean_sum = 195
        hist_std_sum = 15

    last_draw = set(balls_list[0]) if balls_list else set()

    # Filter params
    min_even = filters.get('min_even', 5)
    max_even = filters.get('max_even', 10)
    min_sum = filters.get('min_sum', int(hist_mean_sum - 2 * hist_std_sum))
    max_sum = filters.get('max_sum', int(hist_mean_sum + 2 * hist_std_sum))
    max_consecutive = filters.get('max_consecutive', 5)
    min_repeat = filters.get('min_repeat', 7)

    candidates = []
    numbers = list(range(1, 26))

    for _ in range(num_simulations):
        game = sorted(random.sample(numbers, 15))
        game_set = set(game)

        # Filter: parity
        evens = sum(1 for n in game if n % 2 == 0)
        if evens < min_even or evens > max_even:
            continue

        # Filter: sum range
        s = sum(game)
        if s < min_sum or s > max_sum:
            continue

        # Filter: consecutive sequence length
        max_seq = _max_consecutive_len(game)
        if max_seq > max_consecutive:
            continue

        # Filter: minimum repetition from last draw
        if last_draw:
            repeats = len(game_set & last_draw)
            if repeats < min_repeat:
                continue

        # Score: weighted frequency of chosen numbers
        score = sum(freq_scores[n] for n in game)
        # Penalize deviation from mean sum
        sum_penalty = abs(s - hist_mean_sum) / hist_std_sum
        final_score = score - sum_penalty * 0.5

        candidates.append((game, round(final_score, 4)))

    # Sort by score descending, pick top games with diversity
    candidates.sort(key=lambda x: x[1], reverse=True)

    if not candidates:
        # Fallback: generate random games without filters
        return [sorted(random.sample(numbers, 15)) for _ in range(num_games)]

    selected = _select_diverse(candidates, num_games, min_hamming=4)

    return [{'numbers': g, 'score': s} for g, s in selected]


def genetic_algorithm_generate(results, num_games=6, population_size=200,
                                generations=100, mutation_rate=0.05):
    """
    Genetic Algorithm for Lotofácil game generation.
    Chromosome: 25-bit binary vector (1=selected, 0=not), constraint: exactly 15 ones.
    Fitness: multidimensional score based on frequency, parity, sum, spatial balance.
    """
    balls_list = extract_balls(results)
    total = len(balls_list)

    # Precompute fitness components
    freq_counter = Counter()
    for draw in balls_list:
        freq_counter.update(draw)

    freq_scores = {n: freq_counter.get(n, 0) / max(total, 1) for n in range(1, 26)}
    sums_hist = [sum(d) for d in balls_list] if balls_list else [195]
    mean_sum = np.mean(sums_hist)
    std_sum = max(np.std(sums_hist), 1)

    def decode(chromosome):
        """Convert 25-bit chromosome to list of selected numbers."""
        return [i + 1 for i, bit in enumerate(chromosome) if bit == 1]

    def encode(numbers):
        """Convert list of numbers to 25-bit chromosome."""
        chrom = [0] * 25
        for n in numbers:
            chrom[n - 1] = 1
        return chrom

    def fitness(chromosome):
        nums = decode(chromosome)
        if len(nums) != 15:
            return -1000

        # 1. Frequency score
        freq_score = sum(freq_scores[n] for n in nums)

        # 2. Parity balance (ideal: 7-8 even)
        evens = sum(1 for n in nums if n % 2 == 0)
        parity_score = 1.0 - abs(evens - 7.5) / 7.5

        # 3. Sum proximity to mean
        s = sum(nums)
        sum_score = max(0, 1.0 - abs(s - mean_sum) / (2 * std_sum))

        # 4. Row balance (ideal: 3 per row)
        rows = [0] * 5
        for n in nums:
            rows[(n - 1) // 5] += 1
        row_balance = 1.0 - np.std(rows) / 3.0

        # 5. Prime distribution
        primes = sum(1 for n in nums if n in PRIME_NUMBERS)
        prime_score = 1.0 - abs(primes - 5.4) / 5.4  # Expected ~5.4 primes

        return freq_score * 0.35 + parity_score * 0.20 + sum_score * 0.20 + row_balance * 0.15 + prime_score * 0.10

    def create_individual():
        nums = random.sample(range(1, 26), 15)
        return encode(nums)

    def crossover(parent1, parent2):
        """Uniform crossover maintaining exactly 15 ones."""
        child = [0] * 25
        # Take intersection
        for i in range(25):
            if parent1[i] == 1 and parent2[i] == 1:
                child[i] = 1

        ones = sum(child)
        remaining_indices = [i for i in range(25) if child[i] == 0]
        # Fill remaining from either parent or random
        candidates = [i for i in remaining_indices if parent1[i] == 1 or parent2[i] == 1]
        random.shuffle(candidates)
        extras = [i for i in remaining_indices if i not in candidates]
        random.shuffle(extras)
        fill_pool = candidates + extras

        for idx in fill_pool:
            if ones >= 15:
                break
            child[idx] = 1
            ones += 1

        return child

    def mutate(chromosome):
        """Swap one selected number with one unselected."""
        chrom = chromosome[:]
        ones = [i for i in range(25) if chrom[i] == 1]
        zeros = [i for i in range(25) if chrom[i] == 0]
        if ones and zeros:
            swap_out = random.choice(ones)
            swap_in = random.choice(zeros)
            chrom[swap_out] = 0
            chrom[swap_in] = 1
        return chrom

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    for gen in range(generations):
        scored = [(ind, fitness(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Elitism: top 10%
        elite_size = max(2, population_size // 10)
        new_pop = [s[0] for s in scored[:elite_size]]

        # Tournament selection + crossover
        while len(new_pop) < population_size:
            t1 = random.sample(scored[:population_size // 2], 2)
            parent1 = max(t1, key=lambda x: x[1])[0]
            t2 = random.sample(scored[:population_size // 2], 2)
            parent2 = max(t2, key=lambda x: x[1])[0]

            child = crossover(parent1, parent2)

            if random.random() < mutation_rate:
                child = mutate(child)

            new_pop.append(child)

        population = new_pop

    # Final scoring
    final_scored = [(decode(ind), fitness(ind)) for ind in population]
    final_scored.sort(key=lambda x: x[1], reverse=True)

    # Select diverse top games
    selected = _select_diverse(
        [(sorted(nums), score) for nums, score in final_scored[:50]],
        num_games,
        min_hamming=3,
    )

    return [{'numbers': g, 'score': round(s, 4)} for g, s in selected]


def hamming_distance_optimize(candidates, num_games=6):
    """
    Given a list of candidate games (each a list of 15 numbers),
    selects `num_games` that maximize the minimum pairwise Hamming distance.

    Hamming distance for lottery: d_H(A,B) = |A △ B| (symmetric difference)

    Greedy approach: start with best-scored game, iteratively add game
    that maximizes minimum distance to all already selected games.
    """
    if len(candidates) <= num_games:
        return candidates

    def hamming(a, b):
        sa, sb = set(a), set(b)
        return len(sa.symmetric_difference(sb))

    selected = [candidates[0]]

    remaining = list(candidates[1:])
    while len(selected) < num_games and remaining:
        best_idx = -1
        best_min_dist = -1

        for i, cand in enumerate(remaining):
            min_dist = min(hamming(cand, sel) for sel in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected


def _max_consecutive_len(sorted_game):
    """Returns the length of the longest consecutive sequence in a sorted game."""
    max_len = 1
    current_len = 1
    for i in range(1, len(sorted_game)):
        if sorted_game[i] == sorted_game[i - 1] + 1:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 1
    return max_len


def _select_diverse(scored_candidates, num_games, min_hamming=3):
    """
    From a list of (game, score) tuples sorted by score desc,
    greedily select `num_games` ensuring minimum Hamming distance.
    """
    if not scored_candidates:
        return []

    def hamming(a, b):
        return len(set(a).symmetric_difference(set(b)))

    selected = [scored_candidates[0]]

    for game, score in scored_candidates[1:]:
        if len(selected) >= num_games:
            break
        min_dist = min(hamming(game, sel[0]) for sel in selected)
        if min_dist >= min_hamming:
            selected.append((game, score))

    # If not enough diverse games, fill with remaining top-scored
    if len(selected) < num_games:
        for game, score in scored_candidates:
            if len(selected) >= num_games:
                break
            if (game, score) not in selected:
                selected.append((game, score))

    return selected[:num_games]

def explain_game(game, results):
    """Generates a textual explanation for a given game based on historical data."""
    game_set = set(game)
    balls_list = extract_balls(results)
    
    evens = sum(1 for n in game if n % 2 == 0)
    odds = 15 - evens
    game_sum = sum(game)
    primes = sum(1 for n in game if n in PRIME_NUMBERS)
    
    explanation = f"{evens} pares / {odds} ímpares; soma {game_sum}; {primes} primos"
    
    if balls_list:
        repeats = len(game_set & set(balls_list[0]))
        explanation += f"; {repeats} repetidas do último"
        
    return explanation

def generate_frequency_based(results, num_games=6, hot=5, neutral=6, cold=4):
    """Generates games by composing hot, neutral, and cold numbers based on all-time frequency."""
    balls_list = extract_balls(results)
    if not balls_list:
        return [sorted(random.sample(range(1, 26), 15)) for _ in range(num_games)]
        
    freq_counter = Counter()
    for draw in balls_list:
        freq_counter.update(draw)
        
    ranked = [n for n, _ in freq_counter.most_common()]
    for n in range(1, 26):
        if n not in ranked:
            ranked.append(n)
            
    hot_pool = ranked[:8]
    neutral_pool = ranked[8:17]
    cold_pool = ranked[17:]
    
    games = []
    for _ in range(num_games):
        h = random.sample(hot_pool, min(hot, len(hot_pool)))
        n = random.sample(neutral_pool, min(neutral, len(neutral_pool)))
        c = random.sample(cold_pool, min(cold, len(cold_pool)))
        game = sorted(h + n + c)
        
        while len(game) < 15:
            r = random.randint(1, 25)
            if r not in game:
                game.append(r)
                game.sort()
        
        games.append(game)
    return games

def generate_delay_based(results, num_games=6, mode='most_delayed'):
    """Generates games focusing on the most delayed numbers."""
    balls_list = extract_balls(results)
    if not balls_list:
        return [sorted(random.sample(range(1, 26), 15)) for _ in range(num_games)]
        
    delays = {}
    for num in range(1, 26):
        for i, draw in enumerate(balls_list):
            if num in draw:
                delays[num] = i
                break
        else:
            delays[num] = len(balls_list)
            
    ranked = sorted(range(1, 26), key=lambda x: delays[x], reverse=True)
    
    games = []
    for _ in range(num_games):
        if mode == 'most_delayed':
            core = ranked[:8]
            rest = random.sample(ranked[8:], 7)
            games.append(sorted(core + rest))
        else:
            games.append(sorted(random.sample(range(1, 26), 15)))
    return games
    
def generate_repetition_based(results, num_games=6, min_repeat=8, max_repeat=10):
    """Generates games that repeat a specific amount of numbers from the last draw."""
    balls_list = extract_balls(results)
    if not balls_list:
        return [sorted(random.sample(range(1, 26), 15)) for _ in range(num_games)]
        
    last_draw = list(balls_list[0])
    pool = [n for n in range(1, 26) if n not in last_draw]
    
    games = []
    for _ in range(num_games):
        rep = random.randint(min_repeat, max_repeat)
        core = random.sample(last_draw, rep)
        rest = random.sample(pool, 15 - rep)
        games.append(sorted(core + rest))
        
    return games
