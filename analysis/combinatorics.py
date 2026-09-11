"""
Combinatorial module for Lotofácil.
Handles exact mathematical unfoldings (Desdobramento), coverage optimization (Set Cover),
and hash generation for game validation.
"""
from itertools import combinations
import hashlib

def get_game_hash(game: list) -> str:
    """Generates a unique MD5 hash for a sorted list of numbers to prevent duplicates."""
    sorted_game = sorted(list(game))
    game_str = "-".join(map(str, sorted_game))
    return hashlib.md5(game_str.encode()).hexdigest()

def generate_combinations(pool: list, k: int = 15, max_games: int = 1000):
    """
    Generates exact mathematical combinations (Desdobramento Completo).
    Returns a list of combinations up to max_games to prevent memory exhaustion.
    """
    pool = sorted(list(set(pool)))
    if len(pool) < k:
        return []
        
    comb_iterator = combinations(pool, k)
    
    result = []
    for i, c in enumerate(comb_iterator):
        if i >= max_games:
            break
        result.append(list(c))
        
    return result

def count_combinations(n: int, k: int = 15) -> int:
    """Calculates C(n, k)."""
    import math
    if n < k: return 0
    return math.comb(n, k)

def greedy_set_cover(universe: list, subsets: list, num_games: int):
    """
    Greedy Set Cover Optimization.
    Selects up to `num_games` subsets that maximize the coverage of the `universe`.
    Used for Fechamento.
    """
    uncovered = set(universe)
    selected_subsets = []
    
    # Work with copies to not mutate originals, storing the original index
    candidates = [(i, set(s)) for i, s in enumerate(subsets)]
    
    for _ in range(num_games):
        if not uncovered:
            break
            
        # Find the subset that covers the most uncovered elements
        best_idx = -1
        best_cover_count = -1
        best_subset_elements = None
        
        for i, subset_elements in candidates:
            # How many uncovered items this subset covers
            cover_count = len(subset_elements & uncovered)
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_idx = i
                best_subset_elements = subset_elements
                
        if best_idx == -1 or best_cover_count == 0:
            break
            
        selected_subsets.append(subsets[best_idx])
        uncovered -= best_subset_elements
        
        # Remove the chosen candidate
        candidates = [c for c in candidates if c[0] != best_idx]
        
    return selected_subsets

def generate_reduced_closure(base_pool: list, guarantee: int, match_condition: int, k: int = 15, max_games: int = 100):
    """
    Generates a mathematically reduced closure (Fechamento Reduzido).
    Ensures that IF `match_condition` numbers from `base_pool` are drawn,
    we have AT LEAST ONE game with `guarantee` hits.
    
    To implement this efficiently in real-time, we use a randomized greedy set cover approach.
    """
    base_pool = sorted(list(set(base_pool)))
    if len(base_pool) < k:
        return []
        
    # Universe of possible outcomes we want to cover: 
    # all subsets of size `match_condition` from the `base_pool`
    # E.g., if match_condition = 15, universe size = C(len(base_pool), 15).
    # If base_pool is 18, universe size is 816.
    
    # Generate the universe
    universe_tuples = list(combinations(base_pool, match_condition))
    
    # We will build random candidates of size `k` (games)
    # A candidate game covers an outcome if the intersection size >= guarantee
    
    import random
    candidates = []
    for _ in range(max_games * 20): # Generate a large pool of candidate games
        cand = tuple(sorted(random.sample(base_pool, k)))
        candidates.append(cand)
    
    candidates = list(set(candidates)) # Unique candidates
    
    # Build the subsets for set cover
    # subset[i] = set of universe outcome indices covered by candidates[i]
    subsets_coverage = []
    for cand in candidates:
        cand_set = set(cand)
        covers = set()
        for idx, outcome in enumerate(universe_tuples):
            if len(cand_set & set(outcome)) >= guarantee:
                covers.add(idx)
        subsets_coverage.append(covers)
        
    # Use greedy set cover
    uncovered = set(range(len(universe_tuples)))
    selected_games = []
    
    available_indices = set(range(len(candidates)))
    
    for _ in range(max_games):
        if not uncovered:
            break
            
        best_idx = -1
        best_cover_count = -1
        
        for idx in available_indices:
            cover_count = len(subsets_coverage[idx] & uncovered)
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_idx = idx
                
        if best_idx == -1 or best_cover_count == 0:
            break
            
        selected_games.append(list(candidates[best_idx]))
        uncovered -= subsets_coverage[best_idx]
        available_indices.remove(best_idx)
        
    return selected_games
