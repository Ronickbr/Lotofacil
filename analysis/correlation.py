"""
Correlation analysis for Lotofácil: co-occurrence matrix, top pairs, top triples.
"""

from collections import Counter
from itertools import combinations
from .stats import extract_balls


def calculate_correlation_matrix(results):
    """
    Builds a 25×25 co-occurrence matrix.
    C_ij = P(i AND j) - P(i) * P(j)

    Returns:
    - matrix: dict of {(i, j): correlation_value}
    - individual_probs: dict of {num: probability}
    - joint_probs: dict of {(i, j): probability}
    - top_positive: top 20 most positively correlated pairs
    - top_negative: top 20 most negatively correlated pairs
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total == 0:
        return {'matrix': {}, 'top_positive': [], 'top_negative': [], 'total': 0}

    # Individual probabilities P(i)
    individual_counts = Counter()
    for draw in balls_list:
        individual_counts.update(draw)

    individual_probs = {num: individual_counts.get(num, 0) / total for num in range(1, 26)}

    # Joint probabilities P(i, j)
    joint_counts = Counter()
    for draw in balls_list:
        draw_set = set(draw)
        for pair in combinations(sorted(draw_set), 2):
            joint_counts[pair] += 1

    # Correlation C_ij = P(i,j) - P(i)*P(j)
    matrix = {}
    for i in range(1, 26):
        for j in range(i + 1, 26):
            joint_p = joint_counts.get((i, j), 0) / total
            expected_p = individual_probs[i] * individual_probs[j]
            matrix[(i, j)] = round(joint_p - expected_p, 6)

    sorted_pairs = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
    top_positive = [
        {'pair': list(pair), 'correlation': corr, 'joint_count': joint_counts.get(pair, 0)}
        for pair, corr in sorted_pairs[:20]
    ]
    top_negative = [
        {'pair': list(pair), 'correlation': corr, 'joint_count': joint_counts.get(pair, 0)}
        for pair, corr in sorted_pairs[-20:]
    ]

    # Heatmap-friendly matrix (list of lists)
    heatmap = []
    for i in range(1, 26):
        row = []
        for j in range(1, 26):
            if i == j:
                row.append(0.0)
            elif i < j:
                row.append(matrix.get((i, j), 0.0))
            else:
                row.append(matrix.get((j, i), 0.0))
        heatmap.append(row)

    return {
        'top_positive': top_positive,
        'top_negative': top_negative,
        'heatmap': heatmap,
        'total': total,
    }


def find_top_pairs(results, top_n=20):
    """
    Finds the most frequently co-occurring pairs.
    C(25,2) = 300 possible pairs.
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total == 0:
        return []

    pair_counts = Counter()
    for draw in balls_list:
        for pair in combinations(sorted(set(draw)), 2):
            pair_counts[pair] += 1

    # Expected frequency for any pair in a fair lottery:
    # P(both i and j drawn) = C(23,13) / C(25,15) ≈ 0.3391
    from math import comb
    expected = comb(23, 13) / comb(25, 15)

    top_pairs = []
    for pair, count in pair_counts.most_common(top_n):
        freq = count / total
        top_pairs.append({
            'pair': list(pair),
            'count': count,
            'frequency': round(freq * 100, 2),
            'expected_pct': round(expected * 100, 2),
            'deviation': round((freq - expected) / expected * 100, 2),
        })

    return top_pairs


def find_top_triples(results, top_n=20):
    """
    Finds the most frequently co-occurring triples.
    C(25,3) = 2300 possible triples (expensive for large datasets).
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total == 0:
        return []

    triple_counts = Counter()
    for draw in balls_list:
        for triple in combinations(sorted(set(draw)), 3):
            triple_counts[triple] += 1

    from math import comb
    expected = comb(22, 12) / comb(25, 15)

    top_triples = []
    for triple, count in triple_counts.most_common(top_n):
        freq = count / total
        top_triples.append({
            'triple': list(triple),
            'count': count,
            'frequency': round(freq * 100, 2),
            'expected_pct': round(expected * 100, 2),
        })

    return top_triples
