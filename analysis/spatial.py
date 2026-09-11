"""
Spatial distribution and consecutive sequence analysis for Lotofácil.
Analyzes the 5×5 board layout: rows, columns, quadrants, diagonals, border vs center.
"""

from collections import Counter
from .stats import extract_balls

# Lotofácil board layout (5×5 matrix)
BOARD = [
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9,  10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]

# Precomputed mappings
NUM_TO_ROW = {}
NUM_TO_COL = {}
for r, row in enumerate(BOARD):
    for c, num in enumerate(row):
        NUM_TO_ROW[num] = r
        NUM_TO_COL[num] = c

ROW_LABELS = ['Linha 1 (01-05)', 'Linha 2 (06-10)', 'Linha 3 (11-15)', 'Linha 4 (16-20)', 'Linha 5 (21-25)']
COL_LABELS = ['Coluna 1', 'Coluna 2', 'Coluna 3', 'Coluna 4', 'Coluna 5']

# Border numbers (perimeter of 5x5)
BORDER = {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}
CENTER = {7, 8, 9, 12, 13, 14, 17, 18, 19}

# Quadrants (2x2-ish divisions)
QUADRANT_TL = {1, 2, 3, 6, 7, 8, 11, 12, 13}      # Top-left
QUADRANT_TR = {3, 4, 5, 8, 9, 10, 13, 14, 15}       # Top-right
QUADRANT_BL = {11, 12, 13, 16, 17, 18, 21, 22, 23}  # Bottom-left
QUADRANT_BR = {13, 14, 15, 18, 19, 20, 23, 24, 25}  # Bottom-right

# Main diagonals
DIAGONAL_MAIN = {1, 7, 13, 19, 25}       # top-left to bottom-right
DIAGONAL_ANTI = {5, 9, 13, 17, 21}       # top-right to bottom-left


def analyze_spatial_distribution(results):
    """
    Comprehensive spatial analysis of draws on the 5×5 board.

    Returns per-draw averages and distributions for:
    - rows (0-4), columns (0-4)
    - border vs center count
    - quadrant distribution
    - diagonal coverage
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total == 0:
        return {}

    row_totals = [0] * 5
    col_totals = [0] * 5
    border_totals = 0
    center_totals = 0
    quad_totals = {'TL': 0, 'TR': 0, 'BL': 0, 'BR': 0}
    diag_main_total = 0
    diag_anti_total = 0

    row_distributions = [Counter() for _ in range(5)]
    col_distributions = [Counter() for _ in range(5)]
    border_distribution = Counter()

    for draw in balls_list:
        draw_set = set(draw)
        row_counts = [0] * 5
        col_counts = [0] * 5

        for num in draw:
            r = NUM_TO_ROW[num]
            c = NUM_TO_COL[num]
            row_counts[r] += 1
            col_counts[c] += 1

        for r in range(5):
            row_totals[r] += row_counts[r]
            row_distributions[r][row_counts[r]] += 1
        for c in range(5):
            col_totals[c] += col_counts[c]
            col_distributions[c][col_counts[c]] += 1

        b_count = len(draw_set & BORDER)
        c_count = len(draw_set & CENTER)
        border_totals += b_count
        center_totals += c_count
        border_distribution[b_count] += 1

        quad_totals['TL'] += len(draw_set & QUADRANT_TL)
        quad_totals['TR'] += len(draw_set & QUADRANT_TR)
        quad_totals['BL'] += len(draw_set & QUADRANT_BL)
        quad_totals['BR'] += len(draw_set & QUADRANT_BR)

        diag_main_total += len(draw_set & DIAGONAL_MAIN)
        diag_anti_total += len(draw_set & DIAGONAL_ANTI)

    # Build heatmap data (frequency of each cell being drawn)
    cell_freq = {}
    all_nums_flat = []
    for draw in balls_list:
        all_nums_flat.extend(draw)
    freq_counter = Counter(all_nums_flat)

    heatmap = []
    for r in range(5):
        row_data = []
        for c in range(5):
            num = BOARD[r][c]
            row_data.append({
                'number': num,
                'count': freq_counter.get(num, 0),
                'pct': round(freq_counter.get(num, 0) / total * 100, 1),
            })
        heatmap.append(row_data)

    return {
        'total_games': total,
        'rows': {
            'labels': ROW_LABELS,
            'averages': [round(t / total, 2) for t in row_totals],
        },
        'columns': {
            'labels': COL_LABELS,
            'averages': [round(t / total, 2) for t in col_totals],
        },
        'border_center': {
            'border_avg': round(border_totals / total, 2),
            'center_avg': round(center_totals / total, 2),
            'border_total_pool': len(BORDER),
            'center_total_pool': len(CENTER),
        },
        'quadrants': {
            'TL': round(quad_totals['TL'] / total, 2),
            'TR': round(quad_totals['TR'] / total, 2),
            'BL': round(quad_totals['BL'] / total, 2),
            'BR': round(quad_totals['BR'] / total, 2),
        },
        'diagonals': {
            'main_avg': round(diag_main_total / total, 2),
            'anti_avg': round(diag_anti_total / total, 2),
            'main_numbers': sorted(DIAGONAL_MAIN),
            'anti_numbers': sorted(DIAGONAL_ANTI),
        },
        'heatmap': heatmap,
    }


def analyze_consecutive_sequences(results):
    """
    Analyzes consecutive number sequences in each draw.

    For a draw like [1, 2, 3, 7, 8, 12, ...]:
    - blocks: [[1,2,3], [7,8]]
    - max_sequence: 3
    - num_pairs: count of consecutive pairs

    Returns aggregate stats:
    - avg/max/min block count per draw
    - avg/max longest sequence
    - distribution of max sequence lengths
    """
    balls_list = extract_balls(results)
    total = len(balls_list)
    if total == 0:
        return {}

    all_max_seq = []
    all_num_blocks = []
    all_num_pairs = []
    max_seq_distribution = Counter()

    for draw in balls_list:
        sorted_draw = sorted(draw)
        blocks = []
        current_block = [sorted_draw[0]]

        for i in range(1, len(sorted_draw)):
            if sorted_draw[i] == sorted_draw[i - 1] + 1:
                current_block.append(sorted_draw[i])
            else:
                if len(current_block) >= 2:
                    blocks.append(current_block)
                current_block = [sorted_draw[i]]

        if len(current_block) >= 2:
            blocks.append(current_block)

        max_seq = max((len(b) for b in blocks), default=1)
        num_pairs = sum(1 for i in range(len(sorted_draw) - 1) if sorted_draw[i + 1] == sorted_draw[i] + 1)

        all_max_seq.append(max_seq)
        all_num_blocks.append(len(blocks))
        all_num_pairs.append(num_pairs)
        max_seq_distribution[max_seq] += 1

    import numpy as np
    seq_arr = np.array(all_max_seq)
    blocks_arr = np.array(all_num_blocks)
    pairs_arr = np.array(all_num_pairs)

    return {
        'total_games': total,
        'max_sequence': {
            'avg': round(float(np.mean(seq_arr)), 2),
            'max': int(np.max(seq_arr)),
            'min': int(np.min(seq_arr)),
            'distribution': {int(k): v for k, v in sorted(max_seq_distribution.items())},
        },
        'blocks': {
            'avg': round(float(np.mean(blocks_arr)), 2),
            'max': int(np.max(blocks_arr)),
        },
        'consecutive_pairs': {
            'avg': round(float(np.mean(pairs_arr)), 2),
            'max': int(np.max(pairs_arr)),
        },
    }
