from extensions import mysql
from analysis import (
    calculate_basic_stats,
    calculate_delays,
    calculate_full_delays,
    calculate_hot_cold,
    calculate_windowed_frequency,
    calculate_parity_distribution,
    calculate_sum_analysis,
    calculate_number_classes,
    calculate_bayes_probabilities,
    analyze_consecutive_repeats,
    combine_analysis_methods,
    analyze_spatial_distribution,
    analyze_consecutive_sequences,
    calculate_correlation_matrix,
    find_top_pairs,
    find_top_triples,
    calculate_shannon_entropy,
    detect_mean_regression,
)

def _fetch_all_results():
    """Helper to fetch all results ordered by concurso ASC."""
    cur = mysql.connection.cursor()
    cur.execute(
        """SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8,
                  bola9, bola10, bola11, bola12, bola13, bola14, bola15,
                  data_sorteio
           FROM results ORDER BY concurso ASC"""
    )
    results = cur.fetchall()
    cur.close()
    return results

def _get_next_contest_number():
    """Return the next contest number based on the greatest contest stored in results."""
    cur = mysql.connection.cursor()
    try:
        cur.execute("SELECT MAX(concurso) FROM results")
        row = cur.fetchone()
        last_contest = int(row[0]) if row and row[0] is not None else 0
        return last_contest + 1
    finally:
        cur.close()

def _fetch_latest_result():
    """Return the latest draw as 15 balls plus draw date, or None when the DB is empty."""
    cur = mysql.connection.cursor()
    try:
        cur.execute(
            """SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8,
                      bola9, bola10, bola11, bola12, bola13, bola14, bola15,
                      data_sorteio
               FROM results
               ORDER BY concurso DESC
               LIMIT 1"""
        )
        return cur.fetchone()
    finally:
        cur.close()

def calculate_statistics(results, prediction_type):
    """Calcula estatísticas baseadas no tipo de previsão selecionado"""
    basic_stats = calculate_basic_stats(results)
    total_games = basic_stats['total_games']

    if total_games == 0:
        return {
            'total_games': 0,
            'avg_even': 0.0,
            'avg_odd': 0.0,
            'avg_primes': 0.0,
            'avg_sum': 0.0,
            'frequent_numbers': [],
            'method_name': 'Nenhum Dado',
        }

    delays = calculate_delays(results)
    hot_cold = calculate_hot_cold(results)

    stats = {
        'total_games': total_games,
        'avg_even': basic_stats['avg_even'],
        'avg_odd': basic_stats['avg_odd'],
        'avg_primes': basic_stats['avg_primes'],
        'avg_sum': basic_stats['avg_sum'],
        'delays': delays,
        'hot_cold': hot_cold,
    }

    if prediction_type == 'frequency':
        stats['method_name'] = 'Análise de Frequência'
        num_freq = basic_stats['number_frequencies']
        sorted_freq = sorted(num_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        stats['frequent_numbers'] = [
            {'number': num, 'count': data['count'], 'percentage': data['percentage']}
            for num, data in sorted_freq[:10]
        ]

    elif prediction_type == 'bayes':
        stats['method_name'] = 'Análise Bayesiana'
        bayes_probs = calculate_bayes_probabilities(results)
        sorted_bayes = sorted(bayes_probs.items(), key=lambda x: x[1], reverse=True)
        stats['frequent_numbers'] = [
            {
                'number': num,
                'count': int(round(prob * total_games)),
                'percentage': prob * 100,
            }
            for num, prob in sorted_bayes[:10]
        ]

    elif prediction_type == 'pattern':
        stats['method_name'] = 'Análise de Padrões'
        patterns = analyze_consecutive_repeats(results)
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        stats['frequent_numbers'] = [
            {
                'number': num,
                'count': count,
                'percentage': (count / max(total_games - 1, 1)) * 100,
            }
            for num, count in sorted_patterns[:10]
        ]

    elif prediction_type == 'combined':
        stats['method_name'] = 'Análise Combinada'
        combined_scores = combine_analysis_methods(results)
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        stats['frequent_numbers'] = [
            {
                'number': num,
                'count': round(score * total_games, 2),
                'percentage': score * 100,
            }
            for num, score in sorted_combined[:10]
        ]

    elif prediction_type == 'delay':
        stats['method_name'] = 'Análise de Atraso Completa'
        full_delays = calculate_full_delays(results)
        sorted_delays = sorted(full_delays.items(), key=lambda x: x[1]['current'], reverse=True)
        num_freq = basic_stats['number_frequencies']
        stats['frequent_numbers'] = [
            {
                'number': num,
                'count': data['current'],
                'percentage': num_freq[num]['percentage'],
            }
            for num, data in sorted_delays[:10]
        ]
        stats['full_delays'] = full_delays

    elif prediction_type == 'windowed':
        stats['method_name'] = 'Frequência por Janelas Temporais'
        windowed = calculate_windowed_frequency(results)
        sorted_windowed = sorted(windowed.items(), key=lambda x: x[1]['score'], reverse=True)
        stats['frequent_numbers'] = [
            {
                'number': num,
                'count': data['windows'].get('total', {}).get('count', 0),
                'percentage': data['score'] * 100,
            }
            for num, data in sorted_windowed[:10]
        ]
        stats['windowed_data'] = windowed

    elif prediction_type == 'parity':
        stats['method_name'] = 'Distribuição Par/Ímpar'
        parity = calculate_parity_distribution(results)
        stats['parity_distribution'] = parity
        num_freq = basic_stats['number_frequencies']
        sorted_freq = sorted(num_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        stats['frequent_numbers'] = [
            {'number': num, 'count': data['count'], 'percentage': data['percentage']}
            for num, data in sorted_freq[:10]
        ]

    elif prediction_type == 'sum_analysis':
        stats['method_name'] = 'Análise da Soma das Dezenas'
        sum_data = calculate_sum_analysis(results)
        stats['sum_analysis'] = sum_data
        num_freq = basic_stats['number_frequencies']
        sorted_freq = sorted(num_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        stats['frequent_numbers'] = [
            {'number': num, 'count': data['count'], 'percentage': data['percentage']}
            for num, data in sorted_freq[:10]
        ]

    elif prediction_type == 'spatial':
        stats['method_name'] = 'Distribuição Espacial no Volante'
        spatial = analyze_spatial_distribution(results)
        stats['spatial'] = spatial
        num_freq = basic_stats['number_frequencies']
        sorted_freq = sorted(num_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        stats['frequent_numbers'] = [
            {'number': num, 'count': data['count'], 'percentage': data['percentage']}
            for num, data in sorted_freq[:10]
        ]

    elif prediction_type == 'consecutive':
        stats['method_name'] = 'Análise de Sequências Consecutivas'
        consecutive = analyze_consecutive_sequences(results)
        stats['consecutive'] = consecutive
        num_freq = basic_stats['number_frequencies']
        sorted_freq = sorted(num_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        stats['frequent_numbers'] = [
            {'number': num, 'count': data['count'], 'percentage': data['percentage']}
            for num, data in sorted_freq[:10]
        ]

    elif prediction_type == 'primes':
        stats['method_name'] = 'Primos, Fibonacci e Classes Especiais'
        number_classes = calculate_number_classes(results)
        stats['number_classes'] = number_classes
        num_freq = basic_stats['number_frequencies']
        sorted_freq = sorted(num_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        stats['frequent_numbers'] = [
            {'number': num, 'count': data['count'], 'percentage': data['percentage']}
            for num, data in sorted_freq[:10]
        ]

    elif prediction_type == 'entropy':
        stats['method_name'] = 'Entropia de Shannon'
        entropy = calculate_shannon_entropy(results)
        stats['entropy'] = entropy
        num_freq = basic_stats['number_frequencies']
        sorted_freq = sorted(num_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        stats['frequent_numbers'] = [
            {'number': num, 'count': data['count'], 'percentage': data['percentage']}
            for num, data in sorted_freq[:10]
        ]

    elif prediction_type == 'correlation':
        stats['method_name'] = 'Correlação e Pares Frequentes'
        correlation = calculate_correlation_matrix(results)
        top_pairs = find_top_pairs(results)
        top_triples = find_top_triples(results)
        stats['correlation'] = correlation
        stats['top_pairs'] = top_pairs
        stats['top_triples'] = top_triples
        num_freq = basic_stats['number_frequencies']
        sorted_freq = sorted(num_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        stats['frequent_numbers'] = [
            {'number': num, 'count': data['count'], 'percentage': data['percentage']}
            for num, data in sorted_freq[:10]
        ]

    elif prediction_type == 'regression':
        stats['method_name'] = 'Regressão à Média'
        regression = detect_mean_regression(results)
        stats['regression'] = regression
        sorted_reg = sorted(regression.items(), key=lambda x: abs(x[1]['z_score']), reverse=True)
        stats['frequent_numbers'] = [
            {
                'number': num,
                'count': int(data['recent_freq']),
                'percentage': data['recent_freq'],
            }
            for num, data in sorted_reg[:10]
        ]

    return stats
