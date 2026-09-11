from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, timedelta
from extensions import mysql
from services.stats_service import (
    calculate_statistics,
    _fetch_all_results,
    calculate_full_delays,
    analyze_spatial_distribution,
    analyze_consecutive_sequences,
    calculate_parity_distribution,
    calculate_sum_analysis,
    calculate_number_classes,
    calculate_shannon_entropy,
    calculate_correlation_matrix,
    find_top_pairs,
    find_top_triples,
    detect_mean_regression,
    calculate_windowed_frequency,
    calculate_hot_cold,
)
from analysis import calculate_hypergeometric_table

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/historical-stats')
def historical_stats():
    period = request.args.get('period')
    prediction_type = request.args.get('prediction_type', 'frequency')

    if not period:
        return render_template('historical_stats.html')

    cur = mysql.connection.cursor()

    today = datetime.now()
    if period == 'week':
        start_date = today - timedelta(days=7)
    elif period == 'month':
        start_date = today - timedelta(days=30)
    elif period == 'year':
        start_date = today - timedelta(days=365)
    elif period == 'all':
        start_date = datetime(2003, 1, 1)
    else:
        return render_template('historical_stats.html', error="Período inválido")

    try:
        cur.execute(
            """
            SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8,
                   bola9, bola10, bola11, bola12, bola13, bola14, bola15,
                   data_sorteio
            FROM results
            WHERE data_sorteio >= %s
            ORDER BY data_sorteio ASC
        """,
            (start_date.strftime('%Y-%m-%d'),),
        )

        results = cur.fetchall()

        if not results:
            return render_template(
                'historical_stats.html', error="Nenhum resultado encontrado para o período"
            )

        stats = calculate_statistics(results, prediction_type)

        return render_template(
            'historical_stats.html',
            stats=stats,
            period=period,
            prediction_type=prediction_type,
        )

    except Exception as e:
        return render_template('historical_stats.html', error=f"Erro na análise: {str(e)}")
    finally:
        cur.close()


@analysis_bp.route('/advanced-analysis')
def advanced_analysis():
    return render_template('advanced_analysis.html')


@analysis_bp.route('/api/analysis/<analysis_type>')
def api_analysis(analysis_type):
    """
    Retorna os dados estatísticos para um tipo de análise específico.
    ---
    tags:
      - Analysis
    parameters:
      - name: analysis_type
        in: path
        type: string
        required: true
        description: O tipo de análise a ser executada (ex. delays, spatial, consecutive, parity, sum, number_classes, entropy, correlation, regression, windowed, hypergeometric, hot_cold).
    responses:
      200:
        description: Dados da análise em formato JSON.
      400:
        description: Tipo de análise desconhecido.
      404:
        description: Nenhum dado disponível no banco de dados.
      500:
        description: Erro interno do servidor.
    """
    try:
        results = _fetch_all_results()
        if not results:
            return jsonify({'error': 'Nenhum dado disponível'}), 404

        if analysis_type == 'delays':
            data = calculate_full_delays(results)
        elif analysis_type == 'spatial':
            data = analyze_spatial_distribution(results)
        elif analysis_type == 'consecutive':
            data = analyze_consecutive_sequences(results)
        elif analysis_type == 'parity':
            data = calculate_parity_distribution(results)
        elif analysis_type == 'sum':
            data = calculate_sum_analysis(results)
        elif analysis_type == 'number_classes':
            data = calculate_number_classes(results)
        elif analysis_type == 'entropy':
            data = calculate_shannon_entropy(results)
        elif analysis_type == 'correlation':
            data = {
                'matrix': calculate_correlation_matrix(results),
                'top_pairs': find_top_pairs(results),
                'top_triples': find_top_triples(results),
            }
        elif analysis_type == 'regression':
            data = detect_mean_regression(results)
        elif analysis_type == 'windowed':
            data = calculate_windowed_frequency(results)
        elif analysis_type == 'hypergeometric':
            data = calculate_hypergeometric_table()
            data = {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in data.items()}
        elif analysis_type == 'hot_cold':
            data = calculate_hot_cold(results)
        else:
            return jsonify({'error': f'Tipo de análise desconhecido: {analysis_type}'}), 400

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
