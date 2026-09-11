from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from flask_mysqldb import MySQL
import MySQLdb
from config import Config

from analysis import (
    extract_balls,
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
    train_lotofacil_model,
    load_lotofacil_model,
    predict_next_numbers,
    generate_suggested_games,
    analyze_spatial_distribution,
    analyze_consecutive_sequences,
    calculate_correlation_matrix,
    find_top_pairs,
    find_top_triples,
    calculate_shannon_entropy,
    calculate_hypergeometric,
    calculate_hypergeometric_table,
    detect_mean_regression,
    monte_carlo_generate,
    genetic_algorithm_generate,
    hamming_distance_optimize,
    calculate_game_score,
    rank_games,
    select_diverse_top_games,
    generate_combinations,
    generate_reduced_closure,
    get_game_hash,
    explain_game,
    generate_frequency_based,
    generate_repetition_based,
)
from analysis.mlops import check_and_evaluate_generations, continuous_training_pipeline
import threading



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
                'percentage': score * 100 * 15,
            }
            for num, score in sorted_combined[:10]
        ]

    elif prediction_type == 'delay':
        stats['method_name'] = 'Análise de Atraso Completa'
        full_delays = calculate_full_delays(results)
        # Sort by current delay descending (most overdue)
        sorted_delays = sorted(full_delays.items(), key=lambda x: x[1]['current'], reverse=True)
        stats['frequent_numbers'] = [
            {
                'number': num,
                'count': data['current'],
                'percentage': data['percentile'],
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
        # Top numbers are most frequent overall
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
        # Sort by absolute z-score
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


app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)
app.config.from_object(Config)
mysql = MySQL(app)

_db_initialized = False

@app.before_request
def setup_db():
    global _db_initialized
    if not _db_initialized:
        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT 1")
            mysql.connection.commit()
            cur.close()
            _db_initialized = True
        except Exception as e:
            print(f"Database setup failed: {e}")


@app.context_processor
def inject_now():
    return {'now': datetime.now()}




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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add-concurso', methods=['POST'])
def add_concurso():
    try:
        cur = mysql.connection.cursor()

        concurso = request.form.get('concurso')
        data = request.form.get('data')
        bolas = [request.form.get(f'bola{i}') for i in range(1, 16)]

        if not all([concurso, data] + bolas):
            flash('Todos os campos são obrigatórios!', 'danger')
            return redirect(url_for('upload'))

        cur.execute("SELECT concurso FROM results WHERE concurso = %s", (concurso,))
        if cur.fetchone():
            flash('Concurso já existe!', 'danger')
            return redirect(url_for('upload'))

        query = """
            INSERT INTO results (concurso, data_sorteio, bola1, bola2, bola3, bola4, bola5,
                               bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13,
                               bola14, bola15)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = [concurso, data] + bolas
        cur.execute(query, values)
        mysql.connection.commit()
        check_and_evaluate_generations(mysql)
        flash('Concurso adicionado com sucesso e jogos avaliados!', 'success')

    except Exception as e:
        mysql.connection.rollback()
        flash(f'Erro ao adicionar concurso: {str(e)}', 'danger')
    finally:
        cur.close()

    return redirect(url_for('upload'))


@app.route('/delete-concurso/<int:concurso>', methods=['DELETE'])
def delete_concurso(concurso):
    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM results WHERE concurso = %s", (concurso,))
        mysql.connection.commit()
        return jsonify({'success': True})
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cur.close()


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    cur = None

    if request.method == 'POST':
        arquivo = request.files.get('file')

        if not arquivo or not arquivo.filename:
            flash('Selecione um arquivo Excel para importar.', 'danger')
            return redirect(url_for('upload'))

        if not arquivo.filename.lower().endswith('.xlsx'):
            flash('Formato inválido. Envie um arquivo Excel .xlsx.', 'danger')
            return redirect(url_for('upload'))

        try:
            df = pd.read_excel(arquivo, engine='openpyxl')
            df.columns = [str(col).strip().lower() for col in df.columns]

            colunas_esperadas = ['concurso', 'data'] + [f'bola{i}' for i in range(1, 16)]
            colunas_faltantes = [col for col in colunas_esperadas if col not in df.columns]
            if colunas_faltantes:
                raise ValueError('Colunas ausentes no Excel: ' + ', '.join(colunas_faltantes))

            df = df[colunas_esperadas].dropna(how='all')
            df = df.drop_duplicates(subset=['concurso'], keep='first')

            registros = []
            erros = []
            for numero_linha, (_, linha) in enumerate(df.iterrows(), start=2):
                try:
                    concurso = int(linha['concurso'])
                    data_sorteio = pd.to_datetime(linha['data'], dayfirst=True, errors='coerce')
                    bolas = [int(linha[f'bola{i}']) for i in range(1, 16)]

                    if pd.isna(data_sorteio):
                        raise ValueError('data inválida')
                    if len(set(bolas)) != 15 or any(bola < 1 or bola > 25 for bola in bolas):
                        raise ValueError('as bolas devem ser 15 números únicos entre 1 e 25')

                    registros.append([concurso, data_sorteio.strftime('%Y-%m-%d')] + bolas)
                except (TypeError, ValueError, OverflowError) as erro:
                    erros.append(f'linha {numero_linha}: {erro}')

            if not registros:
                raise ValueError('nenhum registro válido foi encontrado no arquivo')

            cur = mysql.connection.cursor()
            cur.execute('SELECT concurso FROM results')
            existentes = {int(row[0]) for row in cur.fetchall()}
            novos_registros = [registro for registro in registros if registro[0] not in existentes]

            if novos_registros:
                query = """
                    INSERT INTO results (
                        concurso, data_sorteio, bola1, bola2, bola3, bola4, bola5,
                        bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13,
                        bola14, bola15
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.executemany(query, novos_registros)
                mysql.connection.commit()
                check_and_evaluate_generations(mysql)

            mensagem = f'{len(novos_registros)} concurso(s) importado(s) de {len(registros)} registro(s) válido(s).'
            if erros:
                mensagem += f' {len(erros)} linha(s) foram ignoradas por conter dados inválidos.'
            if not novos_registros:
                mensagem = 'Todos os concursos válidos do arquivo já estavam cadastrados.'
            flash(mensagem, 'success')

        except Exception as e:
            mysql.connection.rollback()
            flash(f'Erro ao processar o arquivo Excel: {str(e)}', 'danger')
        finally:
            if cur is not None:
                cur.close()

        return redirect(url_for('upload'))

    try:
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("SELECT * FROM results ORDER BY concurso DESC")
        concursos = cur.fetchall()

        for concurso in concursos:
            if isinstance(concurso['data_sorteio'], str):
                concurso['data_sorteio'] = datetime.strptime(
                    concurso['data_sorteio'], '%Y-%m-%d'
                )

        return render_template('upload.html', concursos=concursos)
    except Exception as e:
        flash(f'Erro ao carregar concursos: {str(e)}', 'danger')
        return render_template('upload.html', concursos=[])
    finally:
        if cur is not None:
            cur.close()


@app.route('/dashboard')
def dashboard():
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 FROM results"
        )
        data = cur.fetchall()

        if not data:
            return render_template(
                'dashboard.html',
                error="Não há dados disponíveis. Faça upload de resultados primeiro.",
            )

        df = pd.DataFrame(data, columns=[f'bola{i}' for i in range(1, 16)])

        all_numbers = df.values.flatten()
        freq = pd.Series(all_numbers).value_counts()

        top_numbers = freq.head(5).index.tolist()
        frequencies = freq.head(5).values.tolist()

        even_count = (df % 2 == 0).sum().sum()
        odd_count = (df % 2 != 0).sum().sum()

        position_freq = df.apply(pd.Series.value_counts).fillna(0).astype(int)
        position_freq_html = position_freq.to_html(classes='table table-striped table-hover')

        return render_template(
            'dashboard.html',
            zip=zip,
            top_numbers=top_numbers,
            frequencies=frequencies,
            even_count=even_count,
            odd_count=odd_count,
            position_freq=position_freq_html,
        )

    except Exception as e:
        return render_template('dashboard.html', error=f"Erro ao carregar dados: {str(e)}")
    finally:
        if 'cur' in locals():
            cur.close()


@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            data = _fetch_all_results()

            if len(data) < 200:
                return jsonify({'error': "Erro: Dados insuficientes para treinar o modelo. É recomendado ter no mínimo 200 concursos."}), 400

            metrics = train_lotofacil_model(data, 'lotofacil_model.pkl')

            return jsonify(metrics)

        except Exception as e:
            return jsonify({'error': f"Erro ao treinar modelo: {str(e)}"}), 500

    return render_template('train_model.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            model = load_lotofacil_model('lotofacil_model.pkl')
        except FileNotFoundError as e:
            return str(e)

        data = _fetch_all_results()

        if not data:
            return "Nenhum resultado encontrado na base de dados."

        valid_numbers = predict_next_numbers(model, data, top_k=10)

        response = "<div class='mb-4 p-3 rounded-4 bg-light border'>"
        response += (
            "<h5 class='fw-bold mb-3 text-dark d-flex align-items-center'><i"
            " class='fa-solid fa-star text-warning me-2"
            " fs-4'></i>Dezenas Mais Prováveis (Top 10):</h5>"
        )
        response += "<div class='d-flex flex-wrap gap-2'>"
        for num in valid_numbers:
            response += f"<span class='lottery-ball'>{num:02d}</span>"
        response += "</div></div>"

        games = generate_suggested_games(valid_numbers, num_games=6)

        response += (
            "<h5 class='fw-bold mb-3 text-dark d-flex align-items-center'><i"
            " class='fa-solid fa-ticket text-danger me-2"
            " fs-4'></i>Bilhetes Sugeridos para Aposta (6 Jogos):</h5>"
        )
        response += "<div class='row g-3'>"
        for i, game in enumerate(games, 1):
            game_str = " ".join(f"{n:02d}" for n in game)
            even_count = sum(1 for n in game if n % 2 == 0)
            odd_count = 15 - even_count
            response += f"""
            <div class='col-md-6'>
                <div class='ticket-card p-3 h-100 d-flex flex-column justify-content-between'>
                    <div class='d-flex justify-content-between align-items-center mb-2 pb-2 border-bottom'>
                        <span class='fw-bold text-dark fs-6 d-flex align-items-center'><i class='fa-solid fa-clover text-warning me-2'></i>Bilhete {i:02d}</span>
                        <span class='badge bg-light text-muted border'>{even_count}P / {odd_count}Í</span>
                    </div>
                    <div class='d-flex flex-wrap gap-1 my-2 justify-content-center'>
            """
            for num in game:
                ball_class = "ball-even" if num % 2 == 0 else "ball-odd"
                response += f"<span class='lottery-ball lottery-ball-sm {ball_class}'>{num:02d}</span>"
            response += f"""
                    </div>
                    <div class='mt-2 pt-2 border-top text-end'>
                        <button class='btn btn-sm rounded-pill px-3 fw-bold' style='color: #7b2cbf; border: 1px solid #7b2cbf;' onclick='navigator.clipboard.writeText("{game_str}"); this.innerHTML="<i class=\\"fa-solid fa-check me-1\\"></i>Copiado!"; setTimeout(() => this.innerHTML="<i class=\\"fa-regular fa-copy me-1\\"></i>Copiar Jogo", 2000);'>
                            <i class='fa-regular fa-copy me-1'></i>Copiar Jogo
                        </button>
                    </div>
                </div>
            </div>
            """
        response += "</div>"

        return response

    return render_template('predict.html')


@app.route('/historical-stats')
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
            ORDER BY data_sorteio DESC
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


@app.route('/advanced-analysis')
def advanced_analysis():
    """Page with tabbed interface for all analysis types."""
    return render_template('advanced_analysis.html')


@app.route('/api/analysis/<analysis_type>')
def api_analysis(analysis_type):
    """JSON API for individual analysis types, consumed via AJAX."""
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
            # Convert int keys to string for JSON
            data = {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in data.items()}
        elif analysis_type == 'hot_cold':
            data = calculate_hot_cold(results)
        else:
            return jsonify({'error': f'Tipo de análise desconhecido: {analysis_type}'}), 400

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/smart-generate', methods=['GET', 'POST'])
def smart_generate():
    """Advanced game factory hub handling diverse strategies and combinatorial logic."""
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            results = _fetch_all_results()
            if not results:
                return jsonify({'error': 'Nenhum dado disponível'}), 404

            strategy = request.form.get('strategy', 'montecarlo')
            num_games = int(request.form.get('num_games', 6))
            
            # Allow larger limits for combinatorial unfoldings but cap output for UI safety
            if strategy not in ['combinatorial_unfold', 'reduced_closure']:
                num_games = max(1, min(num_games, 50))
            else:
                num_games = max(1, min(num_games, 2000))

            import random

            if strategy == 'direto':
                nums = request.form.get('numbers', '')
                parsed = [int(n.strip()) for n in nums.split(',') if n.strip().isdigit()]
                if len(parsed) != 15 or len(set(parsed)) != 15 or any(n < 1 or n > 25 for n in parsed):
                    return jsonify({'error': 'Forneça exatamente 15 dezenas únicas entre 1 e 25.'}), 400
                games = [sorted(parsed)]
                
            elif strategy == 'combinatorial_unfold':
                nums = request.form.get('numbers', '')
                parsed = sorted([int(n.strip()) for n in nums.split(',') if n.strip().isdigit()])
                if len(parsed) < 16 or len(parsed) > 20 or len(set(parsed)) != len(parsed):
                    return jsonify({'error': 'Forneça entre 16 e 20 dezenas únicas para desdobramento.'}), 400
                games = generate_combinations(parsed, k=15, max_games=num_games)
                
            elif strategy == 'reduced_closure':
                nums = request.form.get('numbers', '')
                parsed = sorted([int(n.strip()) for n in nums.split(',') if n.strip().isdigit()])
                guarantee = int(request.form.get('guarantee', 14))
                condition = int(request.form.get('condition', 15))
                if len(parsed) < 16 or len(parsed) > 20:
                    return jsonify({'error': 'Forneça entre 16 e 20 dezenas.'}), 400
                games = generate_reduced_closure(parsed, guarantee=guarantee, match_condition=condition, max_games=num_games)
                
            elif strategy == 'frequency':
                games = generate_frequency_based(results, num_games=num_games)
                
            elif strategy == 'delay':
                mode = request.form.get('delay_mode', 'most_delayed')
                games = generate_delay_based(results, num_games=num_games, mode=mode)
                
            elif strategy == 'repetition':
                games = generate_repetition_based(results, num_games=num_games)
                
            elif strategy == 'ai_based':
                try:
                    model = load_lotofacil_model('lotofacil_model.pkl')
                except FileNotFoundError:
                    return jsonify({'error': 'Modelo de IA não encontrado. Treine-o primeiro.'}), 400
                from analysis.ml import predict_next_numbers_detailed
                detailed = predict_next_numbers_detailed(model, results)
                top_nums = detailed['top18']
                ai_ranking = detailed['ranking']
                # Generate mixed from AI top 18
                candidates = [sorted(random.sample(top_nums, 15)) for _ in range(500)]
                diverse = hamming_distance_optimize(candidates, num_games=num_games)
                games = diverse

            elif strategy == 'montecarlo':
                filters = {
                    'min_even': int(request.form.get('min_even', 5)),
                    'max_even': int(request.form.get('max_even', 10)),
                    'max_consecutive': int(request.form.get('max_consecutive', 5)),
                    'min_repeat': int(request.form.get('min_repeat', 7)),
                }
                games = monte_carlo_generate(results, num_games=num_games, filters=filters)

            elif strategy == 'genetic':
                games = genetic_algorithm_generate(
                    results, num_games=num_games, population_size=200, generations=80
                )

            elif strategy == 'score':
                candidates = [sorted(random.sample(range(1, 26), 15)) for _ in range(5000)]
                scored = select_diverse_top_games(candidates, results, num_select=num_games)
                games = [{'numbers': g['numbers'], 'score': g['total_score']} for g in scored]

            elif strategy == 'hamming':
                candidates = [sorted(random.sample(range(1, 26), 15)) for _ in range(2000)]
                games = hamming_distance_optimize(candidates, num_games=num_games)

            else:
                return jsonify({'error': f'Estratégia desconhecida: {strategy}'}), 400

            # Score and enrich all games
            response_games = []
            
            # Precompute history object once for fast scoring!
            from analysis.scoring import GameScorer
            scorer = GameScorer(results)
            
            for game_data in games:
                nums = game_data['numbers'] if isinstance(game_data, dict) else game_data
                
                score_detail = scorer.score_game(nums)
                game_hash = get_game_hash(nums)
                explanation = explain_game(nums, results)
                
                response_games.append({
                    'hash': game_hash,
                    'numbers': nums,
                    'explanation': explanation,
                    'total_score': score_detail['total_score'],
                    'components': score_detail['components'],
                    'game_sum': score_detail['game_sum'],
                    'evens': score_detail['evens'],
                    'odds': score_detail['odds'],
                    'primes': score_detail['primes'],
                })
                
            # Limit returned games to 100 to prevent browser crash, but notify if more exist
            total_generated = len(response_games)
            returned_games = response_games[:100]

            return jsonify({
                'strategy': strategy,
                'games': returned_games,
                'total_generated': total_generated,
                'cost': total_generated * 3.00,
                'ai_ranking': locals().get('ai_ranking', None)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    return render_template('smart_generate.html')


@app.route('/save-games', methods=['POST'])
def save_games():
    if not _db_initialized:
        return jsonify({'error': 'Banco de dados não inicializado.'}), 500
        
    try:
        data = request.get_json()
        if not data or 'games' not in data:
            return jsonify({'error': 'Dados inválidos.'}), 400
            
        strategy = data.get('strategy', 'Unknown')
        games = data['games']
        ai_ranking = data.get('ai_ranking')
        
        # Determine target_contest
        results = _fetch_all_results()
        target_contest = (int(results[0][0]) + 1) if results else 1
        
        import uuid
        generation_id = f"GEN-{target_contest}-{uuid.uuid4().hex[:6].upper()}"
        
        cur = mysql.connection.cursor()
        saved_count = 0
        
        # 1. Insert Generation
        try:
            cur.execute("""
            INSERT INTO generations 
            (id, target_contest, created_at, model_version, strategy, num_games, status)
            VALUES (%s, %s, NOW(), %s, %s, %s, 'AGUARDANDO_RESULTADO')
            """, (generation_id, target_contest, 'current', strategy, len(games)))
        except Exception as e:
            print(f"Error saving generation: {e}")
            
        # 2. Insert AI Predictions if available
        if ai_ranking and isinstance(ai_ranking, list):
            for i, r in enumerate(ai_ranking):
                try:
                    cur.execute("""
                    INSERT INTO prediction_history 
                    (generation_id, target_contest, number, predicted_probability, ranking_position, model_version, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """, (generation_id, target_contest, r['dezena'], r['score'], i+1, 'current'))
                except Exception as e:
                    pass
        
        # 3. Insert Games
        for g in games:
            hash_val = g.get('hash')
            if not hash_val: continue
            
            balls_str = ",".join(map(str, g['numbers']))
            score = g.get('total_score', 0)
            import json
            details_json = json.dumps({
                'explanation': g.get('explanation', ''),
                'evens': g.get('evens', 0),
                'odds': g.get('odds', 0),
                'primes': g.get('primes', 0),
                'game_sum': g.get('game_sum', 0)
            })
            
            try:
                cur.execute("""
                INSERT IGNORE INTO saved_games 
                (created_at, strategy, balls, score, details, hash, generation_id, target_contest)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
                """, (strategy, balls_str, score, details_json, hash_val, generation_id, target_contest))
                if cur.rowcount > 0:
                    saved_count += 1
            except Exception as inner_e:
                print(f"Skipped saving game: {inner_e}")
                
        mysql.connection.commit()
        cur.close()
        
        return jsonify({
            'message': f'{saved_count} novos jogos salvos com sucesso (ignoradas duplicatas).',
            'generation_id': generation_id,
            'target_contest': target_contest
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/saved-games')
def saved_games():
    if not _db_initialized:
        flash('Banco de dados não pronto.', 'error')
        return redirect(url_for('index'))
        
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM saved_games ORDER BY created_at DESC LIMIT 500")
    saved = cur.fetchall()
    cur.close()
    
    results = _fetch_all_results()
    last_draw = set(results[0][:-1]) if results else set()
    
    import json
    for game in saved:
        balls = [int(n) for n in game['balls'].split(',')]
        game['numbers'] = balls
        if game['details']:
            game['details_obj'] = json.loads(game['details'])
        else:
            game['details_obj'] = {}
            
        if last_draw:
            game['hits_last_draw'] = len(set(balls) & last_draw)
        else:
            game['hits_last_draw'] = 0
            
    return render_template('saved_games.html', saved_games=saved, last_draw=list(last_draw))


@app.route('/export-excel', methods=['GET'])
def export_excel():
    conn = mysql.connection
    cur = conn.cursor()
    cur.execute(
        'SELECT concurso, data_sorteio, bola1, bola2, bola3, bola4, bola5,'
        'bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 '
        'FROM results ORDER BY concurso DESC'
    )
    results = cur.fetchall()
    cur.close()

    columns = [
        'Concurso',
        'Data',
        'Bola1',
        'Bola2',
        'Bola3',
        'Bola4',
        'Bola5',
        'Bola6',
        'Bola7',
        'Bola8',
        'Bola9',
        'Bola10',
        'Bola11',
        'Bola12',
        'Bola13',
        'Bola14',
        'Bola15',
    ]
    data = []
    for row in results:
        date_val = row[1]
        if date_val is None:
            date_str = 'N/A'
        elif isinstance(date_val, str):
            date_str = date_val
        else:
            date_str = date_val.strftime('%d/%m/%Y')

        bolas = list(row[2:17])

        data.append({
            'Concurso': row[0],
            'Data': date_str,
            'Bola1': bolas[0] if len(bolas) > 0 else None,
            'Bola2': bolas[1] if len(bolas) > 1 else None,
            'Bola3': bolas[2] if len(bolas) > 2 else None,
            'Bola4': bolas[3] if len(bolas) > 3 else None,
            'Bola5': bolas[4] if len(bolas) > 4 else None,
            'Bola6': bolas[5] if len(bolas) > 5 else None,
            'Bola7': bolas[6] if len(bolas) > 6 else None,
            'Bola8': bolas[7] if len(bolas) > 7 else None,
            'Bola9': bolas[8] if len(bolas) > 8 else None,
            'Bola10': bolas[9] if len(bolas) > 9 else None,
            'Bola11': bolas[10] if len(bolas) > 10 else None,
            'Bola12': bolas[11] if len(bolas) > 11 else None,
            'Bola13': bolas[12] if len(bolas) > 12 else None,
            'Bola14': bolas[13] if len(bolas) > 13 else None,
            'Bola15': bolas[14] if len(bolas) > 14 else None,
        })

    df = pd.DataFrame(data, columns=columns)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')

    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='resultados_lotofacil.xlsx',
    )


@app.route('/mlops')
def mlops():
    if not _db_initialized:
        flash('Banco de dados não inicializado.', 'error')
        return redirect(url_for('index'))
        
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # 1. Get Model Performance history
    cur.execute("SELECT * FROM model_performance ORDER BY evaluated_contest DESC LIMIT 20")
    model_perf = cur.fetchall()
    
    # 2. Get Generations
    cur.execute("SELECT * FROM generations ORDER BY created_at DESC LIMIT 50")
    generations = cur.fetchall()
    
    import json
    for g in generations:
        if g['eval_metrics']:
            g['metrics_obj'] = json.loads(g['eval_metrics'])
        else:
            g['metrics_obj'] = {}
            
    cur.close()
    return render_template('continuous_learning.html', model_perf=model_perf, generations=generations)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
