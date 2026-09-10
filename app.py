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
    calculate_hot_cold,
    calculate_bayes_probabilities,
    analyze_consecutive_repeats,
    combine_analysis_methods,
    train_lotofacil_model,
    load_lotofacil_model,
    predict_next_numbers,
    generate_suggested_games,
)


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

    return stats


app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)
app.config.from_object(Config)


@app.context_processor
def inject_now():
    return {'now': datetime.now()}


mysql = MySQL(app)


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
        flash('Concurso adicionado com sucesso!', 'success')

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
        cur = mysql.connection.cursor()
        try:
            cur.execute(
                "SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 FROM results ORDER BY concurso ASC"
            )
            data = cur.fetchall()

            if len(data) < 2:
                return "Erro: Não há dados suficientes para treinar o modelo. Faça upload de mais dados."

            _, mean_accuracy = train_lotofacil_model(data, 'lotofacil_model.pkl')

            return f"Modelo treinado com sucesso! Acurácia média: {mean_accuracy:.2f}"

        except Exception as e:
            return f"Erro ao treinar modelo: {str(e)}"
        finally:
            cur.close()

    return render_template('train_model.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            model = load_lotofacil_model('lotofacil_model.pkl')
        except FileNotFoundError as e:
            return str(e)

        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 FROM results ORDER BY concurso DESC LIMIT 1"
        )
        last_result = cur.fetchone()
        cur.close()

        if not last_result:
            return "Nenhum resultado encontrado na base de dados."

        valid_numbers = predict_next_numbers(model, last_result, top_k=10)

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


if __name__ == '__main__':
    app.run(debug=True)
