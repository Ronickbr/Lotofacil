import MySQLdb
from flask import Blueprint, render_template, request, jsonify
from extensions import mysql
from services.stats_service import _fetch_all_results
from analysis import (
    train_lotofacil_model,
    load_lotofacil_model,
    predict_next_numbers,
    generate_suggested_games,
    monte_carlo_generate,
    genetic_algorithm_generate,
    hamming_distance_optimize,
    generate_combinations,
    generate_reduced_closure,
    generate_frequency_based,
    generate_delay_based,
    generate_repetition_based,
)

ml_bp = Blueprint('ml', __name__)

@ml_bp.route('/train-model', methods=['GET', 'POST'])
def train_model():
    """
    Treina o modelo de Machine Learning (RandomForest).
    ---
    tags:
      - Machine Learning
    responses:
      200:
        description: JSON com as métricas do modelo treinado ou renderiza a página HTML se for requisição GET.
      400:
        description: Dados insuficientes para treinamento.
      500:
        description: Erro interno no servidor durante o treinamento.
    """
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


@ml_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Realiza previsões utilizando o modelo de Machine Learning treinado.
    ---
    tags:
      - Machine Learning
    responses:
      200:
        description: Retorna um HTML parcial contendo as dezenas mais prováveis e os bilhetes gerados ou renderiza a tela inteira para GET.
      404:
        description: Arquivo do modelo não encontrado.
    """
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
            " class='fa-solid fa-star text-warning me-2 fs-4'></i>Dezenas Mais Prováveis (Top 10):</h5>"
        )
        response += "<div class='d-flex flex-wrap gap-2'>"
        for num in valid_numbers:
            response += f"<span class='lottery-ball'>{num:02d}</span>"
        response += "</div></div>"

        games = generate_suggested_games(valid_numbers, num_games=6)

        response += (
            "<h5 class='fw-bold mb-3 text-dark d-flex align-items-center'><i"
            " class='fa-solid fa-ticket text-danger me-2 fs-4'></i>Bilhetes Sugeridos para Aposta (6 Jogos):</h5>"
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


@ml_bp.route('/smart-generate', methods=['GET', 'POST'])
def smart_generate():
    """
    Geração Inteligente de Jogos baseada em várias estratégias (IA, Frequência, Atraso, etc).
    ---
    tags:
      - Machine Learning
    parameters:
      - name: strategy
        in: formData
        type: string
        description: Estratégia utilizada (montecarlo, direto, combinatorial_unfold, reduced_closure, frequency, delay, repetition, ai_based, genetic).
      - name: num_games
        in: formData
        type: integer
        description: Quantidade de jogos a serem gerados.
    responses:
      200:
        description: Retorna JSON contendo a lista dos jogos gerados.
      400:
        description: Parâmetros de entrada inválidos para a estratégia selecionada.
      404:
        description: Nenhum dado disponível na base de dados para referenciar.
      500:
        description: Erro interno no processamento ou estratégia não implementada.
    """
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            results = _fetch_all_results()
            if not results:
                return jsonify({'error': 'Nenhum dado disponível'}), 404

            strategy = request.form.get('strategy', 'montecarlo')
            num_games = int(request.form.get('num_games', 6))

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
                if len(parsed) < 16 or len(parsed) > 20 or len(set(parsed)) != len(parsed):
                    return jsonify({'error': 'Forneça entre 16 e 20 dezenas únicas.'}), 400
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
                candidates = [sorted(random.sample(top_nums, 15)) for _ in range(500)]
                games = hamming_distance_optimize(candidates, num_games=num_games)

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

            return jsonify({'games': games})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('smart_generate.html')
@ml_bp.route('/mlops')
def mlops():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT * FROM model_performance ORDER BY evaluated_contest DESC LIMIT 20')
    model_perf = cur.fetchall()
    cur.execute('SELECT * FROM generations ORDER BY created_at DESC LIMIT 50')
    generations = cur.fetchall()
    import json
    for g in generations:
        if g['eval_metrics']:
            g['metrics_obj'] = json.loads(g['eval_metrics'])
        else:
            g['metrics_obj'] = {}
    cur.close()
    return render_template('continuous_learning.html', model_perf=model_perf, generations=generations)