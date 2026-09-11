import random

import MySQLdb
from flask import Blueprint, current_app, jsonify, render_template, request
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
    get_game_hash,
    explain_game,
)
from analysis.scoring import GameScorer

ml_bp = Blueprint('ml', __name__)


def _integer_field(name, default, minimum, maximum):
    """Read and validate a bounded integer form field."""
    raw_value = request.form.get(name, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'O campo {name} deve ser um número inteiro.') from exc
    if value < minimum or value > maximum:
        raise ValueError(f'O campo {name} deve estar entre {minimum} e {maximum}.')
    return value


def _parse_numbers(raw_value, minimum_count, maximum_count):
    """Parse comma-separated Lotofácil numbers and enforce its domain rules."""
    try:
        numbers = [int(value.strip()) for value in (raw_value or '').split(',') if value.strip()]
    except ValueError as exc:
        raise ValueError('As dezenas devem conter apenas números inteiros.') from exc

    if not minimum_count <= len(numbers) <= maximum_count:
        if minimum_count == maximum_count:
            raise ValueError(f'Forneça exatamente {minimum_count} dezenas.')
        raise ValueError(f'Forneça entre {minimum_count} e {maximum_count} dezenas.')
    if len(set(numbers)) != len(numbers):
        raise ValueError('As dezenas não podem se repetir.')
    if any(number < 1 or number > 25 for number in numbers):
        raise ValueError('Todas as dezenas devem estar entre 1 e 25.')
    return sorted(numbers)


def _prediction_html(valid_numbers, games, warning=None):
    """Build the HTML fragment consumed by the existing prediction page."""
    response = ''
    if warning:
        response += (
            "<div class='alert alert-warning rounded-4 shadow-sm mb-4'>"
            "<i class='fa-solid fa-triangle-exclamation me-2'></i>"
            f"{warning}</div>"
        )
    response += "<div class='mb-4 p-3 rounded-4 bg-light border'>"
    response += (
        "<h5 class='fw-bold mb-3 text-dark d-flex align-items-center'><i"
        " class='fa-solid fa-star text-warning me-2 fs-4'></i>Dezenas Mais Prováveis (Top 10):</h5>"
    )
    response += "<div class='d-flex flex-wrap gap-2'>"
    for num in valid_numbers:
        response += f"<span class='lottery-ball'>{num:02d}</span>"
    response += "</div></div>"
    response += (
        "<h5 class='fw-bold mb-3 text-dark d-flex align-items-center'><i"
        " class='fa-solid fa-ticket text-danger me-2 fs-4'></i>Bilhetes Sugeridos para Aposta (6 Jogos):</h5>"
        "<div class='row g-3'>"
    )
    for index, game in enumerate(games, 1):
        game_str = ' '.join(f'{number:02d}' for number in game)
        even_count = sum(1 for number in game if number % 2 == 0)
        response += f"""
        <div class='col-md-6'>
            <div class='ticket-card p-3 h-100 d-flex flex-column justify-content-between'>
                <div class='d-flex justify-content-between align-items-center mb-2 pb-2 border-bottom'>
                    <span class='fw-bold text-dark fs-6'><i class='fa-solid fa-clover text-warning me-2'></i>Bilhete {index:02d}</span>
                    <span class='badge bg-light text-muted border'>{even_count}P / {15 - even_count}Í</span>
                </div>
                <div class='d-flex flex-wrap gap-1 my-2 justify-content-center'>
        """
        for number in game:
            ball_class = 'ball-even' if number % 2 == 0 else 'ball-odd'
            response += f"<span class='lottery-ball lottery-ball-sm {ball_class}'>{number:02d}</span>"
        response += f"""
                </div>
                <div class='mt-2 pt-2 border-top text-end'>
                    <button class='btn btn-sm rounded-pill px-3 fw-bold' style='color:#7b2cbf;border:1px solid #7b2cbf' onclick='navigator.clipboard.writeText("{game_str}")'>
                        <i class='fa-regular fa-copy me-1'></i>Copiar Jogo
                    </button>
                </div>
            </div>
        </div>
        """
    return response + '</div>'

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
        warning = None
        try:
            model = load_lotofacil_model('lotofacil_model.pkl')
        except FileNotFoundError:
            model = None
            warning = (
                'O modelo de IA não está disponível ou está incompatível. '
                'Esta previsão usa o fallback estatístico; retreine o modelo para reativar a IA.'
            )

        try:
            data = _fetch_all_results()
            if not data:
                return 'Nenhum resultado encontrado na base de dados.', 404
            valid_numbers = predict_next_numbers(model, data, top_k=10)
            games = generate_suggested_games(valid_numbers, num_games=6)
            return _prediction_html(valid_numbers, games, warning)
        except Exception:
            current_app.logger.exception('Prediction failed')
            return 'Não foi possível gerar a previsão. Verifique os dados e tente novamente.', 500

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
            maximum_games = 2000 if strategy in {'combinatorial_unfold', 'reduced_closure'} else 50
            num_games = _integer_field('num_games', 6, 1, maximum_games)
            ai_ranking = None

            if strategy == 'direto':
                parsed = _parse_numbers(request.form.get('numbers'), 15, 15)
                games = [sorted(parsed)]

            elif strategy == 'combinatorial_unfold':
                parsed = _parse_numbers(request.form.get('numbers'), 16, 20)
                games = generate_combinations(parsed, k=15, max_games=num_games)

            elif strategy == 'reduced_closure':
                parsed = _parse_numbers(request.form.get('numbers'), 16, 20)
                guarantee = _integer_field('guarantee', 14, 11, 15)
                condition = _integer_field('condition', 15, 11, 15)
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

            else:
                return jsonify({'error': f'Estratégia desconhecida: {strategy}'}), 400

            scorer = GameScorer(results)
            response_games = []
            for game_data in games:
                numbers = game_data['numbers'] if isinstance(game_data, dict) else game_data
                numbers = sorted(int(number) for number in numbers)
                score_detail = scorer.score_game(numbers)
                response_games.append({
                    'hash': get_game_hash(numbers),
                    'numbers': numbers,
                    'explanation': explain_game(numbers, results),
                    'total_score': score_detail['total_score'],
                    'components': score_detail['components'],
                    'game_sum': score_detail['game_sum'],
                    'evens': score_detail['evens'],
                    'odds': score_detail['odds'],
                    'primes': score_detail['primes'],
                })

            total_generated = len(response_games)
            return jsonify({
                'strategy': strategy,
                'games': response_games[:100],
                'total_generated': total_generated,
                'cost': total_generated * 3.00,
                'ai_ranking': ai_ranking,
            })
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        except Exception:
            current_app.logger.exception('Smart game generation failed')
            return jsonify({'error': 'Não foi possível gerar os jogos. Tente novamente.'}), 500
    
    return render_template('smart_generate.html')


@ml_bp.route('/mlops')
def mlops():
    import json

    cur = None
    try:
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute('SELECT * FROM model_performance ORDER BY evaluated_contest DESC LIMIT 20')
        model_perf = cur.fetchall()
        cur.execute('SELECT * FROM generations ORDER BY created_at DESC LIMIT 50')
        generations = cur.fetchall()
        for generation in generations:
            try:
                generation['metrics_obj'] = json.loads(generation['eval_metrics']) if generation['eval_metrics'] else {}
            except (TypeError, json.JSONDecodeError):
                generation['metrics_obj'] = {}
        return render_template('continuous_learning.html', model_perf=model_perf, generations=generations)
    except Exception:
        current_app.logger.exception('MLOps dashboard failed')
        return render_template(
            'continuous_learning.html',
            model_perf=[],
            generations=[],
            error='Não foi possível carregar os dados de aprendizado contínuo.',
        )
    finally:
        if cur is not None:
            cur.close()
