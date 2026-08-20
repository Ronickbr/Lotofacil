from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import random
import os
from flask_mysqldb import MySQL
import MySQLdb
from config import Config
from collections import Counter, defaultdict
from itertools import chain
from scipy import stats as scipy_stats
import math
from io import BytesIO
from werkzeug.utils import secure_filename

def chi_square_test(observed, expected):
    """Realiza teste qui-quadrado para análise de significância"""
    from scipy import stats as scipy_stats
    
    if len(observed) != len(expected):
        return 0.0, 1.0
        
    observed_array = list(observed)
    expected_array = list(expected)
    
    total = sum(observed_array)
    if total == 0:
        return 0.0, 1.0
        
    chi2 = sum((o - e)**2 / e for o, e in zip(observed_array, expected_array) if e > 0)
    df = len(observed_array) - 1
    
    try:
        p = 1.0 - scipy_stats.chi2.cdf(chi2, df) if df > 0 else 1.0
    except:
        p = 1.0
        
    return chi2, p

def analyze_temporal_patterns(results):
    """Analisa padrões por dia da semana, mês e dia do mês"""
    patterns = defaultdict(Counter)
    for result in results:
        date = result[-1]
        balls = list(result[:-1])
        
        dow = date.weekday()
        month = date.month
        day = date.day
        
        for num in balls:
            patterns[('dow', dow)][num] += 1
            patterns[('month', month)][num] += 1
            patterns[('day', day)][num] += 1
            
    return patterns

def detect_seasonal_patterns(results):
    """Detecta tendências sazonais e padrões de relevância estatística"""
    seasonal_patterns = {}
    
    # Análise mensal
    monthly_data = defaultdict(lambda: defaultdict(int))
    for result in results:
        date = result[-1]
        balls = list(result[:-1])
        month = date.month
        for num in balls:
            monthly_data[month][num] += 1
    
    # Análise por dia da semana
    dow_data = defaultdict(lambda: defaultdict(int))
    for result in results:
        date = result[-1]
        balls = list(result[:-1])
        dow = date.weekday()
        for num in balls:
            dow_data[dow][num] += 1
    
    # Identificar padrões mensais significativos
    for month in range(1, 13):
        counts = monthly_data[month]
        total = sum(counts.values())
        if total >= 10:  # Mínimo de dados
            expected = {n: total/25 for n in range(1, 26)}
            chi2, p_value = chi_square_test(
                [counts.get(n, 0) for n in range(1, 26)],
                [expected[n] for n in range(1, 26)]
            )
            significant = p_value < 0.05
            seasonal_patterns[f'month_{month}'] = {
                'numbers': sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10],
                'total': total,
                'chi2': chi2,
                'p_value': p_value,
                'significant': significant
            }
    
    # Identificar padrões diários significativos
    days = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    for dow in range(7):
        counts = dow_data[dow]
        total = sum(counts.values())
        if total >= 10:
            expected = {n: total/25 for n in range(1, 26)}
            chi2, p_value = chi_square_test(
                [counts.get(n, 0) for n in range(1, 26)],
                [expected[n] for n in range(1, 26)]
            )
            significant = p_value < 0.05
            seasonal_patterns[f'dow_{dow}'] = {
                'day_name': days[dow],
                'numbers': sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10],
                'total': total,
                'chi2': chi2,
                'p_value': p_value,
                'significant': significant
            }
    
    return seasonal_patterns

def detect_periodic_combinations(results):
    """Detecta combinações de números que ocorrem em períodos específicos"""
    period_combinations = defaultdict(Counter)
    
    for result in results:
        date = result[-1]
        balls = list(result[:-1])
        
        month = date.month
        dow = date.weekday()
        
        # Comprimir números para análise de padrão
        for i in range(len(balls)):
            for j in range(i+1, len(balls)):
                combo = tuple(sorted([balls[i], balls[j]]))
                period_combinations[f'month_{month}_combo'][combo] += 1
                period_combinations[f'dow_{dow}_combo'][combo] += 1
                
    return period_combinations

def temporal_likelihood_analysis(results, recent_games):
    """Calcula probabilidade baseada em análise temporal completa"""
    likelihood_scores = defaultdict(float)
    
    # Padrões temporais de curto prazo (últimos 10 jogos)
    short_term = results[-10:] if len(results) >= 10 else results
    short_term_counts = Counter()
    for result in short_term:
        balls = list(result[:-1])
        for num in balls:
            short_term_counts[num] += 1
    
    # Padrões temporais de médio prazo (últimos 50 jogos)
    medium_term = results[-50:] if len(results) >= 50 else results
    medium_counts = Counter()
    for result in medium_term:
        balls = list(result[:-1])
        for num in balls:
            medium_counts[num] += 1
    
    # Padrões sazonais
    seasonal = detect_seasonal_patterns(results)
    
    # Calcular pontuações baseadas em peso
    for num in range(1, 26):
        score = 0.0
        
        # Curto prazo: maior peso
        short_weight = 3.0 if short_term else 0.0
        score += short_term_counts.get(num, 0) * short_weight
        
        # Médio prazo: peso moderado
        medium_weight = 2.0 if medium_term else 0.0
        score += medium_counts.get(num, 0) * medium_weight
        
        # Temporal sazonal: peso alto
        seasonal_weight = 2.5
        for pattern_key, pattern_data in seasonal.items():
            for num_in_pattern, count in pattern_data.get('numbers', []):
                if num_in_pattern == num:
                    bonus = pattern_data.get('significant', False) and 2.0 or 1.0
                    score += count * seasonal_weight * bonus
        
        # Efeito de padrão específico: números que aparecem consistentemente
        if any(day_pattern != 'day' and num in [n for n, c in pattern_data.get('numbers', [])][:3]
               for pattern_key, pattern_data in seasonal.items()
               if pattern_key.startswith(('dow_', 'month_'))):
            score *= 1.5
        
        # Normalizar por jogos recentes
        total_games = max(len(results), 1)
        likelihood_scores[num] = score / total_games
    
    return likelihood_scores

def calculate_seasonal_importance(results):
    """Calcula importância sazonal para números"""
    seasonal_importance = defaultdict(list)
    
    # Agrupar por mês
    monthly_patterns = defaultdict(lambda: defaultdict(int))
    for result in results:
        date = result[-1]
        balls = list(result[:-1])
        month = date.month
        for num in balls:
            monthly_patterns[month][num] += 1
    
    # Calcular relevância mensal
    for month in range(1, 13):
        counts = monthly_patterns[month]
        total = sum(counts.values())
        if total >= 5:
            expected = {n: total/25 for n in range(1, 26)}
            chi2, p_value = chi_square_test(
                [counts.get(n, 0) for n in range(1, 26)],
                [expected[n] for n in range(1, 26)]
            )
            for num, count in counts.items():
                base_score = count / total
                significance_bonus = 1.5 if p_value < 0.05 else 1.0
                seasonal_importance[num].append({
                    'month': month,
                    'count': count,
                    'base_score': base_score,
                    'significance_bonus': significance_bonus,
                    'total_score': base_score * significance_bonus
                })
    
    return seasonal_importance

def enhanced_game_generation(likelihood_scores, seasonal_patterns, recent_games):
    """Gera jogos com base em probabilidades temporais avançadas"""
    enhanced_games = []
    
    # Escolher números com base em pesos temporais
    weighted_numbers = []
    for num in range(1, 26):
        base_weight = 1.0
        if likelihood_scores[num] > 0:
            base_weight += likelihood_scores[num] * 10
        
        # Adicionar peso sazonal
        for pattern_key, pattern_data in seasonal_patterns.items():
            if pattern_key.startswith('month_') and 'numbers' in pattern_data:
                for num_in_pattern, count in pattern_data['numbers'][:5]:
                    if num_in_pattern == num:
                        seasonal_bonus = 2.0 if pattern_data.get('significant') else 1.5
                        base_weight += seasonal_bonus
        
        weighted_numbers.extend([num] * int(base_weight))
    
    # Gerar jogos com balanço
    games_generated = 0
    attempts = 0
    max_attempts = 1000
    
    while games_generated < 6 and attempts < max_attempts:
        attempts += 1
        game = random.sample(weighted_numbers, 6) if len(weighted_numbers) >= 6 else random.sample(range(1, 26), 6)
        game_sorted = sorted(game)
        
        # Verificar se já existe no conjunto atual
        if game_sorted not in enhanced_games:
            # Calcular pontuação temporal
            game_score = sum(likelihood_scores[num] for num in game)
            enhanced_games.append((game_score, game_sorted))
            games_generated += 1
    
    # Ordenar por pontuação e retornar top 6
    enhanced_games.sort(key=lambda x: x[0], reverse=True)
    return [game for score, game in enhanced_games]

def calculate_temporal_probability(game, likelihood_scores, seasonal_patterns):
    """Calcula probabilidade de um jogo específico baseada em padrões temporais"""
    probability = 1.0
    game_nums = set(game)
    
    # Peso baseado em probabilidade temporal
    temporal_weight = sum(likelihood_scores[num] for num in game_nums)
    probability *= (temporal_weight / sum(likelihood_scores.values())) if sum(likelihood_scores.values()) > 0 else 0.5
    
    # Peso sazonal baseado em padrões mensais/dow
    seasonal_weight = 1.0
    for pattern_key, pattern_data in seasonal_patterns.items():
        if 'numbers' in pattern_data:
            pattern_nums = {num for num, count in pattern_data['numbers']}
            overlap = len(game_nums.intersection(pattern_nums))
            if overlap > 0:
                seasonal_bonus = 2.0 if pattern_data.get('significant') else 1.5
                seasonal_weight *= (1 + overlap * (seasonal_bonus - 1) / 6)
    
    probability *= seasonal_weight
    
    # Normalizar para 0-1
    max_possible = 25 * 3.0  # Estimativa aproximada
    probability = min(probability / max_possible, 1.0)
    
    return probability

def generate_temporal_seasonal_suggestions(results, model):
    """Gera sugestões com base em análise temporal e sazonal avançada"""
    # Obter dados
    cur = mysql.connection.cursor()
    cur.execute("SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15, data_sorteio FROM results ORDER BY concurso DESC LIMIT 50")
    recent_results = cur.fetchall()
    cur.execute("SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15, data_sorteio FROM results ORDER BY concurso ASC LIMIT 200")
    historical_results = cur.fetchall()
    cur.close()
    
    if not recent_results or not historical_results:
        return None
    
    # Análises temporais
    temporal_patterns = analyze_temporal_patterns(historical_results)
    seasonal_patterns = detect_seasonal_patterns(historical_results)
    likelihood_scores = temporal_likelihood_analysis(historical_results, recent_results)
    
    # Obter previsão do modelo
    last_result = recent_results[0]
    prediction = model.predict([last_result[:-1]])
    predicted_numbers = [int(num) for num in prediction.flatten()]
    
    # Combinar com análise temporal
    valid_numbers = sorted(list(set([num for num in predicted_numbers if 1 <= num <= 25])))
    seasonal_top = sorted([n for n, score in likelihood_scores.items()], reverse=True)[:20]
    valid_numbers += seasonal_top
    
    # Gerar jogos melhorados
    enhanced_games = enhanced_game_generation(likelihood_scores, seasonal_patterns, recent_results)
    
    # Calcular probabilidades
    game_probabilities = []
    for game in enhanced_games:
        prob = calculate_temporal_probability(game, likelihood_scores, seasonal_patterns)
        game_probabilities.append((prob, game))
    
    game_probabilities.sort(key=lambda x: x[0], reverse=True)
    
    return {
        'model_numbers': valid_numbers[:15],
        'temporal_numbers': [(n, likelihood_scores[n]) for n in seasonal_top[:10]],
        'enhanced_games': game_probabilities[:6],
        'seasonal_patterns': seasonal_patterns
    }

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)
app.config.from_object(Config)

# Inicializa o MySQL
mysql = MySQL(app)

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add-concurso', methods=['POST'])
def add_concurso():
    try:
        cur = mysql.connection.cursor()
        
        # Get form data
        concurso = request.form.get('concurso')
        data = request.form.get('data')
        bolas = [request.form.get(f'bola{i}') for i in range(1, 16)]
        
        # Validate data
        if not all([concurso, data] + bolas):
            flash('Todos os campos são obrigatórios!', 'danger')
            return redirect(url_for('upload'))
        
        # Check if concurso already exists
        cur.execute("SELECT concurso FROM results WHERE concurso = %s", (concurso,))
        if cur.fetchone():
            flash('Concurso já existe!', 'danger')
            return redirect(url_for('upload'))
        
        # Insert new concurso
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

# Upload de dados
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
            # Lê todas as linhas do arquivo, sem limite de quantidade.
            df = pd.read_excel(arquivo, engine='openpyxl')
            df.columns = [str(col).strip().lower() for col in df.columns]

            colunas_esperadas = ['concurso', 'data'] + [f'bola{i}' for i in range(1, 16)]
            colunas_faltantes = [col for col in colunas_esperadas if col not in df.columns]
            if colunas_faltantes:
                raise ValueError(
                    'Colunas ausentes no Excel: ' + ', '.join(colunas_faltantes)
                )

            # Mantém somente as colunas necessárias e remove linhas totalmente vazias.
            df = df[colunas_esperadas].dropna(how='all')
            df = df.drop_duplicates(subset=['concurso'], keep='first')

            registros = []
            erros = []
            for numero_linha, (_, linha) in enumerate(df.iterrows(), start=2):
                try:
                    concurso = int(linha['concurso'])
                    data_sorteio = pd.to_datetime(
                        linha['data'], dayfirst=True, errors='coerce'
                    )
                    bolas = [int(linha[f'bola{i}']) for i in range(1, 16)]

                    if pd.isna(data_sorteio):
                        raise ValueError('data inválida')
                    if len(set(bolas)) != 15 or any(bola < 1 or bola > 25 for bola in bolas):
                        raise ValueError('as bolas devem ser 15 números únicos entre 1 e 25')

                    registros.append(
                        [concurso, data_sorteio.strftime('%Y-%m-%d')] + bolas
                    )
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
        # Sem LIMIT: carrega todos os concursos cadastrados.
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

# Dashboard de estatísticas
@app.route('/dashboard')
def dashboard():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 FROM results")
        data = cur.fetchall()

        if not data:
            return render_template('dashboard.html', error="Não há dados disponíveis. Faça upload de resultados primeiro.")

        df = pd.DataFrame(data, columns=[f'bola{i}' for i in range(1, 16)])

        # Calcula os números mais frequentes
        all_numbers = df.values.flatten()
        freq = pd.Series(all_numbers).value_counts()

        # Pega os 5 números mais frequentes
        top_numbers = freq.head(5).index.tolist()
        frequencies = freq.head(5).values.tolist()

        # Pares e ímpares
        even_count = (df % 2 == 0).sum().sum()
        odd_count = (df % 2 != 0).sum().sum()

        # Frequência por posição
        position_freq = df.apply(pd.Series.value_counts).fillna(0).astype(int)
        position_freq_html = position_freq.to_html(classes='table table-striped table-hover')

        return render_template('dashboard.html',
                            zip=zip,  # Add this line
                            top_numbers=top_numbers,
                            frequencies=frequencies,
                            even_count=even_count,
                            odd_count=odd_count,
                            position_freq=position_freq_html)

    except Exception as e:
        return render_template('dashboard.html', error=f"Erro ao carregar dados: {str(e)}")
    finally:
        if 'cur' in locals():
            cur.close()

# Treinar modelo de Machine Learning
@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        cur = mysql.connection.cursor()
        try:
            cur.execute("SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 FROM results")
            data = cur.fetchall()

            if len(data) < 2:
                return "Erro: Não há dados suficientes para treinar o modelo. Faça upload de mais dados."

            df = pd.DataFrame(data, columns=[f'bola{i}' for i in range(1, 16)])

            # Cria matriz de características (X) e rótulos (y)
            X = df.iloc[:-1].values
            y = df.iloc[1:].values

            # Divide os dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Treina o modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Avalia o modelo
            y_pred = model.predict(X_test)

            # Calcula a acurácia por coluna
            accuracies = []
            for i in range(y_test.shape[1]):
                column_accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
                accuracies.append(column_accuracy)

            # Calcula a acurácia média
            mean_accuracy = sum(accuracies) / len(accuracies)

            # Salva o modelo
            joblib.dump(model, 'lotofacil_model.pkl')

            return f"Modelo treinado com sucesso! Acurácia média: {mean_accuracy:.2f}"
            
        except Exception as e:
            return f"Erro ao treinar modelo: {str(e)}"
        finally:
            cur.close()
            
    return render_template('train_model.html')

# Prever números e gerar jogos
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            model = joblib.load('lotofacil_model.pkl')
        except FileNotFoundError:
            return "Modelo não encontrado. Treine o modelo primeiro."

        cur = mysql.connection.cursor()
        cur.execute("SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 FROM results ORDER BY concurso DESC LIMIT 1")
        last_result = cur.fetchone()
        cur.close()

        if not last_result:
            return "Nenhum resultado encontrado na base de dados."

        # Faz a previsão
        prediction = model.predict([last_result])
        predicted_numbers = [int(num) for num in prediction.flatten()]
        
        # Filtra números válidos e remove duplicatas
        valid_numbers = sorted(list(set([num for num in predicted_numbers if 1 <= num <= 25])))[:10]
        
        # Formata a resposta HTML
        response = "<h5>Números mais prováveis:</h5>"
        response += "<div class='mb-4'>"
        for num in valid_numbers:
            response += f"<span class='badge bg-primary m-1'>{num}</span>"
        response += "</div>"
        
        # Gera os jogos
        games = []
        for i in range(6):
            remaining = [n for n in range(1, 26) if n not in valid_numbers]
            additional = random.sample(remaining, 15 - len(valid_numbers))
            game = sorted(valid_numbers + additional)
            games.append(game)
        
        # Adiciona os jogos à resposta
        response += "<h5>Jogos sugeridos:</h5><ul>"
        for i, game in enumerate(games, 1):
            response += f"<li class='game'>Jogo {i}: {', '.join(map(str, game))}</li>"
        response += "</ul>"
        
        return response
        
    return render_template('predict.html')

# Estatísticas históricas
@app.route('/historical-stats')
def historical_stats():
    period = request.args.get('period')
    prediction_type = request.args.get('prediction_type', 'frequency')  # Default to frequency analysis
    
    if not period:
        return render_template('historical_stats.html')
    
    cur = mysql.connection.cursor()
    
    # Define o intervalo de datas
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
        # Busca os jogos do período
        cur.execute("""
            SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, 
                   bola9, bola10, bola11, bola12, bola13, bola14, bola15,
                   data_sorteio 
            FROM results 
            WHERE data_sorteio >= %s 
            ORDER BY data_sorteio DESC
        """, (start_date.strftime('%Y-%m-%d'),))
        
        results = cur.fetchall()
        
        if not results:
            return render_template('historical_stats.html', error="Nenhum resultado encontrado para o período")
        
        stats = calculate_statistics(results, prediction_type)
        
        return render_template('historical_stats.html', 
                             stats=stats, 
                             period=period,
                             prediction_type=prediction_type)
                             
    except Exception as e:
        return render_template('historical_stats.html', error=f"Erro na análise: {str(e)}")
    finally:
        cur.close()

@app.route('/export-excel', methods=['GET'])
def export_excel():
    # Ler dados do banco de dados
    conn = mysql.connection
    cur = conn.cursor()
    cur.execute('SELECT concurso, data_sorteio, bola1, bola2, bola3, bola4, bola5,'
                'bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 '
                'FROM results ORDER BY concurso DESC')
    results = cur.fetchall()
    cur.close()

    # Criar DataFrame do pandas
    columns = ['Concurso', 'Data', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5',
               'Bola6', 'Bola7', 'Bola8', 'Bola9', 'Bola10', 'Bola11', 'Bola12',
               'Bola13', 'Bola14', 'Bola15']
    data = []
    for row in results:
        # Formatar data corretamente - pode vir como string ou datetime.date
        date_val = row[1]
        if date_val is None:
            date_str = 'N/A'
        elif isinstance(date_val, str):
            date_str = date_val
        else:
            date_str = date_val.strftime('%d/%m/%Y')
        
        # Formatar bolas sorteadas (bola1 a bola15)
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

    # Gerar arquivo Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='resultados_lotofacil.xlsx'
    )

def calculate_statistics(results, prediction_type):
    """Calcula estatísticas baseadas no tipo de previsão selecionado"""
    all_numbers = []
    total_even = 0
    total_odd = 0
    
    for result in results:
        numbers = list(result[:-1])  # Exclude data_sorteio
        all_numbers.extend(numbers)
        total_even += len([n for n in numbers if n % 2 == 0])
        total_odd += len([n for n in numbers if n % 2 != 0])
    
    total_games = len(results)
    number_freq = Counter(all_numbers)
    
    stats = {
        'total_games': total_games,
        'avg_even': total_even / total_games,
        'avg_odd': total_odd / total_games,
    }
    
    if prediction_type == 'frequency':
        # Análise de frequência simples
        stats['method_name'] = 'Análise de Frequência'
        stats['frequent_numbers'] = [
            {'number': num, 'count': count, 'percentage': (count/total_games) * 100}
            for num, count in number_freq.most_common(10)
        ]
        
    elif prediction_type == 'bayes':
        # Análise Bayesiana
        stats['method_name'] = 'Análise Bayesiana'
        prior_probs = {num: count/total_games for num, count in number_freq.items()}
        posterior_probs = calculate_bayes_probabilities(results, prior_probs)
        stats['frequent_numbers'] = [
            {'number': num, 'count': int(prob * total_games), 'percentage': prob * 100}
            for num, prob in sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
    elif prediction_type == 'pattern':
        # Análise de Padrões
        stats['method_name'] = 'Análise de Padrões'
        patterns = analyze_patterns(results)
        stats['frequent_numbers'] = [
            {'number': num, 'count': count, 'percentage': (count/total_games) * 100}
            for num, count in patterns.most_common(10)
        ]
        
    elif prediction_type == 'combined':
        # Análise Combinada
        stats['method_name'] = 'Análise Combinada'
        combined_analysis = combine_analysis_methods(results)
        stats['frequent_numbers'] = [
            {'number': num, 'count': score, 'percentage': (score/total_games) * 100}
            for num, score in sorted(combined_analysis.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    return stats

def calculate_bayes_probabilities(results, prior_probs):
    """Calcula probabilidades usando Teorema de Bayes"""
    posterior_probs = {}
    total_games = len(results)
    
    for num in range(1, 26):
        # Likelihood: P(B|A)
        occurrences = sum(1 for result in results if num in result)
        likelihood = occurrences / total_games
        
        # Prior: P(A)
        prior = prior_probs.get(num, 1/25)
        
        # Posterior: P(A|B) ∝ P(B|A) * P(A)
        posterior_probs[num] = likelihood * prior
    
    # Normalize probabilities
    total = sum(posterior_probs.values())
    return {k: v/total for k, v in posterior_probs.items()}

def analyze_patterns(results):
    """Analisa padrões nos resultados"""
    patterns = Counter()
    
    for i in range(len(results)-1):
        current = set(results[i][:-1])
        next_draw = set(results[i+1][:-1])
        
        # Identifica números que se repetem em sorteios consecutivos
        repeated = current & next_draw
        patterns.update(repeated)
    
    return patterns

def combine_analysis_methods(results):
    """Combina diferentes métodos de análise"""
    combined_scores = defaultdict(float)
    
    # Frequência básica
    number_freq = Counter(chain.from_iterable(result[:-1] for result in results))
    
    # Padrões
    patterns = analyze_patterns(results)
    
    # Probabilidades Bayesianas
    prior_probs = {num: count/len(results) for num, count in number_freq.items()}
    bayes_probs = calculate_bayes_probabilities(results, prior_probs)
    
    # Combina os scores com pesos
    for num in range(1, 26):
        combined_scores[num] = (
            0.4 * number_freq.get(num, 0) +
            0.3 * patterns.get(num, 0) +
            0.3 * bayes_probs.get(num, 0)
        )
    
    return combined_scores

if __name__ == '__main__':
    app.run(debug=True)
