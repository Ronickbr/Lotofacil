from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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
    if request.method == 'GET':
        try:
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute("SELECT * FROM results ORDER BY concurso DESC")
            concursos = cur.fetchall()
            
            # Convert date objects to datetime if needed
            for concurso in concursos:
                if isinstance(concurso['data_sorteio'], str):
                    concurso['data_sorteio'] = datetime.strptime(concurso['data_sorteio'], '%Y-%m-%d')
                
            return render_template('upload.html', concursos=concursos)
        except Exception as e:
            flash(f'Erro ao carregar concursos: {str(e)}', 'danger')
            return render_template('upload.html', concursos=[])
        finally:
            cur.close()

    return render_template('upload.html')

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