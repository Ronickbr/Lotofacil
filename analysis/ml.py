"""
Machine Learning integration for Lotofácil predictions.
"""

import os
import random
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .stats import extract_balls


def train_lotofacil_model(results, model_path='lotofacil_model.pkl'):
    """
    Trains a RandomForest model to predict draw N+1 from draw N.
    """
    balls_list = extract_balls(results)
    if len(balls_list) < 2:
        raise ValueError("Dados insuficientes para treinamento (mínimo 2 concursos).")

    df = pd.DataFrame(balls_list, columns=[f'bola{i}' for i in range(1, 16)])

    # Feature matrix X (draw N) and Target y (draw N+1)
    X = df.iloc[:-1].values
    y = df.iloc[1:].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracies = []
    for i in range(y_test.shape[1]):
        col_acc = accuracy_score(y_test[:, i], y_pred[:, i])
        accuracies.append(col_acc)

    mean_accuracy = sum(accuracies) / len(accuracies)

    joblib.dump(model, model_path)
    return model, mean_accuracy


def load_lotofacil_model(model_path='lotofacil_model.pkl'):
    """
    Loads trained RandomForest model from pickle file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo não encontrado. Treine o modelo primeiro.")
    return joblib.load(model_path)


def predict_next_numbers(model, last_draw, top_k=10):
    """
    Uses trained model to predict top numbers for next draw given the last draw (15 integers).
    """
    if len(last_draw) > 15:
        balls = [int(b) for b in last_draw[:15]]
    else:
        balls = [int(b) for b in last_draw]

    prediction = model.predict([balls])
    predicted_flat = [int(num) for num in prediction.flatten()]

    valid_numbers = sorted(list(set([num for num in predicted_flat if 1 <= num <= 25])))
    return valid_numbers[:top_k]


def generate_suggested_games(valid_numbers, num_games=6):
    """
    Generates balanced Lotofácil games (15 numbers each out of 25),
    building around `valid_numbers` core.
    """
    games = []
    core_len = min(len(valid_numbers), 15)
    core = valid_numbers[:core_len]

    remaining_pool = [n for n in range(1, 26) if n not in core]
    needed = 15 - core_len

    for _ in range(num_games):
        if needed > 0 and len(remaining_pool) >= needed:
            additional = random.sample(remaining_pool, needed)
            game = sorted(core + additional)
        else:
            game = sorted(random.sample(range(1, 26), 15))

        games.append(game)

    return games
