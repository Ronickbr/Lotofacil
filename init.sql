CREATE DATABASE IF NOT EXISTS lotofacil;

USE lotofacil;

CREATE TABLE IF NOT EXISTS results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    concurso INT NOT NULL,
    data_sorteio DATE NOT NULL,
    bola1 INT NOT NULL,
    bola2 INT NOT NULL,
    bola3 INT NOT NULL,
    bola4 INT NOT NULL,
    bola5 INT NOT NULL,
    bola6 INT NOT NULL,
    bola7 INT NOT NULL,
    bola8 INT NOT NULL,
    bola9 INT NOT NULL,
    bola10 INT NOT NULL,
    bola11 INT NOT NULL,
    bola12 INT NOT NULL,
    bola13 INT NOT NULL,
    bola14 INT NOT NULL,
    bola15 INT NOT NULL,
    UNIQUE KEY (concurso)
);
CREATE TABLE IF NOT EXISTS uploads (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    upload_date DATETIME NOT NULL,
    records_count INT NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_games (
    id INT AUTO_INCREMENT PRIMARY KEY,
    created_at DATETIME,
    strategy VARCHAR(100),
    balls VARCHAR(100),
    score FLOAT,
    details JSON,
    hash VARCHAR(100) UNIQUE,
    generation_id VARCHAR(50),
    target_contest INT,
    locked_at DATETIME,
    hits INT
);

CREATE TABLE IF NOT EXISTS generations (
    id VARCHAR(50) PRIMARY KEY,
    target_contest INT,
    created_at DATETIME,
    model_version VARCHAR(50),
    strategy VARCHAR(100),
    num_games INT,
    status VARCHAR(50) DEFAULT 'AGUARDANDO_RESULTADO',
    eval_metrics JSON
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    generation_id VARCHAR(50),
    target_contest INT,
    number INT,
    predicted_probability FLOAT,
    ranking_position INT,
    actual_result INT DEFAULT NULL,
    model_version VARCHAR(50),
    created_at DATETIME
);

CREATE TABLE IF NOT EXISTS model_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_version VARCHAR(50),
    evaluated_contest INT,
    top15_hits INT,
    top18_hits INT,
    mean_game_hits FLOAT,
    best_game_hits INT,
    brier_score FLOAT,
    log_loss FLOAT,
    baseline_difference FLOAT,
    created_at DATETIME,
    status VARCHAR(50)
);