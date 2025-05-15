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