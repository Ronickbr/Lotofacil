import MySQLdb
from config import Config

def migrate():
    print("Starting migrations...")
    db = MySQLdb.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        passwd=Config.MYSQL_PASSWORD,
        db=Config.MYSQL_DB
    )
    cur = db.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS generations (
        id VARCHAR(50) PRIMARY KEY,
        target_contest INT,
        created_at DATETIME,
        model_version VARCHAR(50),
        strategy VARCHAR(100),
        num_games INT,
        status VARCHAR(50) DEFAULT 'AGUARDANDO_RESULTADO',
        eval_metrics JSON
    )
    """)
    
    # Try altering saved_games safely
    try:
        cur.execute("ALTER TABLE saved_games ADD COLUMN generation_id VARCHAR(50)")
    except Exception as e: print(e)
    try:
        cur.execute("ALTER TABLE saved_games ADD COLUMN target_contest INT")
    except Exception as e: print(e)
    try:
        cur.execute("ALTER TABLE saved_games ADD COLUMN locked_at DATETIME")
    except Exception as e: print(e)
    try:
        cur.execute("ALTER TABLE saved_games ADD COLUMN hits INT")
    except Exception as e: print(e)
    
    cur.execute("""
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
    )
    """)
    
    cur.execute("""
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
    )
    """)
    
    db.commit()
    cur.close()
    db.close()
    print("Migrations applied.")

if __name__ == '__main__':
    migrate()
