import os
import json
import logging
import numpy as np
from datetime import datetime
from MySQLdb.cursors import DictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_brier_score(predictions_df):
    """
    Computes Brier Score: Mean Squared Error between predicted prob and actual binary outcome.
    predictions_df: list of dicts with 'predicted_probability' and 'actual_result'
    """
    if not predictions_df:
        return 0.0
    errors = [(p['predicted_probability'] - p['actual_result'])**2 for p in predictions_df]
    return np.mean(errors)

def check_and_evaluate_generations(mysql):
    """
    Checks for generations waiting for results.
    If the target_contest is now available in the DB, evaluate all its games and predictions.
    """
    logger.info("Checking for pending generations to evaluate...")
    cur = mysql.connection.cursor(DictCursor)
    
    try:
        # Find generations waiting for results
        cur.execute("SELECT * FROM generations WHERE status = 'AGUARDANDO_RESULTADO'")
        pending_gens = cur.fetchall()
        
        for gen in pending_gens:
            target_contest = gen['target_contest']
            
            # Check if this contest exists in results table
            cur.execute("SELECT * FROM results WHERE concurso = %s", (target_contest,))
            result_row = cur.fetchone()
            
            if not result_row:
                continue # Still waiting
                
            logger.info(f"Result for contest {target_contest} found! Evaluating generation {gen['id']}")
            
            # Extract winning numbers
            winning_balls = set()
            for i in range(1, 16):
                ball_val = result_row[f'bola{i}']
                if ball_val:
                    winning_balls.add(int(ball_val))
                    
            # 1. Evaluate games
            cur.execute("SELECT id, balls FROM saved_games WHERE generation_id = %s", (gen['id'],))
            games = cur.fetchall()
            
            game_hits_list = []
            for g in games:
                balls = set(int(x) for x in g['balls'].split(','))
                hits = len(balls & winning_balls)
                game_hits_list.append(hits)
                # Update game and lock it
                cur.execute("UPDATE saved_games SET hits = %s, locked_at = NOW() WHERE id = %s", (hits, g['id']))
                
            # 2. Evaluate prediction history (Calibrations)
            cur.execute("SELECT id, number, predicted_probability, ranking_position FROM prediction_history WHERE generation_id = %s", (gen['id'],))
            preds = cur.fetchall()
            
            pred_evals = []
            top15_hits = 0
            top18_hits = 0
            for p in preds:
                actual = 1 if p['number'] in winning_balls else 0
                pred_evals.append({'predicted_probability': p['predicted_probability'], 'actual_result': actual})
                
                # Update actual result
                cur.execute("UPDATE prediction_history SET actual_result = %s WHERE id = %s", (actual, p['id']))
                
                if p['ranking_position'] <= 15 and actual == 1:
                    top15_hits += 1
                if p['ranking_position'] <= 18 and actual == 1:
                    top18_hits += 1
                    
            brier = evaluate_brier_score(pred_evals) if pred_evals else 0.0
            
            # 3. Calculate generation metrics
            if game_hits_list:
                mean_hits = float(np.mean(game_hits_list))
                best_hits = int(np.max(game_hits_list))
            else:
                mean_hits = 0.0
                best_hits = 0
                
            eval_metrics = {
                'top15_hits': top15_hits,
                'top18_hits': top18_hits,
                'mean_game_hits': mean_hits,
                'best_game_hits': best_hits,
                'brier_score': float(brier)
            }
            
            # 4. Save into model_performance
            cur.execute("""
            INSERT INTO model_performance 
            (model_version, evaluated_contest, top15_hits, top18_hits, mean_game_hits, best_game_hits, brier_score, created_at, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), 'EVALUATED')
            """, (gen['model_version'], target_contest, top15_hits, top18_hits, mean_hits, best_hits, brier))
            
            # 5. Update generation status
            cur.execute("UPDATE generations SET status = 'AVALIADO', eval_metrics = %s WHERE id = %s", 
                        (json.dumps(eval_metrics), gen['id']))
                        
            mysql.connection.commit()
            logger.info(f"Generation {gen['id']} evaluation completed successfully.")
            
    except Exception as e:
        logger.error(f"Error evaluating generations: {e}")
        mysql.connection.rollback()
    finally:
        cur.close()

def continuous_training_pipeline(mysql, data_service_func, model_training_func):
    """
    Checks if enough new contests have occurred to warrant retraining.
    If so, trains a Challenger model, compares with Champion, and promotes if better.
    This should be run asynchronously.
    """
    # 1. Fetch current Champion
    # We can read the lotofacil_model.pkl history
    # For now, we delegate the logic to analysis/ml.py train_lotofacil_model
    # since it already implements walk-forward testing and history tracking.
    logger.info("Triggering continuous training pipeline...")
    
    try:
        all_results = data_service_func()
        if not all_results: return False
        
        # We assume analysis.ml.train_lotofacil_model checks for sufficient data
        # and runs walk-forward backtest automatically.
        result = model_training_func(all_results)
        
        # Log to DB
        if result and 'history_entry' in result:
            logger.info(f"Training pipeline finished: {result['history_entry']['status']}")
            return True
            
    except Exception as e:
        logger.error(f"Error in continuous training pipeline: {e}")
        
    return False
