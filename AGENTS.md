# AGENTS.md — Lotofácil Analysis System

## Quick Start
```bash
docker-compose up --build
# Access: http://localhost:5000 (app), http://localhost:8081 (phpMyAdmin root/secret)
```

## Existing Analysis Capabilities (in `app.py`)

### 1. Dashboard Statistics (`/dashboard`)
- Overall frequency of all numbers (1-25)
- Even/odd distribution
- Position frequency (which position each number appears in)

### 2. Historical Statistics (`/historical-stats?period=week|month|year&prediction_type=...`)
**Periods**: `week` (7d), `month` (30d), `year` (365d)  
**Methods**:
- `frequency` — simple count/percentage
- `bayes` — Bayesian posterior probabilities (prior × likelihood)
- `pattern` — numbers repeating in consecutive draws
- `combined` — weighted: 40% freq + 30% pattern + 30% Bayes

### 3. ML Model (`/train-model`, `/predict`)
- RandomForest predicting next draw from previous draw
- Trains on sequential pairs (draw N → draw N+1)
- Generates 6 suggested games from top predicted numbers

### 4. Core Functions to Extend
| Function | Location | Purpose |
|----------|----------|---------|
| `calculate_statistics()` | app.py:296 | Main dispatcher for 4 analysis methods |
| `calculate_bayes_probabilities()` | app.py:355 | Bayesian update per number |
| `analyze_patterns()` | app.py:375 | Consecutive-draw repetition |
| `combine_analysis_methods()` | app.py:389 | Weighted ensemble |

## Missing Temporal/Probabilistic Analysis
The user wants **cycle detection by day-of-week, month, specific dates** — not yet implemented. Key gaps:

1. **No temporal grouping**: Results only filtered by date range, not grouped by weekday/month/day
2. **No trend detection**: No hot/cold tracking over sliding windows
3. **No significance testing**: Frequencies reported without confidence intervals
4. **No conditional probabilities**: e.g., P(number X | day=Monday, month=January)

## How to Add Temporal Analysis

### Option A: SQL-based (fastest for exploration)
```sql
-- Day-of-week frequency
SELECT DAYOFWEEK(data_sorteio) AS dow, bola1, COUNT(*) 
FROM results GROUP BY dow, bola1;

-- Month frequency
SELECT MONTH(data_sorteio) AS mes, bola1, COUNT(*) 
FROM results GROUP BY mes, bola1;

-- Specific date patterns (e.g., day 13 of any month)
SELECT DAY(data_sorteio) AS dia, bola1, COUNT(*) 
FROM results GROUP BY dia, bola1;
```

### Option B: Python extension (add to `app.py`)
```python
def analyze_temporal_patterns(results):
    """Returns dict: {(period_type, period_value, number): count}"""
    from collections import defaultdict
    patterns = defaultdict(Counter)
    for row in results:
        balls, date = list(row[:-1]), row[-1]
        dow = date.weekday()  # 0=Mon
        month = date.month
        day = date.day
        for n in balls:
            patterns[('dow', dow)][n] += 1
            patterns[('month', month)][n] += 1
            patterns[('day', day)][n] += 1
    return patterns
```

### Option C: New endpoint for API consumers
```python
@app.route('/temporal-analysis')
def temporal_analysis():
    period_type = request.args.get('type')  # 'dow', 'month', 'day'
    # query, group, return JSON
```

## Database Schema (from `init.sql`)
```sql
results: concurso (PK), data_sorteio (DATE), bola1..bola15 (INT 1-25)
uploads: audit trail for bulk imports
```

## Key Files to Modify
| File | Purpose |
|------|---------|
| `app.py` | All routes + analysis logic (414 lines) |
| `templates/historical_stats.html` | UI for period/method selection |
| `templates/predict.html` | ML prediction UI |
| `requirements.txt` | Dependencies (sklearn, pandas, numpy) |

## Testing Commands
```bash
# Run app locally (outside docker)
pip install -r requirements.txt
python app.py

# Verify DB connection
docker-compose exec db mysql -u root -psecret lotofacil -e "SELECT COUNT(*) FROM results;"

# Check model file exists after training
ls -la lotofacil_model.pkl
```

## Common Pitfalls
- **Model file missing**: Must train via `/train-model` before `/predict` works
- **Empty results**: `/historical-stats` returns error if no draws in period
- **Date parsing**: `data_sorteio` stored as DATE, returned as `datetime.date` object
- **Concurrency**: SQLite not used — MySQL handles concurrent writes via Docker

## Extending for "Formula" Detection
To find "formulas" (recurring patterns by calendar):
1. Add temporal grouping in `calculate_statistics()` or new function
2. Expose via new `prediction_type='temporal'` in `/historical-stats`
3. Consider chi-square test for significance of day/month effects
4. Store precomputed aggregates in a materialized view/table for speed