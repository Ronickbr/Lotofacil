from flask import Blueprint, render_template
import pandas as pd
from extensions import mysql

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/dashboard')
def dashboard():
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 FROM results"
        )
        data = cur.fetchall()

        if not data:
            return render_template(
                'dashboard.html',
                error="Não há dados disponíveis. Faça upload de resultados primeiro.",
            )

        df = pd.DataFrame(data, columns=[f'bola{i}' for i in range(1, 16)])

        all_numbers = df.values.flatten()
        freq = pd.Series(all_numbers).value_counts()

        top_numbers = freq.head(5).index.tolist()
        frequencies = freq.head(5).values.tolist()

        even_count = (df % 2 == 0).sum().sum()
        odd_count = (df % 2 != 0).sum().sum()

        position_freq = df.apply(pd.Series.value_counts).fillna(0).astype(int)
        position_freq_html = position_freq.to_html(classes='table table-striped table-hover')

        return render_template(
            'dashboard.html',
            top_numbers=top_numbers,
            frequencies=frequencies,
            even_count=even_count,
            odd_count=odd_count,
            position_freq=position_freq_html,
        )

    except Exception as e:
        return render_template('dashboard.html', error=f"Erro ao carregar dados: {str(e)}")
    finally:
        if 'cur' in locals() and cur is not None:
            cur.close()
