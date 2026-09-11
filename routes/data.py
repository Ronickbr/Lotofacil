from io import BytesIO
from flask import send_file
from services.stats_service import _fetch_latest_result, _get_next_contest_number
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
from datetime import datetime
from extensions import mysql
import MySQLdb
from analysis.mlops import check_and_evaluate_generations

data_bp = Blueprint('data', __name__)

@data_bp.route('/add-concurso', methods=['POST'])
def add_concurso():
    cur = None
    try:
        cur = mysql.connection.cursor()

        concurso = request.form.get('concurso')
        data = request.form.get('data')
        bolas = [request.form.get(f'bola{i}') for i in range(1, 16)]

        if not all([concurso, data] + bolas):
            flash('Todos os campos são obrigatórios!', 'danger')
            return redirect(url_for('data.upload'))

        cur.execute("SELECT concurso FROM results WHERE concurso = %s", (concurso,))
        if cur.fetchone():
            flash('Concurso já existe!', 'danger')
            return redirect(url_for('data.upload'))

        query = """
            INSERT INTO results (concurso, data_sorteio, bola1, bola2, bola3, bola4, bola5,
                               bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13,
                               bola14, bola15)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = [concurso, data] + bolas
        cur.execute(query, values)
        mysql.connection.commit()
        check_and_evaluate_generations(mysql)
        flash('Concurso adicionado com sucesso e jogos avaliados!', 'success')

    except Exception as e:
        mysql.connection.rollback()
        flash(f'Erro ao adicionar concurso: {str(e)}', 'danger')
    finally:
        if cur is not None:
            cur.close()

    return redirect(url_for('data.upload'))


@data_bp.route('/delete-concurso/<int:concurso>', methods=['DELETE'])
def delete_concurso(concurso):
    cur = None
    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM results WHERE concurso = %s", (concurso,))
        mysql.connection.commit()
        return jsonify({'success': True})
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if cur is not None:
            cur.close()


@data_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    cur = None

    if request.method == 'POST':
        arquivo = request.files.get('file')

        if not arquivo or not arquivo.filename:
            flash('Selecione um arquivo Excel para importar.', 'danger')
            return redirect(url_for('data.upload'))

        if not arquivo.filename.lower().endswith('.xlsx'):
            flash('Formato inválido. Envie um arquivo Excel .xlsx.', 'danger')
            return redirect(url_for('data.upload'))

        try:
            df = pd.read_excel(arquivo, engine='openpyxl')
            df.columns = [str(col).strip().lower() for col in df.columns]

            colunas_esperadas = ['concurso', 'data'] + [f'bola{i}' for i in range(1, 16)]
            colunas_faltantes = [col for col in colunas_esperadas if col not in df.columns]
            if colunas_faltantes:
                raise ValueError('Colunas ausentes no Excel: ' + ', '.join(colunas_faltantes))

            df = df[colunas_esperadas].dropna(how='all')
            df = df.drop_duplicates(subset=['concurso'], keep='first')

            registros = []
            erros = []
            for numero_linha, (_, linha) in enumerate(df.iterrows(), start=2):
                try:
                    concurso = int(linha['concurso'])
                    data_sorteio = pd.to_datetime(linha['data'], dayfirst=True, errors='coerce')
                    bolas = [int(linha[f'bola{i}']) for i in range(1, 16)]

                    if pd.isna(data_sorteio):
                        raise ValueError('data inválida')
                    if len(set(bolas)) != 15 or any(bola < 1 or bola > 25 for bola in bolas):
                        raise ValueError('as bolas devem ser 15 números únicos entre 1 e 25')

                    registros.append([concurso, data_sorteio.strftime('%Y-%m-%d')] + bolas)
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
                check_and_evaluate_generations(mysql)

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

        return redirect(url_for('data.upload'))

    try:
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
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



@data_bp.route('/save-games', methods=['POST'])
def save_games():
    try:
        data = request.get_json()
        if not data or 'games' not in data:
            return (jsonify({'error': 'Dados inv├ílidos.'}), 400)
        strategy = data.get('strategy', 'Unknown')
        games = data['games']
        ai_ranking = data.get('ai_ranking')
        target_contest = _get_next_contest_number()
        import uuid
        generation_id = f'GEN-{target_contest}-{uuid.uuid4().hex[:6].upper()}'
        cur = mysql.connection.cursor()
        saved_count = 0
        cur.execute("\n            INSERT INTO generations\n            (id, target_contest, created_at, model_version, strategy, num_games, status)\n            VALUES (%s, %s, NOW(), %s, %s, %s, 'AGUARDANDO_RESULTADO')\n        ", (generation_id, target_contest, 'current', strategy, len(games)))
        if ai_ranking and isinstance(ai_ranking, list):
            for i, r in enumerate(ai_ranking):
                cur.execute('\n                    INSERT INTO prediction_history\n                    (generation_id, target_contest, number, predicted_probability, ranking_position, model_version, created_at)\n                    VALUES (%s, %s, %s, %s, %s, %s, NOW())\n                ', (generation_id, target_contest, r['dezena'], r['score'], i + 1, 'current'))
        for g in games:
            hash_val = g.get('hash')
            if not hash_val:
                continue
            balls_str = ','.join(map(str, g['numbers']))
            score = g.get('total_score', 0)
            import json
            details_json = json.dumps({'explanation': g.get('explanation', ''), 'evens': g.get('evens', 0), 'odds': g.get('odds', 0), 'primes': g.get('primes', 0), 'game_sum': g.get('game_sum', 0)})
            cur.execute('\n                INSERT IGNORE INTO saved_games\n                (created_at, strategy, balls, score, details, hash, generation_id, target_contest)\n                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)\n            ', (strategy, balls_str, score, details_json, hash_val, generation_id, target_contest))
            if cur.rowcount > 0:
                saved_count += 1
        mysql.connection.commit()
        cur.close()
        return jsonify({'message': f'{saved_count} novos jogos salvos com sucesso (ignoradas duplicatas).', 'generation_id': generation_id, 'target_contest': target_contest})
    except Exception as e:
        mysql.connection.rollback()
        import traceback
        traceback.print_exc()
        return (jsonify({'error': str(e)}), 500)



@data_bp.route('/saved-games')
def saved_games():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT * FROM saved_games ORDER BY created_at DESC LIMIT 500')
    saved = cur.fetchall()
    cur.close()
    latest_result = _fetch_latest_result()
    last_draw = set(latest_result[:15]) if latest_result else set()
    import json
    for game in saved:
        balls = [int(n) for n in game['balls'].split(',')]
        game['numbers'] = balls
        if game['details']:
            game['details_obj'] = json.loads(game['details'])
        else:
            game['details_obj'] = {}
        game['hits_last_draw'] = len(set(balls) & last_draw) if last_draw else 0
    return render_template('saved_games.html', saved_games=saved, last_draw=sorted(last_draw))



@data_bp.route('/export-excel', methods=['GET'])
def export_excel():
    conn = mysql.connection
    cur = conn.cursor()
    cur.execute('SELECT concurso, data_sorteio, bola1, bola2, bola3, bola4, bola5,bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15 FROM results ORDER BY concurso DESC')
    results = cur.fetchall()
    cur.close()
    columns = ['Concurso', 'Data', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6', 'Bola7', 'Bola8', 'Bola9', 'Bola10', 'Bola11', 'Bola12', 'Bola13', 'Bola14', 'Bola15']
    data = []
    for row in results:
        date_val = row[1]
        if date_val is None:
            date_str = 'N/A'
        elif isinstance(date_val, str):
            date_str = date_val
        else:
            date_str = date_val.strftime('%d/%m/%Y')
        bolas = list(row[2:17])
        data.append({'Concurso': row[0], 'Data': date_str, **{f'Bola{i + 1}': bolas[i] if i < len(bolas) else None for i in range(15)}})
    df = pd.DataFrame(data, columns=columns)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    output.seek(0)
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='resultados_lotofacil.xlsx')

