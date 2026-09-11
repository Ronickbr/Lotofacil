"""Incremental synchronization with the public CAIXA Lotofácil endpoint."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CAIXA_API_URL = 'https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil'


class CaixaAPIError(RuntimeError):
    """Raised when CAIXA cannot provide a valid contest payload."""


def _request_json(url, timeout=12):
    request = Request(
        url,
        headers={
            'Accept': 'application/json',
            'User-Agent': 'Lotofacil-Analytics/1.0',
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8-sig'))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, UnicodeError) as exc:
        raise CaixaAPIError('A CAIXA não respondeu com dados válidos.') from exc


def _parse_contest(payload):
    try:
        contest_number = int(payload['numero'])
        draw_date = datetime.strptime(payload['dataApuracao'], '%d/%m/%Y').date()
        numbers = sorted(int(number) for number in payload['listaDezenas'])
    except (KeyError, TypeError, ValueError) as exc:
        raise CaixaAPIError('O concurso recebido da CAIXA possui formato inválido.') from exc

    if contest_number < 1:
        raise CaixaAPIError('O número do concurso recebido é inválido.')
    if len(numbers) != 15 or len(set(numbers)) != 15 or any(number < 1 or number > 25 for number in numbers):
        raise CaixaAPIError('As dezenas recebidas da CAIXA são inválidas.')

    return {
        'concurso': contest_number,
        'data_sorteio': draw_date,
        'numbers': numbers,
    }


def fetch_latest_contest(timeout=12):
    return _parse_contest(_request_json(CAIXA_API_URL, timeout=timeout))


def fetch_contest(contest_number, timeout=12):
    contest = _parse_contest(
        _request_json(f'{CAIXA_API_URL}/{int(contest_number)}', timeout=timeout)
    )
    if contest['concurso'] != int(contest_number):
        raise CaixaAPIError(f'A CAIXA devolveu um concurso diferente do solicitado: {contest_number}.')
    return contest


def _fetch_contest_batch(contest_numbers, timeout, workers):
    contests = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_contest, contest_number, timeout): contest_number
            for contest_number in contest_numbers
        }
        for future in as_completed(futures):
            contests.append(future.result())
    return sorted(contests, key=lambda contest: contest['concurso'])


def sync_missing_contests(mysql, max_batch=250, timeout=12, workers=5):
    """Import contests newer than the greatest local contest in one transaction."""
    latest_api = fetch_latest_contest(timeout=timeout)
    cursor = mysql.connection.cursor()
    try:
        cursor.execute('SELECT MAX(concurso) FROM results')
        row = cursor.fetchone()
        latest_local = int(row[0]) if row and row[0] is not None else None

        if latest_local is not None and latest_local >= latest_api['concurso']:
            return {
                'imported': 0,
                'latest_api': latest_api['concurso'],
                'latest_local': latest_local,
                'remaining': 0,
            }

        # On an empty database, seed the latest draw. Historical bulk import remains available by Excel.
        first_contest = latest_api['concurso'] if latest_local is None else latest_local + 1
        last_contest = min(latest_api['concurso'], first_contest + max(1, max_batch) - 1)
        contest_numbers = range(first_contest, last_contest + 1)

        if first_contest == latest_api['concurso']:
            contests = [latest_api]
        else:
            contests = _fetch_contest_batch(contest_numbers, timeout, max(1, min(workers, 8)))

        query = """
            INSERT IGNORE INTO results (
                concurso, data_sorteio, bola1, bola2, bola3, bola4, bola5,
                bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13,
                bola14, bola15
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        rows = [
            [contest['concurso'], contest['data_sorteio'], *contest['numbers']]
            for contest in contests
        ]
        cursor.executemany(query, rows)
        imported = cursor.rowcount
        mysql.connection.commit()
        return {
            'imported': imported,
            'latest_api': latest_api['concurso'],
            'latest_local': contests[-1]['concurso'],
            'remaining': max(0, latest_api['concurso'] - contests[-1]['concurso']),
        }
    except Exception:
        mysql.connection.rollback()
        raise
    finally:
        cursor.close()
