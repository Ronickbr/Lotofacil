"""JEV classifica pedidos de estratégia; não prevê dezenas sorteadas."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
STRATEGIES = {
    'frequency': 'Analisar frequência histórica; não implica que dezenas quentes sejam mais prováveis.',
    'delay': 'Explorar dezenas atrasadas; atraso não aumenta chance no próximo sorteio.',
    'repetition': 'Escolher cartões com dezenas repetidas do último sorteio.',
    'montecarlo': 'Gerar cartões com filtros combinatórios e de distribuição.',
    'ai_based': 'Usar o modelo estatístico local, sujeito ao backtest temporal.',
    'direto': 'Selecionar manualmente exatamente 15 dezenas.',
}


class JevError(Exception):
    pass


def choose_strategy(text, *, api_key=None, opener=urlopen):
    """Retorna somente uma estratégia conhecida e metadados seguros da API."""
    if not isinstance(text, str) or not 3 <= len(text.strip()) <= 500:
        raise ValueError('Descreva a estratégia em 3 a 500 caracteres.')
    api_key = api_key or os.environ.get('TYPESAFE_API_KEY')
    if not api_key:
        raise JevError('Configure TYPESAFE_API_KEY para usar o assistente JEV.')

    payload = {
        'state': text.strip(),
        'model': 'jev-latest',
        'questions': {
            'strategy': {
                'type': 'choice',
                'instructions': 'Qual estratégia da Lotofácil corresponde melhor ao pedido? Escolha apenas uma.',
                'criteria': STRATEGIES,
            },
        },
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with opener(request, timeout=8) as response:
            answer = json.load(response)['answers']['strategy']
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        raise JevError('O JEV não respondeu corretamente. Escolha a estratégia manualmente.') from exc

    strategy = answer.get('choice')
    confidence = answer.get('confidence')
    if (answer.get('type') != 'choice' or strategy not in STRATEGIES
            or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
        raise JevError('O JEV retornou uma estratégia inválida. Escolha manualmente.')
    return {'strategy': strategy, 'confidence': float(confidence),
            'requires_confirmation': confidence < 0.6,
            'note': 'O JEV interpreta o pedido. Não aumenta a probabilidade de acerto.'}
