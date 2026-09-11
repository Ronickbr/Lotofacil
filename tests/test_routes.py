"""Regression tests for Blueprints, route contracts and ML resilience."""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from app import create_app
from analysis.ml import predict_next_numbers, predict_next_numbers_detailed
from analysis.scoring import GameScorer
from services import stats_service


def sample_results(count=60):
    results = []
    for offset in range(count):
        balls = [((number + offset) % 25) + 1 for number in range(15)]
        results.append(tuple(sorted(balls)) + (f'2026-01-{(offset % 28) + 1:02d}',))
    return results


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret',
            'INIT_DB_ON_REQUEST': False,
        })
        self.client = self.app.test_client()


class RouteSmokeTests(AppTestCase):
    def test_public_get_pages_render(self):
        routes = ['/', '/train-model', '/predict', '/historical-stats', '/advanced-analysis', '/smart-generate']
        for route in routes:
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).status_code, 200)

    def test_unknown_analysis_type_returns_400_when_data_exists(self):
        with patch('routes.analysis._fetch_all_results', return_value=sample_results(1)):
            response = self.client.get('/api/analysis/not-a-real-analysis')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Tipo de análise desconhecido', response.get_json()['error'])


class SmartGenerateContractTests(AppTestCase):
    def test_direct_game_returns_complete_frontend_and_save_contract(self):
        numbers = ','.join(str(number) for number in range(1, 16))
        with patch('routes.ml._fetch_all_results', return_value=sample_results()):
            response = self.client.post(
                '/smart-generate',
                data={'strategy': 'direto', 'num_games': '1', 'numbers': numbers},
                headers={'X-Requested-With': 'XMLHttpRequest'},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['strategy'], 'direto')
        self.assertEqual(payload['total_generated'], 1)
        self.assertEqual(payload['cost'], 3.0)
        self.assertIsNone(payload['ai_ranking'])
        game = payload['games'][0]
        self.assertEqual(game['numbers'], list(range(1, 16)))
        for field in ('hash', 'total_score', 'components', 'game_sum', 'evens', 'odds', 'primes', 'explanation'):
            self.assertIn(field, game)

    def test_unknown_strategy_returns_400(self):
        with patch('routes.ml._fetch_all_results', return_value=sample_results()):
            response = self.client.post(
                '/smart-generate',
                data={'strategy': 'invalid', 'num_games': '1'},
                headers={'X-Requested-With': 'XMLHttpRequest'},
            )
        self.assertEqual(response.status_code, 400)

    def test_duplicate_direct_numbers_are_rejected(self):
        with patch('routes.ml._fetch_all_results', return_value=sample_results()):
            response = self.client.post(
                '/smart-generate',
                data={'strategy': 'direto', 'num_games': '1', 'numbers': ','.join(['1'] * 15)},
                headers={'X-Requested-With': 'XMLHttpRequest'},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn('não podem se repetir', response.get_json()['error'])


class PredictionRouteTests(AppTestCase):
    def test_missing_model_uses_statistical_fallback(self):
        with patch('routes.ml.load_lotofacil_model', side_effect=FileNotFoundError), \
             patch('routes.ml._fetch_all_results', return_value=sample_results()):
            response = self.client.post('/predict', headers={'X-Requested-With': 'XMLHttpRequest'})
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('fallback estatístico', body)
        self.assertIn('Bilhetes Sugeridos', body)

    def test_empty_database_returns_404(self):
        with patch('routes.ml.load_lotofacil_model', return_value=MagicMock()), \
             patch('routes.ml._fetch_all_results', return_value=[]):
            response = self.client.post('/predict', headers={'X-Requested-With': 'XMLHttpRequest'})
        self.assertEqual(response.status_code, 404)


class MLResilienceTests(unittest.TestCase):
    def setUp(self):
        self.results = sample_results()

    def test_prediction_falls_back_when_model_inference_fails(self):
        broken_model = MagicMock()
        broken_model.predict_proba.side_effect = ValueError('incompatible feature count')
        numbers = predict_next_numbers(broken_model, self.results, top_k=10)
        self.assertEqual(len(numbers), 10)
        self.assertEqual(len(set(numbers)), 10)
        self.assertTrue(all(1 <= number <= 25 for number in numbers))

    def test_detailed_prediction_falls_back_when_model_inference_fails(self):
        broken_model = MagicMock()
        broken_model.predict_proba.side_effect = RuntimeError('broken pickle')
        data = predict_next_numbers_detailed(broken_model, self.results)
        self.assertEqual(len(data['ranking']), 25)
        self.assertEqual(len(data['top18']), 18)
        self.assertEqual(len(set(data['top18'])), 18)


class DatabaseHelperRegressionTests(AppTestCase):
    def test_next_contest_uses_max_concurso(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (3487,)
        connection = MagicMock()
        connection.cursor.return_value = cursor
        with patch.object(type(stats_service.mysql), 'connection', new_callable=PropertyMock, return_value=connection):
            result = stats_service._get_next_contest_number()
        self.assertEqual(result, 3488)
        cursor.execute.assert_called_once_with('SELECT MAX(concurso) FROM results')
        cursor.close.assert_called_once()

    def test_latest_result_queries_descending_contest(self):
        latest = tuple(range(1, 16)) + ('2026-09-10',)
        cursor = MagicMock()
        cursor.fetchone.return_value = latest
        connection = MagicMock()
        connection.cursor.return_value = cursor
        with patch.object(type(stats_service.mysql), 'connection', new_callable=PropertyMock, return_value=connection):
            result = stats_service._fetch_latest_result()
        self.assertEqual(result, latest)
        sql = cursor.execute.call_args.args[0]
        self.assertIn('ORDER BY concurso DESC', sql)
        self.assertIn('LIMIT 1', sql)
        cursor.close.assert_called_once()

    def test_saved_games_compares_against_latest_draw(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = [{
            'balls': ','.join(str(number) for number in range(1, 16)),
            'details': None,
            'created_at': None,
            'locked_at': None,
        }]
        connection = MagicMock()
        connection.cursor.return_value = cursor
        latest = tuple(range(1, 16)) + ('2026-09-10',)
        with patch('routes.data._fetch_latest_result', return_value=latest), \
             patch('routes.data.render_template', return_value='ok') as render, \
             patch('routes.data.mysql') as mysql_mock:
            mysql_mock.connection = connection
            response = self.client.get('/saved-games')
        self.assertEqual(response.status_code, 200)
        kwargs = render.call_args.kwargs
        self.assertEqual(kwargs['saved_games'][0]['hits_last_draw'], 15)
        self.assertEqual(kwargs['last_draw'], list(range(1, 16)))


class SaveGamesValidationTests(AppTestCase):
    def test_rejects_malformed_game_before_database_access(self):
        response = self.client.post('/save-games', json={
            'strategy': 'direto',
            'games': [{'numbers': [1] * 15}],
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('15 dezenas únicas', response.get_json()['error'])


class LatestDrawRegressionTests(unittest.TestCase):
    def test_scorer_uses_newest_result(self):
        oldest = tuple(range(1, 16)) + ('2026-01-01',)
        newest = tuple(range(11, 26)) + ('2026-01-02',)
        scorer = GameScorer([oldest, newest])
        self.assertEqual(scorer.last_draw, set(range(11, 26)))


if __name__ == '__main__':
    unittest.main()
