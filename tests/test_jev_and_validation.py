"""Contract tests for JEV routing and exact random baseline."""
import io
import json
import unittest
from unittest.mock import patch

from analysis.ml import _simulate_random_baseline
from services.jev_service import JevError, choose_strategy


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TestJevAndValidation(unittest.TestCase):
    def test_baseline_is_exact_and_repeatable(self):
        baseline = _simulate_random_baseline()
        self.assertEqual(baseline['mean'], 9.0)
        self.assertEqual(baseline, _simulate_random_baseline())
        self.assertAlmostEqual(sum(baseline['dist'][str(k)] for k in range(8, 16)),
                               sum(baseline['dist'][f'ge_{k}'] for k in [10])
                               + baseline['dist']['8'] + baseline['dist']['9'])

    def test_jev_uses_typed_choice_without_exposing_key(self):
        def opener(request, timeout):
            self.assertEqual(timeout, 8)
            body = json.loads(request.data)
            self.assertEqual(body['model'], 'jev-latest')
            self.assertEqual(body['questions']['strategy']['type'], 'choice')
            return FakeResponse(json.dumps({'answers': {'strategy': {
                'type': 'choice', 'choice': 'montecarlo', 'confidence': 0.9,
            }}}).encode())

        result = choose_strategy('Quero filtros de pares', api_key='test-key', opener=opener)
        self.assertEqual(result['strategy'], 'montecarlo')
        self.assertFalse(result['requires_confirmation'])

    def test_rejects_unknown_strategy_and_missing_key(self):
        bad = lambda request, timeout: FakeResponse(json.dumps({'answers': {'strategy': {
            'type': 'choice', 'choice': 'guaranteed_win', 'confidence': 1.0,
        }}}).encode())
        with self.assertRaises(JevError):
            choose_strategy('Quero apostar', api_key='test-key', opener=bad)
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': ''}):
            with self.assertRaises(JevError):
                choose_strategy('Quero apostar')


if __name__ == '__main__':
    unittest.main()
