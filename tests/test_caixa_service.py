"""Tests for the incremental CAIXA contest synchronization."""

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from app import create_app
from services.caixa_service import CaixaAPIError, _parse_contest, sync_missing_contests


def contest(number):
    return {
        'concurso': number,
        'data_sorteio': date(2026, 9, 10),
        'numbers': list(range(1, 16)),
    }


class CaixaPayloadTests(unittest.TestCase):
    def test_parse_valid_payload(self):
        parsed = _parse_contest({
            'numero': 3779,
            'dataApuracao': '03/09/2026',
            'listaDezenas': [str(number) for number in range(1, 16)],
        })
        self.assertEqual(parsed['concurso'], 3779)
        self.assertEqual(parsed['data_sorteio'], date(2026, 9, 3))
        self.assertEqual(parsed['numbers'], list(range(1, 16)))

    def test_rejects_duplicate_numbers(self):
        with self.assertRaises(CaixaAPIError):
            _parse_contest({
                'numero': 3779,
                'dataApuracao': '03/09/2026',
                'listaDezenas': ['1'] * 15,
            })


class CaixaSyncTests(unittest.TestCase):
    def test_up_to_date_database_does_not_insert(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (3779,)
        mysql = MagicMock()
        mysql.connection.cursor.return_value = cursor

        with patch('services.caixa_service.fetch_latest_contest', return_value=contest(3779)):
            result = sync_missing_contests(mysql)

        self.assertEqual(result['imported'], 0)
        cursor.executemany.assert_not_called()
        mysql.connection.commit.assert_not_called()

    def test_imports_only_newer_contests(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (3777,)
        cursor.rowcount = 2
        mysql = MagicMock()
        mysql.connection.cursor.return_value = cursor

        with patch('services.caixa_service.fetch_latest_contest', return_value=contest(3779)), \
             patch('services.caixa_service._fetch_contest_batch', return_value=[contest(3778), contest(3779)]) as fetch_batch:
            result = sync_missing_contests(mysql)

        fetch_batch.assert_called_once()
        self.assertEqual(list(fetch_batch.call_args.args[0]), [3778, 3779])
        self.assertEqual(result['imported'], 2)
        self.assertEqual(result['remaining'], 0)
        cursor.executemany.assert_called_once()
        mysql.connection.commit.assert_called_once()

    def test_database_error_rolls_back(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (3778,)
        cursor.executemany.side_effect = RuntimeError('database unavailable')
        mysql = MagicMock()
        mysql.connection.cursor.return_value = cursor

        with patch('services.caixa_service.fetch_latest_contest', return_value=contest(3779)):
            with self.assertRaises(RuntimeError):
                sync_missing_contests(mysql)
        mysql.connection.rollback.assert_called_once()
        cursor.close.assert_called_once()


class CaixaSyncRouteTests(unittest.TestCase):
    def setUp(self):
        app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret',
            'INIT_DB_ON_REQUEST': False,
        })
        self.client = app.test_client()

    def test_sync_route_returns_import_summary(self):
        summary = {'imported': 2, 'latest_api': 3779, 'latest_local': 3779, 'remaining': 0}
        with patch('routes.data.sync_missing_contests', return_value=summary), \
             patch('routes.data.check_and_evaluate_generations'):
            response = self.client.post('/sync-caixa')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()['success'])
        self.assertEqual(response.get_json()['imported'], 2)

    def test_sync_route_reports_upstream_failure(self):
        with patch('routes.data.sync_missing_contests', side_effect=CaixaAPIError('CAIXA indisponível')):
            response = self.client.post('/sync-caixa')
        self.assertEqual(response.status_code, 502)
        self.assertFalse(response.get_json()['success'])


if __name__ == '__main__':
    unittest.main()
