"""Smoke and regression tests for Flask routes and database helpers."""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import app as app_module


class RouteSmokeTests(unittest.TestCase):
    def setUp(self):
        app_module.app.config.update(TESTING=True, SECRET_KEY="test-secret")
        self.previous_db_state = app_module._db_initialized
        app_module._db_initialized = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module._db_initialized = self.previous_db_state

    def test_public_get_pages_render(self):
        routes = [
            "/",
            "/train-model",
            "/predict",
            "/historical-stats",
            "/advanced-analysis",
            "/smart-generate",
        ]
        for route in routes:
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)

    def test_unknown_analysis_type_returns_400_when_data_exists(self):
        sample = [tuple(range(1, 16)) + ("2026-09-10",)]
        with patch.object(app_module, "_fetch_all_results", return_value=sample):
            response = self.client.get("/api/analysis/not-a-real-analysis")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Tipo de análise desconhecido", response.get_json()["error"])

    def test_delay_generator_is_imported(self):
        self.assertTrue(callable(app_module.generate_delay_based))


class DatabaseHelperRegressionTests(unittest.TestCase):
    def test_next_contest_uses_max_concurso(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (3487,)
        connection = MagicMock()
        connection.cursor.return_value = cursor

        with patch.object(
            type(app_module.mysql),
            "connection",
            new_callable=PropertyMock,
            return_value=connection,
        ):
            result = app_module._get_next_contest_number()

        self.assertEqual(result, 3488)
        cursor.execute.assert_called_once_with("SELECT MAX(concurso) FROM results")
        cursor.close.assert_called_once()

    def test_next_contest_is_one_when_database_is_empty(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (None,)
        connection = MagicMock()
        connection.cursor.return_value = cursor

        with patch.object(
            type(app_module.mysql),
            "connection",
            new_callable=PropertyMock,
            return_value=connection,
        ):
            result = app_module._get_next_contest_number()

        self.assertEqual(result, 1)

    def test_latest_result_queries_descending_contest(self):
        latest = tuple(range(1, 16)) + ("2026-09-10",)
        cursor = MagicMock()
        cursor.fetchone.return_value = latest
        connection = MagicMock()
        connection.cursor.return_value = cursor

        with patch.object(
            type(app_module.mysql),
            "connection",
            new_callable=PropertyMock,
            return_value=connection,
        ):
            result = app_module._fetch_latest_result()

        self.assertEqual(result, latest)
        sql = cursor.execute.call_args.args[0]
        self.assertIn("ORDER BY concurso DESC", sql)
        self.assertIn("LIMIT 1", sql)
        cursor.close.assert_called_once()

    def test_saved_games_compares_against_latest_draw(self):
        app_module.app.config.update(TESTING=True, SECRET_KEY="test-secret")
        client = app_module.app.test_client()

        saved_cursor = MagicMock()
        saved_cursor.fetchall.return_value = [
            {
                "balls": ",".join(str(n) for n in range(1, 16)),
                "details": None,
                "created_at": None,
            }
        ]
        connection = MagicMock()
        connection.cursor.return_value = saved_cursor
        latest = tuple(range(1, 16)) + ("2026-09-10",)

        with patch.object(app_module, "_db_initialized", True), \
             patch.object(
                 type(app_module.mysql),
                 "connection",
                 new_callable=PropertyMock,
                 return_value=connection,
             ), \
             patch.object(app_module, "_fetch_latest_result", return_value=latest), \
             patch.object(app_module, "render_template", return_value="ok") as render:
            response = client.get("/saved-games")

        self.assertEqual(response.status_code, 200)
        kwargs = render.call_args.kwargs
        self.assertEqual(kwargs["saved_games"][0]["hits_last_draw"], 15)
        self.assertEqual(kwargs["last_draw"], list(range(1, 16)))


if __name__ == "__main__":
    unittest.main()
