from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src.backtest.scanner import scan_model_weight
from src.backtest.settle import settle_predictions
from src.config import load_params, validate_params
from src.data import schedule
from src.data.nowgoal import _validate_odds
from src.data.xg import get_expected_goals
from src.model.fusion import de_vig_1x2, fuse_probabilities
from src.model.poisson import outcome_probabilities, score_matrix
from src.model.predict import predict_match, run_daily


class ConfigTests(unittest.TestCase):
    def test_partial_config_merges_defaults_and_rejects_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.json"
            path.write_text('{"lambda_shrink": 0.7, "home_boost": {"PL": 1.1}}', encoding="utf-8")
            params = load_params(path)
        self.assertEqual(params["lambda_shrink"], 0.7)
        self.assertEqual(params["home_boost"]["PL"], 1.1)
        self.assertIn("PD", params["home_boost"])
        with self.assertRaises(ValueError):
            validate_params({"rho": float("nan")})

    def test_corrupt_config_is_not_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_params(path)


class ProbabilityTests(unittest.TestCase):
    def test_dixon_coles_matrix_is_normalized_and_reports_tail(self) -> None:
        result = score_matrix(1.45, 1.15, rho=-0.135, max_goals=8)
        self.assertAlmostEqual(sum(map(sum, result["matrix"])), 1.0)
        self.assertGreaterEqual(result["truncated_mass"], 0.0)
        probs = outcome_probabilities(result["matrix"])
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_rejects_invalid_expected_goals_or_negative_dc_tau(self) -> None:
        with self.assertRaises(ValueError):
            score_matrix(-0.2, 1.1)
        with self.assertRaises(ValueError):
            score_matrix(8.0, 1.0, rho=-0.2)

    def test_market_odds_are_de_vigged_and_stale_market_reduces_weight(self) -> None:
        market = de_vig_1x2({"home": 2.0, "draw": 3.2, "away": 4.0})
        self.assertAlmostEqual(sum(market.values()), 1.0)
        model = {"home": 0.5, "draw": 0.25, "away": 0.25}
        fresh_time = datetime.now(timezone.utc).isoformat()
        fresh = fuse_probabilities(model, market, fresh_time, now=datetime.now(timezone.utc))
        stale = fuse_probabilities(
            model, market, (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            now=datetime.now(timezone.utc),
        )
        self.assertAlmostEqual(fresh["model_weight"], 0.54)
        self.assertEqual(stale["model_weight"], 1.0)
        self.assertAlmostEqual(sum(stale["probabilities"].values()), 1.0)

    def test_live_matches_are_not_scored_as_prematch(self) -> None:
        result = predict_match({
            "match_id": "m1", "home_team": "A", "away_team": "B", "status": "live"
        })
        self.assertEqual(result["status"], "skipped_live")

    def test_xg_shrinkage_is_towards_explicit_baseline(self) -> None:
        params = load_params()
        home, away, source = get_expected_goals({"home_xg": 2.0, "away_xg": 1.0}, params)
        self.assertAlmostEqual(home, 0.81 * 2.0 + 0.19 * params["base_home_xg"])
        self.assertAlmostEqual(away, 0.81 * 1.0 + 0.19 * params["base_away_xg"])
        self.assertIn("baseline", source)

    def test_market_odds_require_timezone_and_valid_prices(self) -> None:
        with self.assertRaises(ValueError):
            _validate_odds(
                {"home": 2, "draw": 3, "away": 4, "updated_at": "2026-01-01T10:00:00"},
                "m1",
            )
        with self.assertRaises(ValueError):
            _validate_odds(
                {"home": 1, "draw": 3, "away": 4, "updated_at": "2026-01-01T10:00:00Z"},
                "m1",
            )


class AdapterTests(unittest.TestCase):
    def test_schedule_dated_fixture_is_explicit_demo_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "data" / "samples"
            samples.mkdir(parents=True)
            (samples / "schedule_2026-11-01.json").write_text(json.dumps({
                "matches": [{
                    "match_id": "m1", "home_team": "A", "away_team": "B",
                    "status": "scheduled",
                }]
            }), encoding="utf-8")
            with patch.object(schedule, "BASE_DIR", root), \
                    patch.dict(os.environ, {"FOOTBALL_PREDICTOR_DEMO": "1"}, clear=True):
                snapshot = schedule.fetch_jingcai_schedule("2026-11-01", load_params())
        self.assertEqual(snapshot["source"], "demo_fixture")
        self.assertEqual(snapshot["freshness"], "demo")

    def test_schedule_without_provider_or_fixture_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(schedule, "BASE_DIR", Path(directory)), \
                    patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "FOOTBALL_SCHEDULE_URL"):
                    schedule.fetch_jingcai_schedule("2026-11-01", load_params())

    def test_provider_schedule_is_cached_with_freshness_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(schedule, "BASE_DIR", Path(directory)), \
                    patch.dict(os.environ, {"FOOTBALL_SCHEDULE_URL": "https://provider.test/schedule"}, clear=True), \
                    patch.object(schedule, "_http_json", return_value={"matches": [{
                        "match_id": "m1", "home_team": "A", "away_team": "B",
                        "status": "scheduled",
                    }]}) as request:
                first = schedule.fetch_jingcai_schedule("2026-11-01", load_params())
                second = schedule.fetch_jingcai_schedule("2026-11-01", load_params())
        self.assertEqual(first["source"], "provider")
        self.assertEqual(second["freshness"], "fresh_cache")
        request.assert_called_once()


class BacktestTests(unittest.TestCase):
    def test_settlement_scores_match_and_rejects_id_mismatch(self) -> None:
        prediction = {
            "match_id": "m1", "status": "predicted",
            "probabilities": {"home": 0.7, "draw": 0.2, "away": 0.1},
        }
        report = settle_predictions([prediction], [
            {"match_id": "m1", "home_goals": 2, "away_goals": 0}
        ])
        self.assertEqual(report["metrics"]["count"], 1)
        self.assertGreaterEqual(report["metrics"]["mean_brier"], 0)
        with self.assertRaisesRegex(ValueError, "sets differ"):
            settle_predictions([prediction], [])

    def test_weight_scan_uses_later_dates_for_validation_without_applying(self) -> None:
        records = []
        for day in range(1, 6):
            for match_num in range(4):
                records.append({
                    "match_id": f"{day}-{match_num}",
                    "date": f"2026-01-{day:02d}",
                    "model_probabilities": {"home": 0.7, "draw": 0.2, "away": 0.1},
                    "market_probabilities": {"home": 0.35, "draw": 0.3, "away": 0.35},
                    "actual_outcome": "home",
                })
        result = scan_model_weight(records)
        self.assertEqual(result["training_matches"], 12)
        self.assertEqual(result["validation_matches"], 8)
        self.assertFalse(result["applied_to_config"])
        with self.assertRaisesRegex(ValueError, "distinct dates"):
            scan_model_weight(records[:4])

    def test_daily_run_saves_all_predictions_but_filters_display(self) -> None:
        snapshot = {
            "matches": [
                {"match_id": "m1", "home_team": "A", "away_team": "B",
                 "status": "scheduled", "home_xg": 8.0, "away_xg": 0.1},
                {"match_id": "m2", "home_team": "C", "away_team": "D",
                 "status": "scheduled"},
            ],
            "source": "provider", "fetched_at": datetime.now(timezone.utc).isoformat(),
            "freshness": "fresh", "age_seconds": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch("src.model.predict.fetch_jingcai_schedule", return_value=snapshot), \
                    patch("src.model.predict.get_data_dir",
                          side_effect=lambda *parts: output_dir.joinpath(*parts)), \
                    patch.dict(os.environ, {}, clear=True):
                result = run_daily("2026-11-01", only_recommend=True)
            saved = json.loads(
                (output_dir / "daily" / "predictions_2026-11-01.json").read_text(encoding="utf-8")
            )
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(len(result["displayed"]), 1)
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0]["date"], "2026-11-01")


if __name__ == "__main__":
    unittest.main()
