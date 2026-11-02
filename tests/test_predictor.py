from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import run_today
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
        self.assertFalse(params.get("allow_baseline_recommend", False))
        with self.assertRaisesRegex(ValueError, "allow_baseline_recommend"):
            validate_params({"allow_baseline_recommend": 1})

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

    def test_xg_falls_back_when_both_values_are_missing(self) -> None:
        params = load_params()
        home, away, source = get_expected_goals({}, params)
        self.assertEqual((home, away), (params["base_home_xg"], params["base_away_xg"]))
        self.assertEqual(source, "unvalidated_configured_baseline")

    def test_xg_requires_both_home_and_away_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "both home_xg and away_xg"):
            get_expected_goals({"home_xg": 1.2})
        with self.assertRaisesRegex(ValueError, "both home_xg and away_xg"):
            get_expected_goals({"away_xg": 0.8})

    def test_xg_rejects_bool_negative_nan_and_infinity(self) -> None:
        for bad_value in (True, -0.1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad_value):
                with self.assertRaisesRegex(ValueError, "home_xg"):
                    get_expected_goals({"home_xg": bad_value, "away_xg": 1.0})

    def test_xg_shrinkage_parameter_must_be_in_range(self) -> None:
        for shrinkage in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(lambda_shrink=shrinkage):
                with self.assertRaises(ValueError):
                    get_expected_goals(
                        {"home_xg": 1.0, "away_xg": 1.0},
                        {"lambda_shrink": shrinkage},
                    )

    def test_league_home_boost_adjusts_prior_not_supplied_xg(self) -> None:
        params = load_params()
        boosted_home, boosted_away, source = get_expected_goals(
            {"league": "PL"}, params
        )
        self.assertAlmostEqual(boosted_home, params["base_home_xg"] * 1.02)
        self.assertEqual(boosted_away, params["base_away_xg"])
        self.assertIn("home_boost", source)

        shrunk_home, shrunk_away, _ = get_expected_goals(
            {"league": "PL", "home_xg": 2.0, "away_xg": 1.0}, params
        )
        self.assertAlmostEqual(
            shrunk_home,
            0.81 * 2.0 + 0.19 * params["base_home_xg"] * 1.02,
        )
        self.assertAlmostEqual(
            shrunk_away,
            0.81 * 1.0 + 0.19 * params["base_away_xg"],
        )
        no_shrink_params = {**params, "lambda_shrink": 1.0}
        supplied_home, _, _ = get_expected_goals(
            {"league": "PL", "home_xg": 2.0, "away_xg": 1.0},
            no_shrink_params,
        )
        self.assertEqual(supplied_home, 2.0)

    def test_missing_xg_config_fields_use_documented_defaults(self) -> None:
        home, away, source = get_expected_goals({}, {"lambda_shrink": 0.5})
        params = load_params()
        self.assertEqual((home, away), (params["base_home_xg"], params["base_away_xg"]))
        self.assertEqual(source, "unvalidated_configured_baseline")
        with self.assertRaisesRegex(ValueError, "base_home_xg"):
            get_expected_goals({}, {"base_home_xg": -0.2})
        with self.assertRaisesRegex(ValueError, "base_away_xg"):
            get_expected_goals({}, {"base_away_xg": float("inf")})
        with self.assertRaisesRegex(ValueError, "home_boost"):
            get_expected_goals({}, {"home_boost": {"PL": True}})

    def test_prediction_surfaces_unvalidated_baseline_source(self) -> None:
        prediction = predict_match({
            "match_id": "m-baseline",
            "home_team": "A",
            "away_team": "B",
            "status": "scheduled",
        })
        self.assertEqual(prediction["xg_source"], "unvalidated_configured_baseline")
        self.assertEqual(
            prediction["expected_goals"]["source"],
            "unvalidated_configured_baseline",
        )
        self.assertIsNone(prediction["recommendation"])
        self.assertEqual(prediction["recommendation_policy"], "disabled_for_unvalidated_baseline")

    def test_expired_odds_are_not_used_for_market_fusion(self) -> None:
        match = {
            "match_id": "m-stale",
            "home_team": "A",
            "away_team": "B",
            "status": "scheduled",
            "home_xg": 1.8,
            "away_xg": 0.8,
        }
        stale_odds = {
            "home": 2.0,
            "draw": 3.2,
            "away": 4.0,
            "updated_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "cache_status": "fresh",
        }
        with_odds = predict_match(match, stale_odds)
        without_odds = predict_match(match)
        self.assertFalse(with_odds["market_used"])
        self.assertIsNone(with_odds["market_probabilities"])
        self.assertEqual(with_odds["odds_status"], "stale_not_fused")
        self.assertEqual(with_odds["probabilities"], without_odds["probabilities"])

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
    def test_user_sample_schedule_is_only_a_demo_contract_fixture(self) -> None:
        demo_matches = schedule._validate_matches([
            {
                "match_id": "demo_001",
                "league": "PL",
                "home_team": "Man City",
                "away_team": "Arsenal",
                "status": "scheduled",
                "kickoff": "2026-11-01T16:30:00+08:00",
                "home_xg": 2.1,
                "away_xg": 1.2,
            },
            {
                "match_id": "demo_002",
                "league": "PD",
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "status": "scheduled",
                "kickoff": "2026-11-01T21:00:00+08:00",
                "home_xg": 1.7,
                "away_xg": 1.5,
            },
            {
                "match_id": "demo_003",
                "league": "BL1",
                "home_team": "Bayern",
                "away_team": "Dortmund",
                "status": "scheduled",
                "kickoff": "2026-11-01T23:30:00+08:00",
            },
            {
                "match_id": "demo_004",
                "league": "SA",
                "home_team": "Inter",
                "away_team": "Juventus",
                "status": "live",
                "kickoff": "2026-11-01T15:00:00+08:00",
            },
        ])
        self.assertEqual(len(demo_matches), 4)
        self.assertEqual([match["status"] for match in demo_matches].count("live"), 1)
        self.assertEqual(demo_matches[0]["home_xg"], 2.1)
        self.assertTrue(all(match["match_id"].startswith("demo_") for match in demo_matches))

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

    def test_no_cache_bypasses_fresh_cached_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "data" / "cache" / "schedule_2026-11-01.json"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(json.dumps({
                "date": "2026-11-01",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "matches": [{
                    "match_id": "cached", "home_team": "A", "away_team": "B",
                    "status": "scheduled",
                }],
            }), encoding="utf-8")
            with patch.object(schedule, "BASE_DIR", root), \
                    patch.dict(os.environ, {"FOOTBALL_SCHEDULE_URL": "https://provider.test/schedule"}, clear=True), \
                    patch.object(schedule, "_http_json", return_value={"matches": [{
                        "match_id": "fresh", "home_team": "C", "away_team": "D",
                        "status": "scheduled",
                    }]}) as request:
                snapshot = schedule.fetch_jingcai_schedule(
                    "2026-11-01", load_params(), use_cache=False
                )
        self.assertEqual(snapshot["matches"][0]["match_id"], "fresh")
        request.assert_called_once()

    def test_test_schedule_empty_result_exits_nonzero(self) -> None:
        snapshot = {
            "matches": [], "source": "provider",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "freshness": "fresh", "age_seconds": 0.0,
        }
        output = StringIO()
        with patch.object(sys, "argv", ["run_today.py", "--test-schedule"]), \
                patch.object(run_today, "fetch_jingcai_schedule", return_value=snapshot), \
                redirect_stdout(output):
            exit_code = run_today.main()
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "empty")

    def test_test_schedule_demo_fixture_is_not_reported_as_live_success(self) -> None:
        snapshot = {
            "matches": [{
                "match_id": "demo-1", "home_team": "A", "away_team": "B",
                "status": "scheduled",
            }],
            "source": "demo_fixture",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "freshness": "demo", "age_seconds": 0.0,
        }
        output = StringIO()
        with patch.object(sys, "argv", ["run_today.py", "--test-schedule"]), \
                patch.object(run_today, "fetch_jingcai_schedule", return_value=snapshot), \
                redirect_stdout(output):
            exit_code = run_today.main()
        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "demo")
        self.assertEqual(report["source"], "demo_fixture")


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

    def test_user_sample_results_are_demo_only_settlement_inputs(self) -> None:
        demo_predictions = [
            {
                "match_id": match_id,
                "status": "predicted",
                "date": "2026-11-01",
                "probabilities": probabilities,
            }
            for match_id, probabilities in (
                ("demo_001", {"home": 0.5, "draw": 0.3, "away": 0.2}),
                ("demo_002", {"home": 0.4, "draw": 0.3, "away": 0.3}),
                ("demo_003", {"home": 0.6, "draw": 0.2, "away": 0.2}),
            )
        ]
        demo_results = [
            {"match_id": "demo_001", "home_goals": 2, "away_goals": 1},
            {"match_id": "demo_002", "home_goals": 1, "away_goals": 1},
            {"match_id": "demo_003", "home_goals": 3, "away_goals": 1},
        ]
        report = settle_predictions(demo_predictions, demo_results)
        self.assertEqual(report["metrics"]["count"], 3)
        self.assertEqual(
            [item["actual_outcome"] for item in report["matches"]],
            ["home", "draw", "home"],
        )
        self.assertTrue(all(item["match_id"].startswith("demo_") for item in report["matches"]))

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
