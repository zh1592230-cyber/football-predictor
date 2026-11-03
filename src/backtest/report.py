"""Build a descriptive JSON/Markdown report from one-to-one settled matches."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.backtest.metrics import calculate_metrics
from src.backtest.settle import settle_predictions
from src.utils import get_data_dir, save_json, validate_date


def _format_score(value: Any) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"# Backtest report: {report['date']}",
        "",
        f"- Matches settled: {metrics['sample_count']}",
        "- Purpose: descriptive historical evaluation only",
        "- Automatic parameter update: disabled",
        "",
        "## Forecast metrics",
        "",
        "| Forecast | Count | Multiclass Brier (lower is better) | Log loss (lower is better) | Hit rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("Model", "model"), ("Final forecast", "final_forecast"), ("Market baseline", "market_baseline")):
        item = metrics[key]
        lines.append(
            f"| {label} | {item['count']} | {_format_score(item['brier'])} | "
            f"{_format_score(item['log_loss'])} | {_format_score(item['hit_rate'])} |"
        )
    paired = metrics["model_vs_market_on_same_matches"]
    lines.extend(["", "## Paired model vs market comparison", ""])
    if paired is None:
        lines.append("Not available: no matches contain market probabilities.")
    else:
        lines.extend([
            f"Compared on the same {paired['count']} matches.",
            f"- Model minus market Brier: {_format_score(paired['model_minus_market_brier'])}",
            f"- Model minus market log loss: {_format_score(paired['model_minus_market_log_loss'])}",
            "- Negative differences favor the model on this sample; this is not evidence of future advantage.",
        ])
    lines.extend(["", "## One-vs-rest calibration by outcome", ""])
    calibration = metrics["calibration"]["outcomes"]
    for outcome in ("home", "draw", "away"):
        lines.extend([
            f"### {outcome.title()}",
            "",
            "| Probability bin | Count | Mean predicted | Observed frequency |",
            "|---|---:|---:|---:|",
        ])
        for item in calibration[outcome]:
            lines.append(
                f"| [{item['bin_lower']:.1f}, {item['bin_upper']:.1f}] | {item['count']} | "
                f"{item['mean_predicted']:.4f} | {item['observed_frequency']:.4f} |"
            )
        if not calibration[outcome]:
            lines.append("| No observations | 0 | n/a | n/a |")
        lines.append("")
    lines.extend(["## Warnings", ""])
    lines.extend(f"- {warning}" for warning in metrics["warnings"])
    if not metrics["warnings"]:
        lines.append("- None.")
    lines.extend(["", "## Interpretation", "", metrics["interpretation"], ""])
    return "\n".join(lines)


def write_backtest_report(
    predictions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    report_date: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    if not isinstance(predictions, list):
        raise ValueError("Predictions input must be a JSON array.")
    if not isinstance(results, list):
        raise ValueError("Results input must be a JSON array.")
    settled = settle_predictions(predictions, results)
    metrics = calculate_metrics(settled["matches"])
    date_label = validate_date(report_date) if report_date is not None else datetime.now(timezone.utc).date().isoformat()
    generated_at = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "report_type": "descriptive_backtest",
        "date": date_label,
        "generated_at": generated_at.isoformat(),
        "metrics": metrics,
        "matches": settled["matches"],
        "auto_param_update": False,
        "optimization_performed": False,
    }

    parent = Path(output_dir) if output_dir is not None else get_data_dir("reports")
    parent.mkdir(parents=True, exist_ok=True)
    stamp = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
    target_dir = parent / f"backtest_{date_label}_{stamp}"
    temp_dir = Path(tempfile.mkdtemp(prefix=".backtest-report-", dir=parent))
    try:
        save_json(report, temp_dir / "report.json")
        markdown_path = temp_dir / "report.md"
        with markdown_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(_markdown(report))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_dir, target_dir)
    except (OSError, ValueError, TypeError) as error:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise OSError(f"Could not publish backtest report in {parent}: {error}") from error
    return report, target_dir
