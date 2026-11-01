"""Small filesystem, JSON, and date helpers shared by the package."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.config import BASE_DIR


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | os.PathLike[str]) -> Path:
    file_path = Path(path)
    ensure_dir(file_path.parent if str(file_path.parent) else Path("."))
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=file_path.parent or Path("."), delete=False
        ) as file:
            temp_name = file.name
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temp_name, file_path)
    except (OSError, TypeError, ValueError) as error:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise ValueError(f"Could not write JSON file {file_path}: {error}") from error
    return file_path


def load_json(path: str | os.PathLike[str]) -> Any:
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON file {file_path}: {error}") from error


def today_str() -> str:
    return date.today().isoformat()


def validate_date(value: str | date | None = None) -> str:
    if value is None:
        return today_str()
    if isinstance(value, datetime):
        raise ValueError("date must be a calendar date, not a datetime.")
    if isinstance(value, date):
        return value.isoformat()
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("date must use YYYY-MM-DD format.") from error
    if parsed.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD format.")
    return value


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_data_dir(*parts: str) -> Path:
    return BASE_DIR / "data" / Path(*parts)


def parse_timestamp(value: str, field: str = "timestamp") -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone.")
    return parsed.astimezone(timezone.utc)
