from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path


def save_event(payload: dict, directory: str = "data/events") -> Path:
    Path(directory).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(directory) / f"exit_{stamp}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
