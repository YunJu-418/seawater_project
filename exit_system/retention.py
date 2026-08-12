from pathlib import Path
import time


def cleanup_old_exit_data(retention_days: int = 7) -> tuple[int, int]:
    """Delete local exit images/events older than retention_days."""
    cutoff = time.time() - (retention_days * 24 * 60 * 60)

    deleted_images = 0
    deleted_events = 0

    for path in Path("data/images").glob("exit_*.jpg"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                deleted_images += 1
        except OSError as exc:
            print(f"[RETENTION WARNING] image delete failed: {path}: {exc}")

    for path in Path("data/events").glob("exit_*.json"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                deleted_events += 1
        except OSError as exc:
            print(f"[RETENTION WARNING] event delete failed: {path}: {exc}")

    return deleted_images, deleted_events
