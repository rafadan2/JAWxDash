from __future__ import annotations

from typing import Any, Optional


def get_file_entry(stored_files: dict | None, filename: str | None) -> Optional[dict]:
    if not stored_files or not filename:
        return None

    if filename not in stored_files:
        return None

    entry: Any = stored_files[filename]
    if isinstance(entry, dict):
        path = entry.get("path")
        title = entry.get("title") or filename
        return {"path": path, "title": title}

    return {"path": entry, "title": filename}


def get_file_path(stored_files: dict | None, filename: str | None) -> Optional[str]:
    entry = get_file_entry(stored_files, filename)
    return entry["path"] if entry else None


def get_file_title(stored_files: dict | None, filename: str | None) -> Optional[str]:
    entry = get_file_entry(stored_files, filename)
    return entry["title"] if entry else None
