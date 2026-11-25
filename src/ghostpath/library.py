"""Inventory and removal of saved artifacts while the workspace is disconnected."""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from .models import SavedKind

ROOTS: dict[SavedKind, str] = {"rest": "poses", "recording": "datasets"}
REST_NAME = re.compile(r"([\w.-]{1,80})-([a-f0-9]{64}|simulation)-rest\.json")
DATASET_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class SavedItem(TypedDict):
    id: str
    label: str
    modified: str
    episodes: int | None


def _describe(path: Path, kind: SavedKind) -> SavedItem | None:
    if path.is_symlink():
        return None
    episodes = None
    modified = path
    if kind == "rest":
        match = REST_NAME.fullmatch(path.name)
        if not match or not path.is_file():
            return None
        robot, calibration = match.groups()
        if calibration == "simulation" and robot != "simulation":
            return None
        label = "Simulator" if calibration == "simulation" else f"{robot} · {calibration[:8]}"
    else:
        if not DATASET_NAME.fullmatch(path.name) or not path.is_dir():
            return None
        meta = path / "meta"
        modified = meta / "info.json"
        if meta.is_symlink() or modified.is_symlink() or not modified.is_file():
            return None
        info = json.loads(modified.read_text())
        if not isinstance(info, dict) or info.get("codebase_version") != "v3.0":
            return None
        label = path.name
        episodes = info.get("total_episodes")
        if type(episodes) is not int or episodes < 0:
            return None
    stamp = datetime.fromtimestamp(modified.stat().st_mtime, UTC).isoformat()
    return {"id": path.name, "label": label, "modified": stamp, "episodes": episodes}


def saved_library(data: Path) -> dict[SavedKind, list[SavedItem]]:
    library = {}
    for kind, directory in ROOTS.items():
        entries = []
        root = data / directory
        if not root.is_symlink() and root.is_dir():
            for path in root.iterdir():
                try:
                    entry = _describe(path, kind)
                except (OSError, ValueError):
                    continue
                if entry is not None:
                    entries.append(entry)
        library[kind] = sorted(entries, key=lambda entry: entry["modified"], reverse=True)
    return library


def delete_saved_item(data: Path, kind: SavedKind, ident: str):
    root = data / ROOTS[kind]
    path = root / ident
    if Path(ident).name != ident or root.is_symlink() or _describe(path, kind) is None:
        raise ValueError("That saved item is unavailable. Refresh the list and try again.")
    if kind == "rest":
        path.unlink()
    else:
        shutil.rmtree(path)
