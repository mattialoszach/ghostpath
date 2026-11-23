"""Calibration-specific start poses and atomic JSON artifacts."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .motion import Pose


@dataclass(frozen=True)
class SavedStart:
    id: str
    label: str
    created: str
    position: Pose
    legacy: bool = False

    def summary(self):
        return {
            "id": self.id, "label": self.label, "created": self.created, "legacy": self.legacy,
        }


def fingerprint(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def calibration_data(robot) -> dict:
    return {name: asdict(calibration) for name, calibration in robot.calibration.items()}


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
