"""Loading of config.yaml and the artifact paths derived from it."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
TASKS = ("binary", "multiclass")


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path

    @property
    def dataset(self) -> Path:
        return self.processed_dir / "flows.parquet"

    @property
    def profile(self) -> Path:
        return self.processed_dir / "profile.json"

    @property
    def explorer_sample(self) -> Path:
        return self.processed_dir / "explorer_sample.parquet"

    @property
    def demo_stream(self) -> Path:
        return self.processed_dir / "demo_stream.parquet"

    def task_models(self, task: str) -> Path:
        return self.models_dir / task

    def task_reports(self, task: str) -> Path:
        return self.reports_dir / task

    @property
    def research_reports(self) -> Path:
        return self.reports_dir / "research"


@dataclass(frozen=True)
class Config:
    path: Path
    paths: Paths
    data: dict
    training: dict
    models: dict
    live_monitor: dict
    research: dict


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the config from `path`, $NIDS_CONFIG, or the project's config.yaml."""
    path = Path(path or os.environ.get("NIDS_CONFIG") or DEFAULT_CONFIG).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base = path.parent
    paths = Paths(**{key: (base / value).resolve() for key, value in raw["paths"].items()})
    return Config(
        path=path,
        paths=paths,
        data=raw.get("data") or {},
        training=raw.get("training") or {},
        models=raw.get("models") or {},
        live_monitor=raw.get("live_monitor") or {},
        research=raw.get("research") or {},
    )
