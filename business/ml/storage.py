"""
Versioned model storage for the ML prediction engine.

A trained model + its fitted encoder are saved together under a
versioned directory:

    ml_data/models/
        winner_v1/
            model.joblib      # the sklearn estimator
            metadata.json     # sklearn_version, feature_names, metrics, encoder

Why versioned directories?  So that:

  - production can pin a known-good version
  - a freshly trained candidate lives alongside the production model
  - a smoke test can load both, compare, and roll back if needed

Why a sidecar `metadata.json` next to the model file?  Two reasons:

  1. We want to log training context (sklearn version, label balance,
     n_samples) WITHOUT re-loading the joblib blob.
  2. We can quickly list available models and their metrics without
     importing sklearn / numpy.

The `HeroWinRateEncoder` is round-tripped through `to_dict()` / `from_dict()`
inside the metadata so the encoder and model can never drift out of sync —
load returns them as a pair.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib  # sklearn's preferred serialiser; handles numpy arrays inside
import numpy as np  # used only for version reporting

from .features import FEATURE_ORDER, N_FEATURES, HeroWinRateEncoder


# ---------------------------------------------------------------------------- #
# Public types
# ---------------------------------------------------------------------------- #

@dataclass
class ModelMetadata:
    """Everything we need to know about a saved model without loading it.

    `metrics` is intentionally untyped — it varies by problem (log_loss for
    classification, MAE/RMSE for regression, CRPS for quantiles). The shape
    is documented per release in the trainer.
    """

    name: str                     # logical name, e.g. "winner"
    version: str                  # string, e.g. "1", "2026-07-24-001"
    trained_at: str               # ISO 8601 UTC
    sklearn_version: str
    numpy_version: str
    python_version: str
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    train_data: Dict[str, Any] = field(default_factory=dict)
    encoder: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "trained_at": self.trained_at,
            "sklearn_version": self.sklearn_version,
            "numpy_version": self.numpy_version,
            "python_version": self.python_version,
            "feature_names": list(self.feature_names),
            "n_features": int(self.n_features),
            "metrics": self.metrics,
            "train_data": self.train_data,
            "encoder": self.encoder,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            name=d["name"],
            version=d["version"],
            trained_at=d["trained_at"],
            sklearn_version=d["sklearn_version"],
            numpy_version=d["numpy_version"],
            python_version=d["python_version"],
            feature_names=list(d.get("feature_names", [])),
            n_features=int(d.get("n_features", 0)),
            metrics=dict(d.get("metrics", {})),
            train_data=dict(d.get("train_data", {})),
            encoder=dict(d.get("encoder", {})),
        )


@dataclass
class LoadedModel:
    """A model + its metadata + its fitted encoder, all in one bundle."""

    model: Any                    # the sklearn estimator (or None on miss)
    encoder: HeroWinRateEncoder
    metadata: ModelMetadata
    path: Path                    # directory on disk

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version


# ---------------------------------------------------------------------------- #
# Storage
# ---------------------------------------------------------------------------- #

class ModelStorage:
    """File-system backed versioned model store.

    Default root is `ml_data/models` (relative to project root). Override
    with the `MODEL_DIR` env var (the app reads it in `app.py`).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- directory layout helpers -------------------------------------- #

    def _version_dir(self, name: str, version: str) -> Path:
        return self.root / f"{name}_v{version}"

    def _model_file(self, name: str, version: str) -> Path:
        return self._version_dir(name, version) / "model.joblib"

    def _metadata_file(self, name: str, version: str) -> Path:
        return self._version_dir(name, version) / "metadata.json"

    # ---- list / discover ---------------------------------------------- #

    def list_versions(self, name: str) -> List[str]:
        """Return sorted version strings for a model name. Empty if none.

        Versions are sorted numerically when they parse as integers
        (`"9"` < `"10"` < `"11"`); otherwise lexicographic.  This
        matters for `latest_version()`: with lexicographic sort
        `"9" > "11"`, so a freshly-trained `winner_v11` would be
        ignored in favour of the older `winner_v9`.
        """
        prefix = f"{name}_v"
        out: List[str] = []
        if not self.root.is_dir():
            return out
        for child in self.root.iterdir():
            if not child.is_dir() or not child.name.startswith(prefix):
                continue
            v = child.name[len(prefix):]
            if (child / "metadata.json").is_file():
                out.append(v)
        # Numeric sort when all versions parse as ints; fallback to
        # lexicographic for ISO-style timestamped versions.
        try:
            return sorted(out, key=int)
        except ValueError:
            return sorted(out)

    def latest_version(self, name: str) -> Optional[str]:
        versions = self.list_versions(name)
        return versions[-1] if versions else None

    def exists(self, name: str, version: str) -> bool:
        return (
            self._model_file(name, version).is_file()
            and self._metadata_file(name, version).is_file()
        )

    # ---- save ---------------------------------------------------------- #

    def save(
        self,
        name: str,
        version: str,
        model: Any,
        encoder: HeroWinRateEncoder,
        metrics: Dict[str, Any],
        train_data: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ) -> Path:
        """Atomically write model + metadata sidecar to a new version dir.

        Returns the path of the created version directory.

        `feature_names` defaults to the canonical `FEATURE_ORDER`
        (24 features in 0.3.10) but the trainer can pass a subset
        for the 0.3.9 baseline / C / D v2 experiments.  The exact
        order is part of the model contract — the trainer is
        responsible for keeping it in sync with `extract_features`.
        """
        if feature_names is None:
            feature_names = list(FEATURE_ORDER)
        # Light validation: every name must be in the canonical
        # FEATURE_ORDER (no typos), and the count must match the
        # model's n_features_in_ if the model supports that attr.
        from .features import FEATURE_GROUPS  # local: avoid cycle at import
        canonical = {n for grp in FEATURE_GROUPS.values() for n in grp}
        unknown = [n for n in feature_names if n not in canonical]
        if unknown:
            raise ValueError(
                f"feature_names has {len(unknown)} entries not in "
                f"FEATURE_GROUPS: {unknown[:5]}{'...' if len(unknown) > 5 else ''}"
            )
        n_features = len(feature_names)
        if hasattr(model, "n_features_in_") and model.n_features_in_ != n_features:
            raise ValueError(
                f"model n_features_in_={model.n_features_in_} disagrees with "
                f"feature_names count {n_features}"
            )

        import sys
        import sklearn  # local import keeps the top of the file light

        version_dir = self._version_dir(name, version)
        # If we're clobbering an existing version, that's a programmer
        # error — train scripts should mint a new version string.
        if version_dir.exists():
            raise FileExistsError(
                f"{version_dir} already exists; pick a new version string"
            )

        version_dir.mkdir(parents=True)
        # Write into a tmp file in the same dir, then atomic-rename.
        # joblib is unsafe across filesystems; keeping the tmp in the
        # target dir means `os.replace` is a true atomic move.
        model_path = self._model_file(name, version)
        with tempfile.NamedTemporaryFile(
            dir=str(version_dir), delete=False, suffix=".joblib.tmp"
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            joblib.dump(model, tmp_path)
            os.replace(tmp_path, model_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        meta = ModelMetadata(
            name=name,
            version=version,
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sklearn_version=sklearn.__version__,
            numpy_version=np.__version__,
            python_version=sys.version.split()[0],
            feature_names=list(feature_names),
            n_features=n_features,
            metrics=dict(metrics),
            train_data=dict(train_data),
            encoder=encoder.to_dict(),
        )
        meta_path = self._metadata_file(name, version)
        with tempfile.NamedTemporaryFile(
            dir=str(version_dir), delete=False, suffix=".json.tmp", mode="w", encoding="utf-8"
        ) as tmp:
            json.dump(meta.to_jsonable(), tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        try:
            os.replace(tmp_path, meta_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return version_dir

    # ---- load ---------------------------------------------------------- #

    def load(self, name: str, version: Optional[str] = None) -> Optional[LoadedModel]:
        """Load a saved model. Returns None if the version is missing.

        If `version` is None, loads the lexicographically latest version.

        The `feature_names` recorded in the model metadata must
        still be a subset of the current `FEATURE_GROUPS` — this
        catches the "stale model" footgun where someone trains
        against a feature name that no longer exists at predict
        time.  We do NOT require exact equality with `FEATURE_ORDER`
        (which is the full 24-feature set in 0.3.10) because a
        model trained on a subset is a valid scenario.
        """
        if version is None:
            version = self.latest_version(name)
        if version is None or not self.exists(name, version):
            return None

        model = joblib.load(self._model_file(name, version))
        with self._metadata_file(name, version).open("r", encoding="utf-8") as fh:
            meta = ModelMetadata.from_dict(json.load(fh))
        encoder = HeroWinRateEncoder.from_dict(meta.encoder)

        # 0.3.15+: optional PlayerWinRateEncoder is stored alongside in
        # `player_encoder.json`.  Attach it to the hero encoder so
        # `_features_player` can use it.  Missing file is fine — older
        # models (v1..v13) just have `encoder.player_encoder = None`.
        pe_path = self._version_dir(name, version) / "player_encoder.json"
        if pe_path.is_file():
            with open(pe_path, encoding="utf-8") as fh:
                pe_dict = json.load(fh)
            from .features import PlayerWinRateEncoder  # local: avoid cycle
            encoder.player_encoder = PlayerWinRateEncoder.from_dict(pe_dict)

        # Sanity check: every feature name must still resolve to a
        # known group.  Subset is fine (0.3.9 baseline = hero only;
        # 0.3.10 experiments = hero+team / hero+lane / all).
        from .features import FEATURE_GROUPS  # local: avoid cycle
        canonical = {n for grp in FEATURE_GROUPS.values() for n in grp}
        unknown = [n for n in meta.feature_names if n not in canonical]
        if unknown:
            raise RuntimeError(
                f"model {name} v{version} was trained on feature names that "
                f"are no longer in FEATURE_GROUPS: {unknown[:5]}"
            )
        if hasattr(model, "n_features_in_") and model.n_features_in_ != meta.n_features:
            raise RuntimeError(
                f"model {name} v{version} has n_features_in_={model.n_features_in_} "
                f"but metadata says n_features={meta.n_features}"
            )

        return LoadedModel(
            model=model,
            encoder=encoder,
            metadata=meta,
            path=self._version_dir(name, version),
        )

    # ---- housekeeping -------------------------------------------------- #

    def delete(self, name: str, version: str) -> bool:
        """Remove a version directory. Returns True if something was deleted."""
        d = self._version_dir(name, version)
        if not d.is_dir():
            return False
        shutil.rmtree(d)
        return True
