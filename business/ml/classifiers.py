"""
Classifier factories for the categorical heads of the ML engine.

Right now the only categorical head is `multikill` (Low/Medium/High
— 0.3.0).  First-to-15 (radiant/dire) lands in 0.3.1 once the
training corpus carries the per-kill timeline we need.

The 3-class `multikill` problem is small enough that a single
estimator handles it well.  We use
`HistGradientBoostingClassifier` for the same reasons the
regressors do: it has a joblib-friendly contract, supports
categorical-ish feature spaces natively, and trains in under
a second on the 1111-match corpus.

Class labels are passed through `fit()` as a list of strings
("Low", "Medium", "High") — sklearn handles the integer
encoding internally; the engine reverses it with the `classes_`
attribute when it needs to render the prediction.
"""

from __future__ import annotations

from typing import Any, List

from sklearn.ensemble import HistGradientBoostingClassifier


# --------------------------------------------------------------------------- #
# Multikill — 3-class classifier (Low / Medium / High)
# --------------------------------------------------------------------------- #

#: Canonical class order.  The integer encoding sklearn uses
#: internally is `np.argsort(classes_)`, so we sort the labels
#: alphabetically to keep the encoding stable across runs.
MULTIKILL_CLASSES: List[str] = ["High", "Low", "Medium"]


def make_multikill_classifier(random_state: int = 42) -> HistGradientBoostingClassifier:
    """3-class classifier for the multikill level.

    The corpus has the order: Low (most matches) > Medium >
    High (rare).  `class_weight="balanced"` compensates for the
    heavy class imbalance so the model doesn't just learn to
    always say "Low" (which would still get >70% accuracy).
    """
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=random_state,
        class_weight="balanced",
    )


# --------------------------------------------------------------------------- #
# First-to-15 — 2-class classifier (0.3.1; deferred until timeline data lands)
# --------------------------------------------------------------------------- #

def make_first_to_15_classifier(random_state: int = 42) -> Any:
    """Placeholder for the first-to-15 / f10 binary classifier.

    We do NOT train this in 0.3.0 because the corpus lacks a
    per-kill timeline — the 15th kill's team is what we'd
    predict, but `ml_data/full_matches/*.json` only carries the
    final kill counts, not the order.  This factory returns
    something so the registry stays complete; training will
    be enabled once the data lands.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        random_state=random_state,
    )


# --------------------------------------------------------------------------- #
# Public registry
# --------------------------------------------------------------------------- #

#: Maps the CLI `--target` flag to the factory function.
#: `y_attr` on the registry entry tells the trainer which
#: `MatchTarget` field holds the y-vector.  0.3.0 has just the
#: multikill classifier; first_to_15 will be enabled once the
#: training corpus carries the per-kill timeline.
CLASSIFIER_REGISTRY: dict[str, tuple] = {
    "multikill": (make_multikill_classifier, "multikill_level"),
}


def make_classifier(target: str, random_state: int = 42):
    """Factory lookup by name. Raises `ValueError` on unknown target."""
    if target not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"unknown classification target {target!r}; "
            f"expected one of {sorted(CLASSIFIER_REGISTRY)}"
        )
    factory, _ = CLASSIFIER_REGISTRY[target]
    return factory(random_state)
