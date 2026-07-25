"""
ML prediction package — Strategy pattern implementation.

Replaces the old isolated `ml_trainer.py`. Provides:
  - `IPredictionEngine` — abstract base
  - `HeuristicEngine` — wraps the existing `analysis.analyze()` (current behaviour)
  - `MLEngine` — uses a trained sklearn model on real match data
  - `HeroWinRateEncoder` — per-hero, per-side target encoding for the 0.2.0 MVP
  - `ModelStorage` — versioned load/save with a `metadata.json` sidecar

The engine is selected at process start via the `PREDICTION_ENGINE` env var
(`heuristic` or `ml`). The two engines are interchangeable from the
business service's perspective — same input, same output shape.
"""
