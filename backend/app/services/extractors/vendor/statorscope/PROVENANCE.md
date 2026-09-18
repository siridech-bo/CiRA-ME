# Vendored: statorscope

**Upstream:** https://github.com/ali-kin4/statorscope
**Vendored commit:** `8a296ba4d42c89eceab9a77ee49d084ff720e5c5` (2026-08-25)
**Upstream version:** 0.3.0
**License:** Apache-2.0 (see [LICENSE](LICENSE) — retained verbatim)
**Author:** Ali Jabbary

## Why vendored (not a pip dependency)

Per [docs/PLAN_2026-09-17_solutions-catalog.md](../../../../../../docs/PLAN_2026-09-17_solutions-catalog.md)
**vendor-and-freeze policy**: statorscope is a small single-author repo. We copy
it in, freeze it at a known-good commit, own the copy, and test it against our
extractor contract — rather than tracking it as a live dependency whose
maintenance we do not control. Apache-2.0 permits this; the LICENSE is retained.

## What was changed from upstream

- **Dropped `cli.py`** — it imports the optional `typer`/`rich` extras we do not
  install. Nothing in the package's `__init__.py` imports it, so removal is clean.
- Everything else (`signals`, `spectrum`, `faults`, `detect`, `calibrate`,
  `quality`, `synth`, `datasets`) copied verbatim. All intra-package imports are
  relative, so the package works unchanged as a sub-package.
- Runtime deps: `numpy>=1.26`, `scipy>=1.11` only — both already in the backend.

## How we use it

Wrapped by [`../../mcsa.py`](../../mcsa.py), the `mcsa` feature extractor, which
adapts our `FeatureExtractor` contract (window, fs, params) → statorscope
`diagnose()` → a flat physically-meaningful feature dict.

## Updating

To bump: re-clone upstream, re-drop `cli.py`, diff against this tree, re-run the
`mcsa` extractor tests, and update the commit/version above. Do NOT edit the
vendored source in place — keep it a faithful frozen copy so diffs stay meaningful.
