# PLAN — MOMENT Foundation Model as TimesNet Alternative

**Status**: parked. Bugs 1-3 from the 2026-07-XX workshop debrief take priority.
**Owner**: siridech-bo
**Author**: Claude (with product discussion)
**Depends on**: nothing new — extends the existing DL training path.
**Follow-on** (deferred to Phase D+, may never build): TSQA (Time Series QA). Dropped from this plan per product decision.

---

## Why

Customer asked for LLM/transformer-based analysis as an alternative to TimesNet — bypass feature engineering, use transferable inductive bias from a pretrained model. Traditional ML + hand-crafted features is not robust enough for their signals.

Instead of building yet another end-to-end architecture from scratch (like TimesNet was), we borrow a **pretrained time-series foundation model** and light-fine-tune it on the customer's data. Gets the "transformer" ask satisfied AND gets the pretraining benefit that TimesNet lacks.

**Time-MQA** (the paper the customer pointed at — `https://huggingface.co/papers/2503.01875`) was considered and rejected as the CORE. It's Mistral-7B / Llama-3-8B / Qwen-2.5-7B fine-tuned on a QA dataset. 7-8B params kills our "everything runs at workshop scale" story: browser deployment needs 4-6 GB VRAM, backend deployment serializes requests through one GPU. It stays on the table as a possible admin-tier "premium analysis" feature for on-prem customers with dedicated GPUs.

## Target model

**MOMENT-small (40M, MIT)** — `AutonLab/MOMENT-1-small` on HuggingFace.

Why this one:
- **MIT license** — no restrictions on commercial use.
- **Classification-native** — trained specifically for anomaly/classification/forecasting tasks. TimesFM/Chronos are forecasting-first; less ideal for our tasks.
- **Small enough for on-prem AND browser** — 40M params ≈ 160 MB fp32, ~40 MB int8 quantized. Browser inference clean.
- **Windowed input** — designed for exactly 128-512 timestep chunks. Matches the CiRA ME windowing model.
- **HuggingFace-native** — `AutoModel.from_pretrained("AutonLab/MOMENT-1-small")` and go.

Alternatives kept in reserve for later phases:
- **Chronos-tiny (8M)** — even smaller; add if we ever add a forecasting mode.
- **Moirai-small (14M)** — Salesforce; consider if masked anomaly detection becomes a first-class task.

## Light fine-tuning strategy

**Freeze the backbone; train a small head only.**

```
[MOMENT-small backbone — FROZEN, 40M params]         (pretrained knowledge)
              ↓
     [pooled representation — 1024-dim]
              ↓
[task head — TRAINABLE, ~50K params]                  (classification head OR
              ↓                                        regression head OR
        class probs / value / anomaly score            anomaly-score head)
```

- Trainable params: ~50K vs frozen 40M.
- Training time: **minutes on CPU** for typical CiRA ME datasets (hundreds of windows).
- No GPU strictly required (though it helps for training runs > few thousand windows).

Why not LoRA / full fine-tune?
- LoRA is overkill for a 40M model; the win only kicks in at 7B+.
- Full fine-tuning risks catastrophic forgetting of the pretrained inductive bias — the whole reason we're using a foundation model.
- Head-only ("linear probing") is the standard MOMENT paper recipe and matches how customers actually use these models in production.

## How it fits the existing pipeline

New training approach `'foundation'` alongside the current `'ml' | 'dl' | 'custom' | 'ti'` in the pipeline store.

**User flow (unchanged UX pattern):**
1. Data Source → Windowing (existing).
2. **Skip Features** (like DL/TimesNet — foundation model consumes raw windows).
3. Training view → choose Foundation Model → pick MOMENT-small → click Train.
4. Progress bar → done → model appears in ME-LAB alongside all others.
5. Deploy via App Builder like any other model.

**Backend pipeline:**
```
Windowed data (N × 128 × 3)
      ↓
MOMENT backbone forward pass    ← pretrained weights, cached under models/foundation/
      ↓
Pooled features (N × 1024)
      ↓
Task head training              ← trainable, mode-aware (classification/regression/anomaly)
      ↓
Save pickle: {head_weights, backbone_ref, label_map, config}
```

## Files to touch

### Backend
- **NEW** `backend/app/services/foundation_trainer.py` (~400 lines)
  - `train_foundation(windowed_session_id, base_model, mode, head_config)` — mirrors `timesnet_trainer.train_timesnet` API so it slots in cleanly.
  - Handles all three modes (classification / regression / anomaly detection).
  - First run downloads MOMENT weights (~160 MB) to `models/foundation/moment-small/`. Cached forever after.
- **NEW** `backend/app/services/foundation_inference.py` (~200 lines)
  - `predict(model_pickle, windows)` — used by App Builder runtime.
  - Loads backbone once per process, holds in memory (~160 MB RAM cost).
- **MODIFY** `backend/app/routes/training.py`
  - Add `foundation` as new `training_approach` in existing endpoints. ~30 lines.
- **MODIFY** `backend/app/routes/app_builder.py`
  - Add `model.endpoint.foundation` node type handler alongside existing `model.endpoint.*` variants. ~50 lines.

### Frontend
- **MODIFY** `frontend/src/views/pipeline/TrainingView.vue`
  - Add "Foundation Model" tile to the training approach picker.
  - Config panel: model selector (MOMENT-small default), pooling strategy (CLS vs mean, default CLS), head layers (linear default), epochs, learning rate.
  - Progress display reuses TimesNet's existing pattern.
- **MODIFY** `frontend/src/stores/pipeline.ts`
  - Add `'foundation'` to `TrainingApproach` type. Update step-status logic (Features step skipped, same as DL).

### Storage
- **New directory** `models/foundation/moment-small/` — cached backbone, shared across all projects.
- **Per-project** trained heads live inside existing SavedModel storage (~100 KB per model).

## Phase breakdown

### Phase A — Backend integration + MVP training (~2 weeks)
Implement `foundation_trainer.py` + `foundation_inference.py`. Wire into training routes + App Builder endpoint execution. Frontend training UI. Verified end-to-end: user trains MOMENT-small on their data, deploys, runs inference server-side.

**Milestone**: A customer can train and deploy a foundation-based classifier through the existing UI. Server-side inference only.

### Phase B — Browser inference via ONNX (~1 week)
Add ONNX export path in `foundation_trainer.py` (`torch.onnx.export`). ONNX Runtime Web + WebGPU integration in `PublishedAppView.vue`. Extend Fast Mode toggle: today it means "compute features in browser"; now it also means "run foundation inference in browser". Backend queue used only for training + fallback.

**Milestone**: Live MQTT inference on foundation models runs entirely browser-side. 65-attendee workshop safe.

### Phase C — Model portfolio + polish (~1 week, optional)
Add Chronos-tiny as a second option ("smaller, forecasting-oriented"). Model card UI: "MOMENT-small, 40M params, best for classification and anomaly". Auto-recommend based on selected mode (regression → suggest Chronos, classification → suggest MOMENT).

**Milestone**: Multiple foundation model choices with sensible defaults.

**Total: ~4 weeks calendar time.** Phase A alone is a shippable feature; B and C add scale + polish.

## Product-approved specifics

1. **First target: MOMENT-small only.** No Chronos in Phase A. Multi-model support in Phase C.
2. **Backbone download**: lazy on first training with a "downloading MOMENT-small (160 MB)…" progress bar. Cached forever after. Not baked into Docker image.
3. **TSQA (Time-Series QA) dropped from this plan.** May revisit later as either (a) a separate feature stacked on top of foundation model predictions using a small local LLM (WebLLM), or (b) an admin-tier Time-MQA-Mistral integration for GPU-equipped on-prem customers.

## Open questions to resolve at Phase A kickoff

1. **CPU-only fine-tuning acceptable for MVP?** Head-only training with a frozen 40M backbone takes ~2-5 minutes on CPU for typical dataset sizes (hundreds of windows). If we want GPU-accelerated training, add ~2 days to Phase A for CUDA plumbing. Default: CPU-only, GPU is an optimization for later.
2. **Windowing size constraint**: MOMENT is trained on 512-timestep windows by default. Our CiRA ME pipeline defaults to 128 samples. Need to either (a) pad on load, (b) use MOMENT's shorter-window pretrained variants if available, or (c) recommend users increase window size.
3. **Fine-tuning data floor**: MOMENT paper says "hundreds of samples usually sufficient for classification". Need to test on a real CiRA ME dataset before writing the training UI copy.

## Not doing

- **Full fine-tuning of the backbone** — head-only ("linear probing") is the standard MOMENT recipe. Full FT risks losing the transferable inductive bias.
- **LoRA** — overkill for a 40M model. LoRA's win kicks in at 7B+ where full FT is impossible.
- **Time-MQA-Mistral** — too big for browser (4-6 GB VRAM), too slow for backend workshop scaling (single GPU serialized). Revisit as an admin-tier feature only.
- **RAG / vector DBs / agent frameworks** — those are AnythingLLM territory. Not needed for the "transformer as end-to-end model" ask.
- **Text-token time-series encoding** (like Time-MQA does) — loses numeric precision, caps at 256 timesteps. Foundation models keep full float precision and handle 512-2048+ timesteps.

## Follow-up ideas (not committed)

- Fine-tune a smaller LLM (Qwen-2.5-1.5B) on the released TSQA dataset for a browser-viable TSQA layer if the customer really wants natural-language querying.
- Add support for **multivariate windows with per-channel metadata** so vibration + temperature + audio can be jointly reasoned over.
- Model card page showing pretraining datasets, license, expected performance ranges per task type.
- Admin-tier "premium analysis" feature that routes to a backend Time-MQA-Mistral (or Claude/GPT-4 via user API key) for heavier reasoning tasks. Off by default; on-prem customers with a GPU can enable it.
