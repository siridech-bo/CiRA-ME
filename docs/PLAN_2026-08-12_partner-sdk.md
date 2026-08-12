# Partner SDK — Trainer Container Contract

**Status:** 🔧 Active — start immediately
**Date:** 2026-08-12
**Owner:** siridech.bo@kmitl.ac.th
**Related tracker item:** to be added under OPEN ITEMS as `F6`

---

## Why this exists

Customer wants to sell a "CiRA ME Partner Edition" where third parties
(their OEMs or ML vendors) can plug in their own proprietary model trainers.
Neither side sees the other's source code. Partner ships their own trainer
container to end-users **independently** of our release cycle.

The pattern already exists internally — `cirame-ti-modelmaker` is exactly this
(TI's trainer in a separate container, HTTP boundary, we don't see their code).
This project generalizes that pattern into a public contract.

## Locked decisions (from the 2026-08-12 discussion)

| # | Decision |
|---|---|
| 1 | **Container plugin architecture** (Option A) — partner ships a Docker image, our backend talks to it over HTTP |
| 2 | **Unified ME-LAB endpoint list** — partner-trained models appear as first-class ME-LAB endpoints, indistinguishable from ours |
| 3 | **We store the model files** — partner trainer returns bytes, we persist under our own `/app/models/` |
| 4 | **Partner containers can bind-mount our `datasets/` folder** — same pattern TI ModelMaker uses (`./datasets:/app/data/datasets`) |
| 5 | **Partners ship independently** — decoupled from our release cycle |

## The killer constraint — "ship independently"

Because partners release on their own schedule:

- **The Trainer Contract must be versioned** and additive-only within a major
  version. We can't rename or remove endpoints without a long deprecation window.
- **We can't force partner containers to update.** If a security fix requires
  their side to change, we can only ask.
- **Our backend maintains a compatibility matrix** — knows which contract
  versions it can talk to. If it sees an unknown version at startup, it
  refuses to register that partner (with a clear message pointing at
  the "supported contract versions" list).
- **The v1 spec must be frozen** before partners start building. Post-freeze
  changes are v1.1 (additive) or v2 (breaking).

## The Trainer Contract v1.0 — endpoints partner container MUST implement

All endpoints listen on a single HTTP port that we discover via `/info`.
All requests + responses are JSON unless noted (binary payloads are `application/octet-stream`).

### `GET /info`

Called once at CiRA ME startup for every registered partner container.
Also called on-demand from the ME-LAB endpoint create UI to populate the
hyperparameter form.

**Response:**
```json
{
  "contract_version": "1.0",
  "provider": "acme-trainer",              // stable machine-readable ID
  "display_name": "ACME Custom Trainer",   // shown to end users
  "algorithm": "acme-conv1d-v3",           // freeform, used for logging/analytics
  "modes": ["classification", "regression"],
  "hyperparameter_schema": {               // used by our UI to render a form
    "learning_rate": {"type": "float", "default": 0.001, "min": 1e-5, "max": 1.0},
    "epochs": {"type": "int", "default": 20, "min": 1, "max": 500}
  },
  "model_format": "onnx",                  // "onnx" | "pickle" | "safetensors"
  "requires_datasets_mount": true,         // whether partner needs bind-mount access
  "license_id": "ACME-LICENSE-A1B2C3"      // for our license validation (see below)
}
```

### `POST /train`

Called when the user clicks Train in the CiRA ME UI.

**Request:**
```json
{
  "session_id": "cirame-train-abc123",     // we generate, partner echoes back for status
  "mode": "classification",                // one of the modes reported in /info
  "windowed_session_id": "ws-xyz789",      // reference to windowed data — partner reads via /app/data/...
  "windows_path": "/app/data/windows/ws-xyz789.npz",  // path inside container (npz format we define)
  "labels_path": "/app/data/windows/ws-xyz789-labels.json",
  "sensor_columns": ["accY", "accZ"],
  "num_classes": 3,
  "class_names": ["idle", "shake", "updown"],
  "hyperparameters": {"learning_rate": 0.001, "epochs": 20}
}
```

**Response:** `202 Accepted` immediately (training is async):
```json
{"session_id": "cirame-train-abc123", "status": "training", "started_at": "..."}
```

### `GET /train/status/<session_id>`

Called by our backend every 5s to poll training progress.

**Response:**
```json
{
  "session_id": "cirame-train-abc123",
  "status": "training",                    // "training" | "done" | "failed"
  "progress": 0.42,                        // 0.0 - 1.0
  "current_epoch": 8,
  "metrics": {                             // partner emits whatever it computes
    "train_loss": 0.234,
    "val_loss": 0.301,
    "val_accuracy": 0.89
  }
}
```

On success:
```json
{
  "session_id": "cirame-train-abc123",
  "status": "done",
  "progress": 1.0,
  "final_metrics": {"val_accuracy": 0.94, "val_f1": 0.93},
  "model_size_bytes": 1245678
}
```

On failure:
```json
{"session_id": "...", "status": "failed", "error": "OOM during epoch 12"}
```

### `GET /train/model/<session_id>`

Called once training is done. Returns the raw model bytes.
Content-Type: `application/octet-stream`. Content-Disposition includes
suggested filename. **We** store this under `/app/models/`.

### `POST /predict/<model_id>`

Called from our unified inference layer (`ModelManager.predict_by_endpoint`).

**Request:**
```json
{
  "model_id": "our-uuid-for-this-model",   // we assigned this
  "model_data_path": "/app/models/model-uuid.bin",  // partner reads from here
  "features": [[...], [...]],              // 2D array of shape (n_samples, n_features)
                                           // OR 3D windows for raw-mode models
  "shape_hint": "windows"                  // "features" | "windows"
}
```

**Response:**
```json
{
  "predictions": [
    {"label": "shake", "confidence": 0.87, "probabilities": {"idle": 0.05, "shake": 0.87, "updown": 0.08}},
    ...
  ]
}
```

(Same shape as `ModelManager.predict` returns today — partner predictions
flow through the existing sklearn / TimesNet unified path.)

### `GET /health`

Standard health check. `200 OK` if partner container is ready. Called by
`docker-compose healthcheck` and by our registry probe.

## Architecture changes on our side

### Backend

- **New `backend/app/services/partner_registry.py`** — reads
  `deployment/partner_trainers.yaml` at startup, probes each partner's
  `/info`, builds an in-memory registry.
- **`ModelManager.predict_by_endpoint` extended** — when an endpoint has
  `provider` field set to a partner ID, dispatch to that partner's
  `/predict/<model_id>` endpoint instead of the sklearn path.
- **New backend routes:**
  - `GET /api/partner-trainers` — list registered partners for the training UI dropdown
  - `POST /api/training/train/partner` — new training endpoint that dispatches to a partner container
  - Existing `/api/training/save-benchmark` handles partner models identically (partner's returned bytes are stored under `/app/models/`).
- **SavedModel table** gets a new `provider` column (nullable, defaults to
  `builtin`). Used by ME-LAB and Wizard to know which runtime path to call.
- **License validation** — each partner container includes a `license_id` in
  its `/info` response. Our backend verifies it against a signed license file
  we ship (Fernet-signed JSON with allowed partner IDs + expiry). Ties into
  the licensing plan in memory. Ships in a later phase — not blocking.

### Frontend

- **Training page:** "Algorithm" dropdown shows built-in options
  (ML/DL/TI/Custom) plus a "Partner" group with registered partner display
  names. Hyperparameter form auto-renders from the partner's
  `hyperparameter_schema`.
- **ME-LAB endpoint list:** partner-trained models appear identically to
  ours. Optional small badge (e.g., "by ACME") in the endpoint card is
  a nice-to-have.

### Partner SDK deliverable

- **`partner-sdk/README.md`** — how to build a partner container
- **`partner-sdk/reference-trainer/`** — reference implementation (Python +
  Flask) that a partner can fork. Implements all 6 endpoints, uses a trivial
  sklearn model. About 200 lines total. Ships as its own Docker image customers
  can `docker compose` alongside CiRA ME.
- **`partner-sdk/contract-v1.0.md`** — the frozen contract spec (essentially
  a cleaned-up version of the "Trainer Contract v1.0" section above)
- **`partner-sdk/docker-compose.example.yml`** — shows how a partner container
  slots into an existing CiRA ME deployment

## Milestones

| # | Milestone | Effort | Ship criteria |
|---|---|---|---|
| 1 | **Contract v1.0 spec frozen** — internal doc, no code yet | 2-3d | Written, reviewed with customer, signed off |
| 2 | **Reference partner container** (Python + Flask + sklearn stub) | 3-4d | Implements all 6 endpoints, passes contract test suite |
| 3 | **Partner registry on our backend** — reads YAML, probes /info | 2-3d | Registered partners appear in `/api/partner-trainers` |
| 4 | **Training dispatch to partner** — new route, hyperparameter form auto-render | 3-4d | User can click Train, see partner train, get model back |
| 5 | **ME-LAB inference dispatch to partner** — predict routing | 2-3d | Partner-trained model works in ME-LAB / Wizard / App Builder |
| 6 | **Contract test suite** — synthetic partner container we run in CI to enforce contract compat | 2d | New CiRA ME versions fail CI if they break v1.0 contract |
| 7 | **Partner SDK repo** (README + reference impl + example compose) | 2-3d | Handoff-ready — a stranger can build a partner container from this |
| 8 | License validation (deferred, separate phase) | ~1w | Not blocking v1 shipping |
| **Total v1** | | **~3 weeks** | Reference partner training + inference works end-to-end |

## Open questions still to resolve

1. **Async training coordination** — if partner container restarts mid-training,
   what happens to the session? Options: (a) our backend polls until N failed
   status checks, then marks failed; (b) partner exposes `POST /train/resume`.
   Option (a) is simpler for v1.
2. **Multi-GPU / resource limits** — do we let the partner container declare
   GPU requirements in `/info`, and our backend enforces at compose time?
   Nice-to-have for v1.
3. **Model format standardization** — should we require ONNX for the model
   artifact? Pros: portable, single runtime for inference. Cons: forces
   partners to add ONNX export. Recommend: accept `onnx` | `pickle` |
   `custom`; if `custom`, always dispatch inference via partner's `/predict`
   (skip our unified runtime).
4. **Partner license enforcement** — plan is a signed JSON we ship. But do we
   want online activation? Recommend offline for v1, matches the existing
   licensing plan.
5. **Where does the reference partner container live?** In our monorepo under
   `partner-sdk/`, or as a separate repo we open-source? Repo split makes
   external contributions easier but adds ops overhead. Recommend: monorepo
   under `partner-sdk/` for v1, split later.

## Risks

- **Contract lock-in** — once v1 ships and partners build against it, we
  can't change it. High blast radius if we get the shape wrong. Mitigation:
  freeze v1 in a spec doc and dogfood with the reference container for
  a week before external release.
- **Debug complexity across container boundary** — a partner bug looks like
  a CiRA ME bug to the end user. Mitigation: partner container name +
  version + license ID surfaced prominently in any error toast that came
  from a partner endpoint. "Error from ACME Trainer v2.3: OOM at epoch 12".
- **Partner container using stale contract** — mitigation: contract
  compatibility check at CiRA ME startup, refuse to register incompatible
  partners with a clear message.
- **Security** — partners could ship a malicious container. Mitigation: this
  is out of scope; we treat partner containers as trusted (they're paid
  vendors). Document it clearly in the SDK README.

## Handoff / sequencing

- Milestones 1-2 (contract spec + reference container) unblock everything
  else. Do these first, in parallel if possible.
- Milestone 3 (registry) and Milestone 6 (test suite) block Milestone 4-5
  (real usage). Do 3 + 6 next.
- Milestone 4-5 (real training + inference) are the value delivery.
- Milestone 7 (SDK package) is the customer-facing artifact — do last.

## Companion plans

- **`PLAN_2026-08-12_app-builder-widgets.md`** — parallel workstream
- **`PLAN_2026-08-12_sql-data-feed.md`** — parked, later
