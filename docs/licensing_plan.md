---
name: Licensing plan for commercial deployment
description: Machine-bound licensing via /etc/machine-id + signed license.json — implement before shipping to customers
type: project
---

## Licensing System — Implementation Plan (DEFERRED)

**Status:** Not implemented. Implement before commercial release.

### Architecture: Machine-ID Binding (Offline, Option B)

```
Customer sends: /etc/machine-id → CiRA generates signed license.json → customer drops in deployment/
```

### Components to Build

#### 1. Key Pair Generation (one-time, CiRA internal)
```bash
python tools/generate_keys.py
# Outputs: private_key.pem (SECRET — keep at CiRA)
#          public_key.pem  (embed in Docker image)
```

#### 2. License Generator Script (CiRA internal, never shipped)
```bash
python tools/generate_license.py \
  --machine-id "a1b2c3d4e5f6789012345678" \
  --company "ACME Factory" \
  --max-users 10 \
  --expiry 2027-03-28 \
  --features "ml,dl,ti,mqtt,app_builder"
# Outputs: license.json
```

#### 3. Backend License Validator (~100 lines in Flask)
- On startup: read `/host-machine-id` (volume-mounted from host `/etc/machine-id`)
- Hash: `sha256(machine_id + salt)`
- Read `license.json` from mounted volume
- Verify Ed25519 signature using embedded public key
- Check: hash match, expiry date, feature flags
- Enforce: `max_users` on user creation, feature gates on routes

#### 4. Docker Compose Changes
```yaml
backend:
  volumes:
    - /etc/machine-id:/host-machine-id:ro
    - ./license.json:/app/license.json:ro
```

### License File Format
```json
{
  "license": {
    "machine_hash": "sha256hex...",
    "company": "ACME Factory",
    "max_users": 10,
    "expiry": "2027-03-28",
    "features": ["ml", "dl", "ti", "mqtt", "app_builder"],
    "issued_at": "2026-03-29"
  },
  "signature": "ed25519hex..."
}
```

### Feature Tiers (suggested)
| Tier | Users | Features | Price |
|---|---|---|---|
| Basic | 3 | ML training + deploy | - |
| Pro | 10 | + DL, TI, App Builder | - |
| Enterprise | Unlimited | + MQTT, custom models, API endpoints | - |

### Dependencies
- `cryptography` Python package (already common, add to requirements.txt)
- No external server, database, or internet needed

### Files to Create
```
tools/
  generate_keys.py          # One-time key pair generation
  generate_license.py       # License generator (CiRA internal)
backend/app/
  license.py                # Validator module
  routes/ (modify)          # Feature gates on routes
deployment/
  license.json              # Customer's license file (placeholder)
```

### Windows Host Note
- Windows doesn't have `/etc/machine-id`
- Use `wmic csproduct get UUID` or `reg query` for machine UUID
- Docker Desktop on Windows: mount host UUID via environment variable
