# Solution App — Example Settings Reference

Quick reference for what to enter in the **Motor Current (MCSA)** and **Machine
Vibration** Solution Apps. The example columns are known-good starting points
(the built-in demo datasets); the "Where to get it" column tells you how to find
the value for your own machine.

> Tip: the app has an **Advanced** section for less-common settings; the tables
> below cover everything, with the advanced ones marked *(advanced)*.

---

## Motor Current Diagnosis (MCSA)

Diagnoses induction-motor faults (broken rotor bars, eccentricity) from 3-phase
stator current.

| Field | Example — synthetic demo (`motor_labeled.csv`) | Example — UNESP motor | Where to get it |
|---|---|---|---|
| **Sample rate (Hz)** | `5000` | `50000` | Your DAQ / acquisition rate |
| **Window (s)** | `10` | `10` | Keep **10 s** — MCSA needs long windows to resolve sidebands |
| **Pole pairs** | `2` | `2` | Motor nameplate: pole **pairs** = poles ÷ 2 (a 4-pole motor = 2) |
| **Line frequency (Hz)** | `50` | `60` | Your mains supply (50 or 60 Hz) |
| **Rotor bars** *(optional)* | `28` | `34` | Motor datasheet; improves eccentricity models. Leave blank if unknown |
| **Rated rpm** *(optional)* | *(blank)* | `1715` | Nameplate full-load speed; bounds the slip search |
| **Phase to analyze** *(advanced)* | `0` | `0` | Which current column (Ia=0). Any phase works |
| **Grid-lock / calibrate** *(advanced)* | on | on | Leave **on** unless you have a verified hardware clock |
| **Absent floor (dBc)** *(advanced)* | `-120` | `-120` | Use `-90` if you train an **Anomaly** model (kinder to IsolationForest) |
| **Current phase columns** | `Ia, Ib, Ic` | `Ia, Ib, Ic` | The 3 current channels in your CSV |

**Reading the result:** a *less-negative* broken-bar sideband (e.g. −33 vs −50
dBc) means a stronger fault. In the spectrum, watch the dashed markers at
**f₀ ± sidebands** rise from the noise floor as severity increases.

---

## Machine Vibration Diagnosis (bearing)

Detects rolling-element bearing faults from accelerometer vibration via envelope
analysis (BPFO / BPFI / BSF / FTF).

| Field | Example — real CWRU demo (`bearing_labeled.csv`) | Where to get it |
|---|---|---|
| **Sample rate (Hz)** | `12000` | Your DAQ rate (bearing resonances need ≥ ~5 kHz) |
| **Window (s)** | `1` | 1 s is plenty at kHz rates |
| **Shaft speed (Hz)** | `29.9` | Motor rpm ÷ 60 (e.g. 1797 rpm → 29.95 Hz) |
| **Rolling elements** | `9` | Bearing datasheet (number of balls/rollers) |
| **Ball diameter (mm)** *(optional)* | `7.94` | Bearing datasheet. With pitch dia → exact fault freqs; else approximations |
| **Pitch diameter (mm)** *(optional)* | `39.04` | Bearing datasheet |
| **Contact angle (deg)** | `0` | Bearing datasheet (0 for deep-groove ball bearings) |
| **Resonance band (Hz)** | `1000` – `5000` | The high-frequency band where the bearing resonates. 1–5 kHz is a good default |
| **Vibration channel** | `accel` | The accelerometer column in your CSV |
| **Axis** *(advanced)* | `0` | For multi-axis sensors, which axis to analyze |

The demo bearing is a **SKF 6205** (CWRU drive-end) — those geometry values
(9 balls, 7.94 mm ball, 39.04 mm pitch, 0° angle) come straight from its
datasheet, which is how you'd fill in any bearing.

**Reading the result:** an **outer-race** fault raises **BPFO**; an **inner-race**
fault raises **BPFI**; a **ball** fault raises **BSF**. In the envelope spectrum,
a peak at the dashed fault-frequency marker is the fault signature.

---

## Choosing the approach (both apps)

| Your data | Approach | Model |
|---|---|---|
| Only healthy recordings | **Anomaly** (learns a healthy baseline, flags drift) | Isolation Forest |
| Labeled healthy + fault (a `label` column) | **Classification** | Random Forest |
| Not sure | **Auto** — picks Classification if ≥ 2 labels, else Anomaly | — |

## Where do these values come from?

- **Motor nameplate / datasheet** → pole pairs, line frequency, rated rpm, rotor bars.
- **Bearing datasheet** (SKF/NSK/etc., by part number) → ball count, ball & pitch
  diameter, contact angle. Search the bearing number + "geometry" or "bearing
  calculator".
- **Shaft speed** → measured rpm (or motor rpm through the drive) ÷ 60.
- **Sample rate** → set on your data-acquisition device when recording.

## Built-in demo files

| App | File (under `datasets/…_demo/`) | Real / synthetic |
|---|---|---|
| MCSA | `mcsa_demo/motor_labeled.csv`, `motor_healthy.csv` | synthetic (statorscope) |
| Vibration | `vibration_demo/bearing_labeled.csv`, `bearing_healthy.csv` | **real (CWRU)** |
