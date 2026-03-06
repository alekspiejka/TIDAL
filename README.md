# TIDAL — Testing Interpersonal Distance And Loneliness

**Authors**: Aleksandra Piejka (<piejka@cbs.mpg.de>), Christian Abele (<christian.abele@uni-bielefeld.de>)

**License**: MIT · **Python**: ≥ 3.13 · **Status**: Pilot Study · Under Active Development 🚧

---

## Overview

TIDAL is a VR-based scientific paradigm investigating the relationship between **interpersonal distance preferences** and **loneliness**. Participants interact with virtual agents in a controlled environment while multi-modal physiological data is continuously recorded.

This repository contains the full data processing pipeline — from raw multi-modal recordings (LSL, Unity, Polar ECG, Eye-tracking, EMA) through BIDS-compliant preprocessing to time-synchronized feature extraction and segmentation for statistical analysis.

> [!IMPORTANT]
> **Data Privacy**: This repository is for **code and documentation only**. Participant data (raw or processed) is strictly confidential and excluded via `.gitignore`.

---

## External Tools & Repositories

This project depends on several external tools and frameworks for data acquisition, preprocessing, and real-time visualization. **Install and configure these before running the pipeline:**

| Tool | Purpose | Link |
|---|---|---|
| **EDIA Framework** | VR paradigm architecture and XR experiment infrastructure | [https://edia-toolbox.github.io/](https://edia-toolbox.github.io/) |
| **Polar2LSL** | Streams ECG data from Polar H10 to LSL | [https://markspan.github.io/Polar/](https://markspan.github.io/Polar/) |
| **LabRecorder** | Records LSL streams to `.xdf` files | [GitHub Releases](https://github.com/labstreaminglayer/App-LabRecorder/releases) |
| **BrainVision LSL Viewer** | Real-time visualization of LSL signal streams | [BrainProducts Downloads](https://www.brainproducts.com/downloads/more-software/) |
| **BBSIG** | ECG preprocessing, R-peak detection & correction | [https://martager.github.io/bbsig/](https://martager.github.io/bbsig/) |

> **Note on BBSIG**: When running BBSIG's ECG preprocessing pipeline, use `systole_editor_fixed.py` (included in this repo) to adjust for specific sampling frequency of the PolaeH10 device. See [Processing Pipeline](#processing-pipeline) for details.

---

## Experimental Design

Each participant completes **64 experimental trials** across **2 blocks** (32 trials each), preceded by 2 practice trials and separated by a break period. In each trial, a virtual agent approaches the participant, and the following data streams are recorded simultaneously:

| Modality | Device / Source | Sampling Rate | Format |
|---|---|---|---|
| ECG (R-peaks, HR, HRV) | Polar H10 via LSL | 130 Hz | `.xdf` → `.tsv.gz` |
| Eye-tracking (pupil diameter, openness) | Varjo Aero | 120 Hz | `.tsv.gz` |
| Head movement (position, rotation) | Varjo Aero | 90 Hz | `.tsv.gz` |
| VR events (trial markers, agent states) | Unity via LSL | Event-driven | `.tsv` / `.csv` |
| Questionnaires (loneliness, depression, etc.) | SoSci Survey | — | `.tsv` |
| Ecological Momentary Assessment (EMA) | m-Path | — | `.xlsx` → `.tsv` |

Five key phases are extracted per trial, anchored to specific events:

**Triphasic Data (Full Trial Progression)**

- **Baseline** — 3 seconds before the agent appears (`ShowAgent`)
- **Approach** — from `ShowAgent` to `AgentStopped`
- **Recovery** — 3 seconds after `AgentStopped`

**Biphasic Data (Stop-Related)**

- **Pre-stop** — 2 seconds before `AgentStopped`
- **Post-stop** — 2 seconds after `AgentStopped`

---

## Repository Structure

```
tidal/
├── code/                          # All processing scripts (see below)
├── pyproject.toml                 # Project config & dependencies
├── CHANGES                        # Version history
└── README.md
```

The dataset follows [BIDS v1.11.0](https://bids-specification.readthedocs.io/) conventions. Currently **18 participants** (sub-01P through sub-18P) have been recorded.

---

## Processing Pipeline

The scripts in `code/` form a sequential pipeline. Each stage produces BIDS-formatted derivative outputs.

### Data Conversion & BIDS Formatting

| Script | Purpose |
|---|---|
| `lsl_preproc.py` | Converts raw `.xdf` files into BIDS-compliant `.tsv.gz` + JSON sidecars. Extracts ECG, eye-tracking, head movement, and event marker streams. |
| `vr_preproc.py` | Processes Unity behavioral logs (`trial_results.csv`) into BIDS files. Computes derived metrics (e.g., Approach Ratio) and anonymizes IDs. |
| `questionnaire_preproc.py` | Scores standardized questionnaires (R-UCLA, CESD-R, DACOBS, BIS/BAS, MAIA, SSQ) with reverse-coding and subscale computation. |
| `ema_preproc.py` | Preprocesses Ecological Momentary Assessment (EMA) data from m-Path exports (`.xlsx`). Computes daily mood and social interaction summaries. |
| `ema_merge.py` | Merges individual participant EMA TSV files into a unified dataset. |

### Segmentation & Feature Extraction

| Script | Purpose |
|---|---|
| `segmenting.py` | **Master pipeline**: synchronizes all data streams to a common LSL timeline, extracts trial/block/phase windows, computes CWT features, and exports gzipped JSON with all segmented physiological data. |
| `time_alignment_check.py` | Pre- and post-alignment quality verification. Validates LSL-vs-Unity input consistency (pre) and structural integrity of aligned output — NaN checks, per-segment monotonicity, trial counts, durations, block ordering, delta preservation (post). Saves BIDS-compliant report files. |
| `hrv_extraction.py` | Computes time-domain HRV features (RMSSD, SDNN, pNN50) from corrected R-peak intervals. |
| `cwt.py` | Continuous Wavelet Transform for frequency-domain HRV — extracts LF and HF power bands from RR intervals. |
| `graham_weighted_hr.py` | Computes second-by-second event-related heart rate changes using Graham's (1978) weighted IBI method. |
| `vr_summary_by_speed.py` | Aggregates VR behavioral metrics per participant, split by fast vs. slow approach speeds. 

### Utilities

| Script | Purpose |
|---|---|
| `generate_trials_tidal.py` | Generates randomized, balanced trial lists for Unity with anti-repetition constraints. Supports incremental participant folder creation. |
| `inspect_xdf.py` | Diagnostic tool for raw XDF files — verifies stream presence, sampling rates, and data integrity after recording. |
| `merge_interrupted_session.py` | Merges separate recording files from interrupted sessions (e.g., sub-12P) into a single continuous dataset. |
| `systole_editor_fixed.py` | Patch utility for ECG R-peak correction in systole when using the BBSIG ECG preprocessing pipeline. adjustment of sampling frequency and decim parameters. |

---

## Time Alignment

Synchronizing multi-modal data streams is critical. The pipeline uses **`AgentStopped` events** as temporal anchors — these events exist in both the Unity execution log and the LSL event stream, providing a reliable mapping between Unity's internal clock and the LSL timeline.

`segmenting.py` performs the alignment via `execution_order_time_alignment()`, which:

1. Matches `AgentStopped` events between LSL and Unity
2. Re-anchors timestamps at every match point
3. Extrapolates timestamps for surrounding events within each segment

`time_alignment_check.py` validates this process with two independent checks:

- **Pre-alignment**: Compares raw LSL event deltas against Unity deltas (input consistency)
- **Post-alignment**: Validates the aligned output — no NaN timestamps, per-segment monotonicity, 64 trials present, positive/reasonable durations, correct block ordering, and delta preservation (≤20ms tolerance)

Results are printed as a compact summary per participant and saved as `alignment-report.txt` files alongside the segmented data.

---

## Key Analysis Methods

Given the dynamic nature of our VR paradigm and the short duration of specific trial phases, we rely on specialized analytical tools that overcome the limitations of classic, long-term HRV measures:

- **Continuous Wavelet Transform (CWT):** Classic frequency-domain HRV methods (like Fourier transforms) require continuous, stable recordings over longer periods (often 2–5 minutes) and struggle to capture rapid physiological shifts. We instead extract frequency-domain HRV features (LF and HF power bands) using CWT. CWT provides excellent time-frequency resolution, making it much more reliable for evaluating non-stationary autonomic nervous system fluctuations during our short, dynamic trial segments.
- **Graham's Weighted Heart Rate:** When analyzing event-related cardiac responses (e.g., the exact moment the agent stops), simple HR averages or raw inter-beat intervals (IBIs) are often too sparse and loosely aligned with physical time. We utilize Graham's (1978) weighted IBI method to proportion each heartbeat's duration precisely into fixed time bins (e.g., 0.5s). This produces a continuous, high-resolution time course of heart rate deceleration or acceleration that is strictly time-locked to the VR events, improving both interpretability and statistical reliability compared to standard beat-to-beat analyses.

---

## Getting Started

### 1. Environment Setup

```bash
# Clone and install (editable mode with dev tools)
cd tidal/
pip install -e ".[develop]"
```

**Core dependencies** (auto-installed): `pandas`, `numpy`, `scipy`, `neurokit2`, `pyxdf`, `pylsl`, `PyWavelets`, `systole`, `bleak`

### 2. Running the Pipeline

```bash
# Step 1: Convert raw data to BIDS
python code/lsl_preproc.py
python code/vr_preproc.py
python code/questionnaire_preproc.py

# Step 2: ECG preprocessing (recommended)
# Apply BBSIG ECG preprocessing pipeline from https://martager.github.io/bbsig/
# For R-peak correction assistance, use systole_editor_fixed.py

# Step 3: Feature extraction & segmentation
python code/segmenting.py

# Optional: Standalone alignment verification
python code/time_alignment_check.py --all
```

### 3. Output

Segmented data is saved as gzipped JSON files at:

```
data/derivatives/segmenting/sub-XXP/ses-001/sub-XXP_ses-001_task-TIDAL_run-001_segmented.json.gz
```

Each file contains nested dictionaries with Baseline, Approach, Recovery, pre-stop, and post-stop phase data for all 64 trials, including synchronized ECG peaks, HR, pupil diameter, head movement, and HRV frequency features.

---

## Current Status

- **18 participants** recorded (sub-01P through sub-18P)
- **Segmentation completed** for all available subjects (except sub-16P — interrupted session)
- **BIDS v1.11.0** compliant dataset structure
- **Statistical Analysis and Visualization** Work in progress!
- Active development!

---

*For questions, please contact the authors.*
