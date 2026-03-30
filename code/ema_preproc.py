"""
EMA Data Preprocessing & BIDS Conversion
========================================

This script is responsible for preprocessing Ecological Momentary Assessment (EMA) data,
standardizing it into BIDS format, and generating both participant-level summaries and
merged long-form datasets.

Key Functionalities:
1.  Loads raw EMA data from Excel files for configured participants.
2.  Cleans column names and recodes scale values.
3.  Calculates derived metrics, including composite loneliness and social/solitude appraisals.
4.  Standardizes contexts into a fixed set of labels (alone, known, other, both).
5.  Outputs individual-level BIDS-compliant '.tsv' and '.json' sidecar files.
6.  Updates the BIDS 'scans.tsv' file for each session.
7.  Aggregates all participant data to compute participant-level summaries (means, percentages)
    and saves this as 'ema_summary.tsv'.
8.  Merges all valid EMA data into a single long-form dataset ('merged_ema_data.tsv')
    for downstream time-series analysis or mixed-effects modeling.

Usage:
Edit the configuration section below (participants, sessions, and paths), then run:
    python code/ema_preproc.py
"""
# %%
# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

# %%
# Initialize logger for tracking script execution details
LOG = logging.getLogger(__name__)

# =================================================================================================
# Global Configuration & Path Setup
# =================================================================================================
# Set up the directory structure relative to the project root.
root = Path(__file__).parent.parent  # Move up two levels to reach the project root
data_dir = root / "data"             # Main data directory containing all subfolders
sourcedata_dir = data_dir / "sourcedata"
derivatives_dir = data_dir / "derivatives"

# ---------------------------------------------------------
# Subject & Session Definition
# ---------------------------------------------------------
# Define the list of subjects to process.
# pfxs contains the subject identifiers (e.g., "01P").
pfxs = ["01P", "02P", "03P", "04P", "05P", "06P", "07P", "08P", "09P", "10P", "11P", "12P", "13P", "14P", "15P", "16P", "17P","18P","19P","20P"]
subjects = [f"sub-{pfx}" for pfx in pfxs]

# Define the sessions to process.
sessions = ["ses-002"]

# ----------------------------------------------------------------------------
# CONSTANTS: Context Mapping
# ----------------------------------------------------------------------------
# Maps raw context text responses to standardized context labels.
# This fixed mapping ensures consistent context codes across all participants,
# even if some participants don't have all context types.
CONTEXT_MAPPING = {
    'no, i\'m alone': 'alone',
    'no i\'m alone': 'alone',
    'no im alone': 'alone',
    'no': 'alone',
    'yes, people i know': 'known',
    'yes people i know': 'known',
    'yes people i know,': 'known',
    'yes, people i don\'t know': 'other',
    'yes people i don\'t know': 'other',
    'yes people i dont know': 'other',
    'yes, some people i know and people i don\'t know': 'both',
    'yes some people i know and people i don\'t know': 'both',
    'yes some people i know and people i dont know': 'both',
    'both': 'both',
}
# Context names in a consistent order
CONTEXT_NAMES = ['alone', 'other', 'known', 'both']
CONTEXT_NAMES_SOC = ['other', 'known', 'both']  # Subset for social contexts only

# %%
# ============================================================================
# HELPER FUNCTIONS: Scale Recoding
# ============================================================================

def recode_scales(df: pd.DataFrame) -> None:
    """Create new columns with recoded EMA scales from 0-100 to 1-5 bins for easier interpretation

    Modifies the dataframe in-place by adding new columns with suffix '_bin' for each numeric column.
    Values are binned into 5 equal-width bins corresponding to the original 0-100 scale.
    Bins are defined as 0-20 = 1, 20-40 = 2, 40-60 = 3, 60-80 = 4, 80-100 = 5. Non-numeric columns are not modified.

    Args:
        df: DataFrame with EMA item columns (numeric).
    """
    bins = [0, 20, 40, 60, 80, 100]
    labels = [1, 2, 3, 4, 5]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)


# ============================================================================
# CORE PROCESSING: Single EMA File Handler
# ============================================================================

def process_ema_file(path: Path, data_dir: Path) -> Optional[str]:

    """Read, clean, enrich and save a single EMA Excel file.

    Processing assumptions and behavior:
    - The input `path` is an Excel file ('.xlsx' or '.xls') created by the EMA task. The reader uses
      pandas.read_excel(..., header=1) because the files have a descriptive header row at row 1.
    - Column headers are cleaned by removing parenthetical notes (e.g. "item (0-100)").
    - The function attempts to infer `ppid` and `ses-id` from parent directories that begin with
      `sub-` and `ses-` respectively; if not found, the values `sub-unknown` / `ses-unknown` are used.
    - If a "Date and time" column exists it is parsed to a `datetime` column and split into `date` and `time`.
    Outputs (side effects): writes a TSV file to `data/<ppid>/<ses-id>/<stem>.tsv` and a
      JSON sidecar `<stem>.json` describing columns. Files are overwritten if they already exist.

    Returns
    - participant id string (e.g. 'sub-01P') on success, or None if the file could not be read.
"""
    try:
        ema = pd.read_excel(path, engine="openpyxl", header=1)
    except Exception as exc:  # pragma: no cover - file IO
        print(f"ERROR: Could not read {path}: {exc}")
        LOG.debug("Skipping unreadable file %s: %s", path, exc)
        return None

    # Filter columns early to keep only expected EMA items and metadata
    base_cols = {
        'Date and time', 'context',
    }
    allowed_prefixes = (
        'pa_', 'na_', 'lon_', 'soc_', 'comfort',
    )
    allowed_extra = {
        'neg_affect', 'pos_affect',
    }
    keep_cols = []
    for col in ema.columns:
        cleaned_col = re.sub(r"\s*\([^)]*\)", "", col).strip() if isinstance(col, str) else col
        if (
            cleaned_col in base_cols
            or cleaned_col in allowed_extra
            or (isinstance(cleaned_col, str) and cleaned_col.startswith(allowed_prefixes))
        ):
            keep_cols.append(col)
    if keep_cols:
        ema = ema.loc[:, keep_cols]

    # ------------------------------------------------------------------------
    # 1. Column Cleaning & Metadata Extraction
    # ------------------------------------------------------------------------
    # Clean column headers: Remove parenthetical info like "(0-100)" or "(item 1)" to get clean variable names
    ema.columns = [re.sub(r"\s*\([^)]*\)", "", col).strip() if isinstance(col, str) else col for col in ema.columns]

    # Recode 'context' column to standardized context labels (alone, other, known, both)
    # This uses a fixed mapping to ensure consistent context codes across all participants
    if "context" in ema.columns:
        ema["context_factor"] = ema["context"].str.strip().str.lower().map(CONTEXT_MAPPING)

    # Derive BIDS ids from path structure (expects .../sub-XXX/ses-YYY/...)
    # This makes the script location-agnostic as long as folder structure is BIDS-like
    sub_id = next((p.name for p in path.parents if p.name.startswith("sub-")), "sub-unknown")
    ses_id = next((p.name for p in path.parents if p.name.startswith("ses-")), "ses-unknown")

    # Extract run number if present in filename (e.g., _run-01), default to run-001
    run_match = re.search(r"run[-_](\d+)", path.name)
    run_id = f"run-{run_match.group(1)}" if run_match else "run-001"

    ema["ppid"] = sub_id
    ema["ses-id"] = ses_id
    ema["run-id"] = run_id
    ema["task"] = "EMA"

    # Rename and parse datetime column
    if "Date and time" in ema.columns:
        ema = ema.rename(columns={"Date and time": "datetime"})
        ema["datetime"] = pd.to_datetime(ema["datetime"], errors="coerce")
        ema["date"] = ema["datetime"].dt.date
        ema["time"] = ema["datetime"].dt.time
    else:
        ema["datetime"] = pd.NaT

    cols = ema.columns.tolist()
    new_order = ["ppid", "ses-id", "run-id", "task", "datetime", "date", "time"] + [c for c in cols if c not in ["ppid", "ses-id", "run-id", "task", "datetime", "date", "time"]]
    ema = ema.loc[:, new_order]

    # ------------------------------------------------------------------------
    # 2. Score Calculation: Averaging PA, NA, and Social Metrics
    # ------------------------------------------------------------------------
    # Identify columns for Positive Affect (PA) and Negative Affect (NA) based on prefixes
    pa_cols = [col for col in ema.columns if col.startswith("pa_")]
    na_cols = [col for col in ema.columns if col.startswith("na_")]

    # Split into social vs non-social sub-scales based on keywords
    pa_soc_cols = [col for col in pa_cols if any(x in col for x in ("belonging", "supported"))]
    na_soc_cols = [col for col in na_cols if any(x in col for x in ("lonely", "isolated"))]
    pa_nonsoc_cols = [col for col in pa_cols if col not in pa_soc_cols]
    na_nonsoc_cols = [col for col in na_cols if col not in na_soc_cols]

    # Calculate mean scores (ignoring NaNs)
    ema["pa_avg"] = ema[pa_cols].mean(axis=1, skipna=True)
    ema["na_avg"] = ema[na_cols].mean(axis=1, skipna=True)
    ema["pa_soc_avg"] = ema[pa_soc_cols].mean(axis=1, skipna=True) if pa_soc_cols else pd.NA
    ema["na_soc_avg"] = ema[na_soc_cols].mean(axis=1, skipna=True) if na_soc_cols else pd.NA
    ema["pa_nonsoc_avg"] = ema[pa_nonsoc_cols].mean(axis=1, skipna=True) if pa_nonsoc_cols else pd.NA
    ema["na_nonsoc_avg"] = ema[na_nonsoc_cols].mean(axis=1, skipna=True) if na_nonsoc_cols else pd.NA

    # Create composite loneliness score (higher is more lonely)
    # Logic: Average of NA_social items (lonely/isolated) and REVERSED PA_social items (belonging/supported).
    # Since PA items are 0-100 (where 100=supported), we use (100-x) so that higher values = less supported (more lonely).
    if pa_soc_cols:
        pa_soc_reversed = ema[pa_soc_cols].apply(lambda x: 100 - x)
    else:
        pa_soc_reversed = pd.DataFrame()

    if na_soc_cols or pa_soc_cols:
        concat_cols = [ema[na_soc_cols]] if na_soc_cols else []
        if not pa_soc_reversed.empty:
            concat_cols.append(pa_soc_reversed)
        ema["loneliness_avg"] = pd.concat(concat_cols, axis=1).mean(axis=1, skipna=True)
    else:
        ema["loneliness_avg"] = pd.NA

    # Create average scores for social appraisals and solitude appraisals (higher is more positive)
    soc_pos_app_cols = [col for col in ema.columns if col.startswith("soc_") and not any(x in col for x in ("distance", "aloof", "unapproachable"))]
    soc_neg_app_cols = [col for col in ema.columns if col.startswith("soc_") and not any(x in col for x in ("distance", "friendly", "trustworthy"))]
    lon_pos_app_cols = [col for col in ema.columns if col.startswith("lon_") and not any(x in col for x in ("distance","stressful","draining"))]
    lon_neg_app_cols = [col for col in ema.columns if col.startswith("lon_") and not any(x in col for x in ("distance", "pleasant","comfortable"))]

    # Use reversed scale (100-x) for negative appraisal columns
    if soc_neg_app_cols:
        soc_neg_reversed = ema[soc_neg_app_cols].apply(lambda x: 100 - x)
    else:
        soc_neg_reversed = pd.DataFrame()
    if soc_pos_app_cols or soc_neg_app_cols:
        concat_cols = [ema[soc_pos_app_cols]] if soc_pos_app_cols else []
        if not soc_neg_reversed.empty:
            concat_cols.append(soc_neg_reversed)
        soc_app = pd.concat(concat_cols, axis=1).mean(axis=1, skipna=True)
        ema["soc_app"] = soc_app
        ema["soc_app_avg"] = soc_app
    else:
        ema["soc_app"] = pd.NA
        ema["soc_app_avg"] = pd.NA

    if lon_neg_app_cols:
        lon_neg_reversed = ema[lon_neg_app_cols].apply(lambda x: 100 - x)
    else:
        lon_neg_reversed = pd.DataFrame()
    if lon_pos_app_cols or lon_neg_app_cols:
        concat_cols = [ema[lon_pos_app_cols]] if lon_pos_app_cols else []
        if not lon_neg_reversed.empty:
            concat_cols.append(lon_neg_reversed)
        ema["lon_app_avg"] = pd.concat(concat_cols, axis=1).mean(axis=1, skipna=True)
    else:
        ema["lon_app_avg"] = pd.NA

    # Add columns with values recoded to 1-5 bins for easier interpretation (e.g., pa_avg_bin, na_avg_bin, etc.)
    recode_scales(ema)

    # ------------------------------------------------------------------------
    # 3. Output Generation: TSV and JSON Sidecar
    # ------------------------------------------------------------------------
    # Create output directory structure: data/sub-XXX/ses-XXX/beh/
    out_dir = data_dir / sub_id / ses_id / "beh"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.tsv"

    # Save as tab-separated values, using 'n/a' for missing data (BIDS standard)
    ema.to_csv(out_path, sep="\t", index=False, na_rep="n/a")

    # Create JSON sidecar describing every column in the TSV.
    # Keys MUST match the exact TSV column names for BIDS compliance.

    # Pre-defined descriptions for metadata columns
    COLUMN_DEFS = {
        "ppid":        {"Description": "Participant ID (BIDS format, e.g. sub-04P)"},
        "ses-id":      {"Description": "Session identifier (e.g. ses-002)"},
        "run-id":      {"Description": "Run identifier (e.g. run-001)"},
        "task":        {"Description": "Task label (EMA)"},
        "datetime":    {"Description": "Date and time of EMA prompt", "Units": "ISO 8601"},
        "date":        {"Description": "Date of EMA prompt", "Units": "YYYY-MM-DD"},
        "time":        {"Description": "Time of EMA prompt", "Units": "HH:MM:SS"},
    }

    meta = {}
    for col in ema.columns:
        # Use pre-defined description if available
        if col in COLUMN_DEFS:
            meta[col] = COLUMN_DEFS[col].copy()
            continue

        # Build description using the actual column name as key
        desc = f"EMA item: {col}"
        entry = {"Description": desc}
        col_lower = col.lower()

        # Add specific units/levels for known scale types
        if col_lower.startswith(("pa_", "na_", "lon_", "soc_")) or col_lower == "comfort":
            entry["Description"] = desc + " (0-100 scale; higher is more)"
            entry["Units"] = "a.u."
            if col_lower.startswith("pa_"):
                entry["Levels"] = {"0": "Not at all", "50": "neutral", "100": "Very much"}
        if col_lower == "context":
            entry["Description"] = "Social context of EMA response (multiple choice)"
        if col_lower == "context_factor":
            entry["Description"] = "Standardized social context label (alone, other, known, both)"
        meta[col] = entry

    out_json = out_path.with_suffix(".json")
    with open(out_json, "w") as fh:
        json.dump(meta, fh, indent=2)

    LOG.info("Wrote %s and %s", out_path, out_json)

    # ------------------------------------------------------------------------
    # 4. Update scans.tsv: Register the data file with its acquisition time
    # ------------------------------------------------------------------------
    # Use the first valid datetime from the EMA data as acquisition time
    session_dir = out_dir.parent  # data/sub-XXX/ses-XXX/
    acq_time = None
    if "datetime" in ema.columns:
        first_valid = ema["datetime"].dropna().min()
        if pd.notna(first_valid):
            acq_time = pd.Timestamp(first_valid).strftime("%Y-%m-%dT%H:%M:%S")

    # Build scans.tsv path: sub-XXX_ses-YYY_scans.tsv
    scans_filename = f"{sub_id}_{ses_id}_scans.tsv"
    scans_path = session_dir / scans_filename

    # Relative path from session dir to the data file (BIDS uses forward slashes)
    rel_path = out_path.relative_to(session_dir).as_posix()

    # Create or update scans.tsv (only data files, never JSON sidecars)
    new_entry = {"filename": rel_path}
    if acq_time:
        new_entry["acq_time"] = acq_time

    if scans_path.exists():
        try:
            scans_df = pd.read_csv(scans_path, sep="\t")
            if rel_path not in scans_df["filename"].values:
                scans_df = pd.concat([scans_df, pd.DataFrame([new_entry])], ignore_index=True)
                scans_df.to_csv(scans_path, sep="\t", index=False, na_rep="n/a")
                LOG.info("Updated %s: added %s", scans_path.name, rel_path)
        except Exception as exc:
            LOG.warning("Could not update %s: %s", scans_path, exc)
    else:
        pd.DataFrame([new_entry]).to_csv(scans_path, sep="\t", index=False, na_rep="n/a")
        LOG.info("Created %s", scans_path.name)

    return sub_id

# %%

def find_ema_files(root: Path, participants: Optional[Iterable[str]] = None, session: Optional[str] = None) -> List[Path]:
    """Return a list of EMA files under `root/data` that match participants and session.

    Search details:
    - The function searches recursively under `root/data` for files matching the glob patterns
      `*EMA*.xlsx` and `*EMA*.xls`.
    - `participants` can be `None` (match all) or an iterable of suffixes like `04P` or full ids like
      `sub-04P`. Short suffixes are normalized to `sub-<suffix>` for matching.
    - `session` expects an exact session folder name (e.g. `ses-002`), or `None` to match any session.
      The caller may pass "all" to indicate no session filtering (this function interprets `None` as no filter).

    Returns
    - A list of `Path` objects pointing to matched EMA files.
    """
    candidates = list((root / "data").rglob("*EMA*.xlsx")) + list((root / "data").rglob("*EMA*.xls"))

    def matches(p: Path) -> bool:
        # 1. Filter by Participant ID (found in parent folders)
        pid = next((pp.name for pp in p.parents if pp.name.startswith("sub-")), None)
        if not pid:
            return False
        if participants is not None:
            # normalize participants list to form 'sub-XX' standard
            allowed = {f if f.startswith("sub-") else f"sub-{f}" for f in participants}
            if pid not in allowed:
                return False

        # 2. Filter by Session ID (found in parent folders)
        if session is not None:
            if not any(pp.name == session for pp in p.parents):
                return False
        return True

    return [p for p in candidates if matches(p)]

# ============================================================================
# HELPER FUNCTIONS: Context Factor Creation
# ============================================================================

def _create_context_factor(df: pd.DataFrame) -> None:
    """Create context_factor column from context column if not present.

    Maps context values to standardized labels (alone, other, known, both).
    Uses a fixed mapping to ensure consistent context codes across all participants.
    Modifies the dataframe in-place.

    Args:
        df: DataFrame with 'context' column. If 'context_factor' already exists, no change.
    """
    if 'context_factor' not in df.columns and 'context' in df.columns:
        df['context_factor'] = df['context'].str.strip().str.lower().map(CONTEXT_MAPPING)


def _create_summary_sidecar(tsv_path: Path) -> None:
    """Create a JSON sidecar file describing the EMA summary TSV columns.

    Args:
        tsv_path: Path to the ema_summary.tsv file.
    """
    df = pd.read_csv(tsv_path, sep='\t', nrows=1)  # Read header only

    meta = {
        "Description": "Per-participant aggregated EMA (Ecological Momentary Assessment) summary statistics",
        "columns": {}
    }

    # Column descriptions and metadata
    for col in df.columns:
        entry = {}

        if col == 'ppid':
            entry["Description"] = "Participant ID (BIDS format: sub-XX)"

        # Overall averages
        elif col == 'pa_avg':
            entry["Description"] = "Overall positive affect average"
            entry["Units"] = "0-100"
        elif col == 'na_avg':
            entry["Description"] = "Overall negative affect average"
            entry["Units"] = "0-100"
        elif col == 'pa_soc_avg':
            entry["Description"] = "Positive affect in social contexts (belonging, supported)"
            entry["Units"] = "0-100"
        elif col == 'na_soc_avg':
            entry["Description"] = "Negative affect in social contexts (lonely, isolated)"
            entry["Units"] = "0-100"
        elif col == 'pa_nonsoc_avg':
            entry["Description"] = "Positive affect in non-social contexts"
            entry["Units"] = "0-100"
        elif col == 'na_nonsoc_avg':
            entry["Description"] = "Negative affect in non-social contexts"
            entry["Units"] = "0-100"
        elif col == 'loneliness_avg':
            entry["Description"] = "Composite loneliness score (higher = more lonely)"
            entry["Units"] = "0-100"
        elif col == 'soc_app_avg':
            entry["Description"] = "Average social appraisal score"
            entry["Units"] = "0-100"
        elif col == 'lon_app_avg':
            entry["Description"] = "Average solitude appraisal score"
            entry["Units"] = "0-100"

        #Overall averages for bins (1-5)
        elif col.endswith('_bin'):
            base_col = col.replace('_bin', '')
            entry["Description"] = f"Binned version of {base_col} (1-5 scale)"
            entry["Units"] = "1-5"

        # Context percentages
        elif col.startswith('context_') and col.endswith('_per'):
            ctx = col.replace('context_', '').replace('_per', '')
            entry["Description"] = f"Percentage of EMA responses in {ctx} context"
            entry["Units"] = "percent"
        elif col == 'social_contexts_per':
            entry["Description"] = "Percentage of EMA responses in any social context (other, known, both)"
            entry["Units"] = "percent"

        # Comfort averages per context
        elif col.startswith('comfort_'):
            ctx = col.replace('comfort_', '')
            if ctx == '':
                entry["Description"] = "Average comfort in social contexts combined (other, known, both)"
            else:
                entry["Description"] = f"Average comfort in {ctx} context"
            entry["Units"] = "0-100"

        # Social distance per context
        elif col.startswith('soc_distance_'):
            ctx = col.replace('soc_distance_', '')
            entry["Description"] = f"Average social distance perception in {ctx} context"
            entry["Units"] = "0-100"

        # Social approval per context
        elif col.startswith('soc_app_'):
            ctx = col.replace('soc_app_', '')
            entry["Description"] = f"Average social approval perception in {ctx} context"
            entry["Units"] = "0-100"

        # Individual EMA items (generic)
        else:
            entry["Description"] = f"EMA item: {col}"
            # Infer scale for numeric items with common prefixes
            if any(col.startswith(p) for p in ['pa_', 'na_', 'lon_', 'soc_', 'comfort']):
                entry["Units"] = "0-100"

        meta["columns"][col] = entry

    # Write JSON sidecar
    json_path = tsv_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    LOG.info("Wrote summary sidecar %s", json_path)

# %%
# ============================================================================
# SUMMARY GENERATION: Aggregate Individual Data into Participant Summaries
# ============================================================================

def generate_ema_summary(files: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate per-participant summary statistics and a merged long-form dataset from preprocessed EMA files.

    Combines individual-level EMA responses into participant-level summaries with:
    - Overall averages (PA, NA, loneliness, social appraisals, etc.)
    - Context-specific statistics (percentages, comfort, social distance/approval per context)
    - Individual EMA item responses

    Args:
        files: List of TSV/Excel file paths containing preprocessed EMA data.

    Returns:
        A tuple of two DataFrames:
        1. summary_df: One row per participant (ppid) and columns for all summary statistics.
        2. merged_df: The fully merged long-form dataset of all valid EMA records.
        Returns empty DataFrames if no valid files found.
    """
    all_ema = []
    for f in files:
        if str(f).endswith('.xlsx') or str(f).endswith('.xls'):
            df = pd.read_excel(f, engine='openpyxl', header=1)
        elif str(f).endswith('.tsv'):
            df = pd.read_csv(f, sep='\t')
        else:
            continue

        # Add participant/session info if not present
        if 'ppid' not in df.columns:
            sub_id = next((p.name for p in Path(f).parents if p.name.startswith('sub-')), 'sub-unknown')
            ses_id = next((p.name for p in Path(f).parents if p.name.startswith('ses-')), 'ses-unknown')
            df['ppid'] = sub_id
            df['ses-id'] = ses_id

        # Create context_factor from context column if needed
        _create_context_factor(df)
        all_ema.append(df)

    if not all_ema:
        return pd.DataFrame(), pd.DataFrame()

    ema = pd.concat(all_ema, ignore_index=True)

    # Ensure context_factor is created and is string type
    _create_context_factor(ema)  # In case any file was missing context_factor after concat

    # Per-participant averages for all columns that are numeric and not id columns
    # We group by 'ppid' (Participant ID) and calculate the mean for each numeric metric
    id_cols = ['ppid', 'ses-id', 'run-id', 'task', 'datetime', 'date', 'time', 'context_factor']
    avg_cols = [c for c in ema.columns if c not in id_cols and pd.api.types.is_numeric_dtype(ema[c])]
    summary = ema.groupby('ppid')[avg_cols].mean().reset_index()

    # ------------------------------------------------------------------------
    # Calculate Context-Specific Metrics (Percentages, Comfort, Social)
    # ------------------------------------------------------------------------
    if 'context_factor' in ema.columns and ema['context_factor'].notna().any():
        # Only count rows where context_factor is defined for the denominator of percentage calculations
        total_counts = ema[ema['context_factor'].notna()].groupby('ppid').size()

        # Context percentages: How often was the participant in each context?
        for ctx_name in CONTEXT_NAMES:
            ctx_counts = ema[ema['context_factor'] == ctx_name].groupby('ppid').size()
            per = (ctx_counts / total_counts * 100).reindex(total_counts.index).fillna(0).round(1)
            summary[f'context_{ctx_name}_per'] = per.values

        # Social (other, known, both) percentages: Combine all social contexts (diff from 'alone')
        social_counts = ema[ema['context_factor'].isin(CONTEXT_NAMES_SOC)].groupby('ppid').size()
        social_per = (social_counts / total_counts * 100).reindex(total_counts.index).fillna(0).round(1)
        summary['social_contexts_per'] = social_per.values

        # Comfort averages per participant per context
        if 'comfort' in ema.columns:
            for ctx_name in CONTEXT_NAMES:
                mask = ema['context_factor'] == ctx_name
                comfort_avg = ema[mask].groupby('ppid')['comfort'].mean().reindex(total_counts.index).fillna(0).round(3)
                summary[f'comfort_{ctx_name}'] = comfort_avg.values

            # Comfort average for social contexts combined (other, known, both)
            social_mask = ema['context_factor'].isin(CONTEXT_NAMES_SOC)
            comfort_social_avg = ema[social_mask].groupby('ppid')['comfort'].mean().reindex(total_counts.index).fillna(0).round(3)
            summary['comfort_social'] = comfort_social_avg.values
        else:
            LOG.warning("'comfort' column not found in EMA data")

        # Social distance and social approval averages per context (other, known, both)
        for ctx_name in CONTEXT_NAMES_SOC:
            mask = ema['context_factor'] == ctx_name
            if 'soc_distance' in ema.columns:
                soc_distance_avg = ema[mask].groupby('ppid')['soc_distance'].mean().reindex(total_counts.index).fillna(0).round(3)
                summary[f'soc_distance_{ctx_name}'] = soc_distance_avg.values
            if 'soc_app' in ema.columns:
                soc_app_avg = ema[mask].groupby('ppid')['soc_app'].mean().reindex(total_counts.index).fillna(0).round(3)
                summary[f'soc_app_{ctx_name}'] = soc_app_avg.values

        if 'soc_distance' not in ema.columns or 'soc_app' not in ema.columns:
            LOG.warning("'soc_distance' and/or 'soc_app' columns not found in EMA data")
    else:
        # Fallback: If context_factor is missing or all NaN, we cannot calculate context-specific stats.
        # Fill with zeros to ensure output schema consistency.
        LOG.warning("No valid context_factor data found; context-specific columns will be zero-filled")
        for ctx_name in CONTEXT_NAMES:
            summary[f'context_{ctx_name}_per'] = 0.0
        summary['social_contexts_per'] = 0.0
        if 'comfort' in ema.columns:
            for ctx_name in CONTEXT_NAMES:
                summary[f'comfort_{ctx_name}'] = 0.0
            summary['comfort_social'] = 0.0
        # Fill soc_distance and soc_app columns with zeros if context_factor missing
        for ctx_name in CONTEXT_NAMES_SOC:
            summary[f'soc_distance_{ctx_name}'] = 0.0
            summary[f'soc_app_{ctx_name}'] = 0.0

    # Remove deprecated context numeric suffix columns (if present) and context_factor
    drop_cols = [
        col for col in summary.columns
        if (re.match(r'context[0-9]+_per', col))
        or (col == 'context_factor')
    ]
    summary = summary.drop(columns=drop_cols, errors='ignore')

    # Remove any duplicate columns introduced by merges or derived fields
    if summary.columns.duplicated().any():
        summary = summary.loc[:, ~summary.columns.duplicated()]

    # Round all numeric columns (including individual items) to 3 decimals
    for col in summary.columns:
        if col != 'ppid' and pd.api.types.is_numeric_dtype(summary[col]):
            summary[col] = summary[col].round(3)

    # Reorder columns: id, overall averages, context-specific stats, individual items at end
    id_cols = ['ppid']

    # Overall averages (excluding context-specific ones)
    avg_prefixes = [
        'pa_avg', 'na_avg', 'pa_soc_avg', 'na_soc_avg', 'pa_nonsoc_avg', 'na_nonsoc_avg',
        'loneliness_avg', 'soc_app_avg', 'lon_app_avg',
        'neg_affect', 'pos_affect',
    ]
    avg_cols = [col for col in summary.columns if any(col.startswith(prefix) for prefix in avg_prefixes)]

    # Context-specific statistics (percentages, comfort, soc_distance, soc_app per context)
    context_cols = [col for col in summary.columns if
                   col.startswith('context_') or col.startswith('comfort_') or
                   col.startswith('social_contexts_') or col.startswith('alone_context_') or
                   col.startswith('soc_distance_') or (col.startswith('soc_app_') and col != 'soc_app_avg')]

    # All other columns (individual items) - keep only expected EMA prefixes
    allowed_item_prefixes = (
        'pa_', 'na_', 'lon_', 'soc_', 'comfort',
    )
    indiv_cols = [
        col for col in summary.columns
        if col not in id_cols + avg_cols + context_cols
        and col.startswith(allowed_item_prefixes)
    ]

    # Keep only id + numeric metrics to avoid unexpected text columns.
    new_order = id_cols + avg_cols + context_cols + indiv_cols
    new_order = list(dict.fromkeys(new_order))
    summary = summary.loc[:, new_order]
    return summary, ema

# %%
# =================================================================================================
# Main Processing Loop
# =================================================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

all_files = []
for session in sessions:
    session_filter = None if session == "all" else session
    files = find_ema_files(root, participants=subjects, session=session_filter)
    if not files:
        LOG.warning("No EMA files found for session=%s", session)
        continue
    all_files.extend(files)

processed_subs = set()
for f in all_files:
    sid = process_ema_file(Path(f), data_dir)
    if sid:
        processed_subs.add(sid)

if processed_subs:
    print("Processed EMA for participants: " + ", ".join(sorted(processed_subs)))
else:
    print("No EMA files were processed.")

processed_files = []
for sub_dir in data_dir.glob("sub-*"):
    if sub_dir.is_dir():
        processed_files.extend(sub_dir.rglob("*EMA*.tsv"))
summary_df, merged_df = generate_ema_summary(processed_files)
if summary_df.empty:
    LOG.warning("No EMA files found for summary; skipping summary output")
else:
    out_dir = derivatives_dir / "ema-analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ema_summary.tsv"
    summary_df.to_csv(out_path, sep="\t", index=False, na_rep="n/a")
    _create_summary_sidecar(out_path)
    recode_scales(summary_df)
    print(f"Saved summary table to {out_path}")

    # Output merged long-form EMA data (incorporating original ema_merge.py logic)
    merged_path = out_dir / "merged_ema_data.tsv"
    if 'ppid' in merged_df.columns:
        merged_df = merged_df.rename(columns={'ppid': 'participant_id'})
    merged_df.to_csv(merged_path, sep="\t", index=False, na_rep="n/a")
    print(f"Merged EMA data saved to {merged_path}")

# %%
