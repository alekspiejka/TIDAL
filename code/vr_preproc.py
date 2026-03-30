"""
VR Metrics Preprocessing & Extraction (EDIA Logs)
=================================================

**Description:**
    This script preprocesses relevant VR behavioral metrics derived from the Unity task logs (EDIA format).
    It extracts raw trial data, calculates derived metrics (e.g., Approach/Avoidance Ratio, Stopping Distance),
    and organizes the data into BIDS-compliant tabular files for further analysis.

**Key Features:**
    1.  **Metric Calculation**: Computes `approach_ratio`, `distance_stopped`, and `distance_travelled`.
    2.  **Data Cleaning**: Excludes practice trials and break blocks automatically.
    3.  **BIDS Standardization**: Renames columns to standard BIDS terms (e.g., `start_time` -> `onset`).
    4.  **Summary Generation**: Creates a subject-level summary file with means and standard deviations for key metrics.

**Inputs:**
    -   Unity Trial Results (`trial_results.csv`) located in `data/sourcedata/unity-edia/sub-XXX/S001/`.
    -   Project Directory Structure: Expects standard TIDAL folder hierarchy.

**Outputs:**
    -   **Behavioral Data**: `data/sub-XXX/ses-001/beh/sub-XXX_ses-001_task-VR_run-001_beh.tsv` (and `.json`).
    -   **Scans Info**: Updated `scans.tsv` in the session folder.
    -   **Summary Statistics**: `data/derivatives/vr-analysis/VR_stats_summary.tsv`.

**Usage:**
    1.  **Process All Participants**:
        `python code/vr_preproc.py`

    2.  **Process Single Participant**:
        `python code/vr_preproc.py --participant 01A`

"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings
import argparse
import sys
import os
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

# Paths (auto-detected or set via setup_paths)
PROJECT_ROOT: Path = None
EDIA_LOGS_PATH: Path = None
BIDS_OUTPUT_PATH: Path = None

# Analysis settings
EXCLUDE_PRACTICE_TRIALS = True  # Exclude first 2 practice trials (block_num == 1)
PRACTICE_BLOCK_NUM = 1

# Manual participant exclusions — full exclusion from all outputs
EXCLUDE_PARTICIPANTS: List[str] = []

# Participants whose VR questionnaire ratings are invalid — approach/distance metrics are kept,
# but current_discomfort and current_feeling are set to NaN
NULL_QUESTIONNAIRE_PARTICIPANTS: List[str] = [
    "19P",  # VR questionnaire data invalid; spatial metrics retained
]

# =============================================================================
# BIDS Column Descriptions (edit these as needed)
# =============================================================================

COLUMN_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "onset": {
        "Description": "Trial onset time (start of trial)",
        "Units": "seconds"
    },
    "duration": {
        "Description": "Duration of the trial",
        "Units": "seconds"
    },
    "experiment": {
        "Description": "TIDAL"
    },
    "ppid": {
        "Description": "Participant ID"
    },
    "session_num": {
        "Description": "Session number"
    },
    "trial_num": {
        "Description": "Trial number (global, across all blocks)"
    },
    "block_num": {
        "Description": "Block number"
    },
    "trial_num_in_block": {
        "Description": "Trial number within the current block"
    },
    "end_time": {
        "Description": "Trial end time",
        "Units": "seconds"
    },
    "speed": {
        "Description": "Agent walking speed",
        "Units": "m/s"
    },
    "distance": {
        "Description": "Initial distance between participant and agent at trial start",
        "Units": "meters"
    },
    "gender": {
        "Description": "Gender of the virtual agent",
        "Levels": {
            "M": "Male",
            "F": "Female"
        }
    },
    "actor": {
        "Description": "Actor/avatar identifier"
    },
    "blockType": {
        "Description": "Type of block",
        "Levels": {
            "task": "Experimental task block",
            "break": "Break period"
        }
    },
    "blockId": {
        "Description": "Block identifier with condition information"
    },
    "subType": {
        "Description": "Sub-type of the trial/block"
    },
    "distance_stopped": {
        "Description": "Distance between participant and agent when trial ended (agent was stopped)",
        "Units": "meters"
    },
    "current_discomfort": {
        "Description": "Participant-reported discomfort rating for this trial",
        "Units": "arbitrary units (5 point Likert scale)"
    },
    "current_feeling": {
        "Description": "Participant-reported feeling/valence rating for this trial",
        "Units": "arbitrary units (5 point Likert scale)"
    },
    "distance_travelled": {
        "Description": "Actual distance the agent travelled (distance - distance_stopped)",
        "Units": "meters"
    },
    "approach_ratio": {
        "Description": "Ratio of distance_stopped to initial distance. Lower values indicate more approach behavior, higher values indicate more avoidance.",
        "Units": "proportion (0-1)"
    }
}

# Summary statistics column descriptions (for VR_stats_summary.json)
SUMMARY_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "ppid": {"Description": "Participant ID"},
    "distance_stopped_mean": {"Description": "Mean stopping distance", "Units": "meters"},
    "distance_stopped_SD": {"Description": "SD of stopping distance", "Units": "meters"},
    "distance_stopped_min": {"Description": "Minimum stopping distance", "Units": "meters"},
    "distance_stopped_max": {"Description": "Maximum stopping distance", "Units": "meters"},
    "distance_travelled_mean": {"Description": "Mean distance travelled by agent", "Units": "meters"},
    "distance_travelled_SD": {"Description": "SD of distance travelled", "Units": "meters"},
    "distance_travelled_min": {"Description": "Minimum distance travelled", "Units": "meters"},
    "distance_travelled_max": {"Description": "Maximum distance travelled", "Units": "meters"},
    "current_discomfort_mean": {"Description": "Mean discomfort rating", "Units": "5-point Likert scale"},
    "current_discomfort_SD": {"Description": "SD of discomfort rating"},
    "current_feeling_mean": {"Description": "Mean feeling/valence rating", "Units": "5-point Likert scale"},
    "current_feeling_SD": {"Description": "SD of feeling rating"},
    "duration_mean": {"Description": "Mean trial duration", "Units": "seconds"},
    "duration_SD": {"Description": "SD of trial duration", "Units": "seconds"},
    "approach_ratio_mean": {"Description": "Mean approach ratio", "Units": "proportion"},
    "approach_ratio_SD": {"Description": "SD of approach ratio"},
    "approach_duration_mean": {"Description": "Mean duration of approach phase (calculated from distance/speed)", "Units": "seconds"},
    "approach_duration_SD": {"Description": "SD of approach phase duration"},
    "post_approach_duration_mean": {"Description": "Mean duration after approach until trial end (e.g. ratings)", "Units": "seconds"},
    "post_approach_duration_SD": {"Description": "SD of post-approach duration"},
    "n_trials": {"Description": "Number of trials per participant"}
}

def setup_paths(project_root: str = None) -> None:
    """
    Set up all paths based on project root.

    Auto-detection: If this script is in 'tidal/code/', the project root is
    automatically detected as two directories up from the script location.

    Args:
        project_root: Optional explicit path. If None, auto-detects or prompts user.
    """
    global PROJECT_ROOT, EDIA_LOGS_PATH, BIDS_OUTPUT_PATH

    print("\n" + "=" * 60)
    print("EDIA-LOGS DATA EXTRACTION - PATH SETUP")
    print("=" * 60)

    if project_root is None:
        # Auto-detect based on script location
        # Script is in: <project_root>/code/edia_data_extraction.py
        # So project_root = script_dir.parent
        script_dir = Path(__file__).resolve().parent
        auto_detected_root = script_dir.parent

        # Validate auto-detected path has expected structure
        expected_edia_path = auto_detected_root / "data" / "sourcedata" / "unity-edia"

        if expected_edia_path.exists():
            print(f"\n[OK] Auto-detected project root: {auto_detected_root}")
            user_input = input("Press Enter to use this path, or type a different path: ").strip()

            if user_input:
                # User provided a different path
                project_root = user_input.strip('"').strip("'")
            else:
                # Use auto-detected path
                project_root = str(auto_detected_root)
        else:
            # Auto-detection failed, ask user
            print("\n✗ Could not auto-detect project root (unity-edia folder not found in expected location)")
            project_root = input("Enter the project root path: ").strip()
            project_root = project_root.strip('"').strip("'")

    PROJECT_ROOT = Path(project_root)

    # Validate path exists
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Project root not found: {PROJECT_ROOT}")

    # Set derived paths
    EDIA_LOGS_PATH = PROJECT_ROOT / "data" / "sourcedata" / "unity-edia"
    BIDS_OUTPUT_PATH = PROJECT_ROOT / "data"

    # Validate unity-edia exists
    if not EDIA_LOGS_PATH.exists():
        raise FileNotFoundError(f"unity-edia folder not found: {EDIA_LOGS_PATH}")

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"EDIA logs: {EDIA_LOGS_PATH}")
    print(f"Output: {BIDS_OUTPUT_PATH}")

# =============================================================================
# Data Loading Functions
# =============================================================================

def get_participant_folders(base_path: Path) -> List[Path]:
    """Get sorted list of participant folders (ending with 'P') in the given directory."""
    return sorted(p for p in base_path.iterdir() if p.is_dir() and p.name.endswith('P'))

def load_trial_results(participant_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load trial_results.csv for a participant and get its modification time.
    Returns: (DataFrame, ISO8601 timestamp string)
    """
    trial_file = participant_path / "S001" / "trial_results.csv"

    if not trial_file.exists():
        warnings.warn(f"Trial results file not found: {trial_file}")
        return None, None

    try:
        df = pd.read_csv(trial_file)

        # Get modification time as acquisition time
        mtime = os.path.getmtime(trial_file)
        acq_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%dT%H:%M:%S')

        return df, acq_time
    except Exception as e:
        warnings.warn(f"Error loading {trial_file}: {e}")
        return None, None

# =============================================================================
# Derived Metrics Functions
# =============================================================================

def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived metrics using vectorized operations.

    Metrics computed:
    - duration: end_time - onset
    - distance_travelled: distance (initial) - distance_stopped
    - approach_ratio: distance_stopped / distance (initial).
      Note: A ratio of 1.0 means no approach (stopped where they started).
            A ratio of 0.0 means full approach (stopped at distance 0).

    Column renames for BIDS compliance:
    - start_time -> onset (Standard BIDS event start time)
    - distance_travelled (from log) -> distance_stopped (Correcting misnomer in Unity logs)
    """
    # Rename misnamed column from log files: Unity logs "distance_travelled" but it actually captures
    # the distance REMAINING between agent and participant. We rename it to 'distance_stopped' to be clear.
    if 'distance_travelled' in df.columns:
        df = df.rename(columns={'distance_travelled': 'distance_stopped'})

    if 'start_time' in df.columns:
        df = df.rename(columns={'start_time': 'onset'})

    df['duration'] = df['end_time'] - df['onset']
    # Calculate actual distance travelled by the agent
    df['distance_travelled'] = df['distance'] - df['distance_stopped']

    # Approach Ratio: Fraction of initial distance remaining.
    # We handle division by zero or NaN distances safely.
    df['approach_ratio'] = np.where(
        df['distance'].isna() | (df['distance'] == 0),
        np.nan,
        df['distance_stopped'] / df['distance']
    )

    # Approach Duration: Calculated from distance moved and speed to isolate movement phase
    # (Excludes questionnaire time which is included in raw duration)
    if 'speed' in df.columns:
        df['approach_duration'] = df['distance_travelled'] / df['speed']

        # Post-Approach Duration: Total duration minus approach duration
        # Captures time spent on ratings or waiting after the agent stops
        df['post_approach_duration'] = df['duration'] - df['approach_duration']

    # Reorder columns: onset, duration first (BIDS requirement)
    priority_cols = ['onset', 'duration']
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    return df

# =============================================================================
# Main Extraction Functions
# =============================================================================

def extract_participant_data(participant_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Extract all trial data for a single participant.

    Args:
        participant_path: Path to participant folder

    Returns:
        Tuple of (DataFrame with trial data, acq_time string)
    """
    # Load trial results
    df, acq_time = load_trial_results(participant_path)
    if df is None:
        return None, None

    print(f"  Loaded {len(df)} trials for {participant_path.name} (Acq time: {acq_time})")

    # Null out questionnaire ratings for participants with invalid VR questionnaire data
    ppid = participant_path.name  # e.g. "sub-19P"
    ppid_short = ppid.replace("sub-", "")  # e.g. "19P"
    if any(excl in ppid or excl in ppid_short for excl in NULL_QUESTIONNAIRE_PARTICIPANTS):
        for col in ("current_discomfort", "current_feeling"):
            if col in df.columns:
                df[col] = np.nan
        print(f"  NOTE: Questionnaire columns nulled for {participant_path.name} (invalid VR ratings)")

    # Drop tracker location columns (analyzed via LSL streams)
    tracker_cols = [c for c in df.columns if c.endswith('_location_0')]
    if tracker_cols:
        df = df.drop(columns=tracker_cols)

    df = compute_derived_metrics(df)

    return df, acq_time

def extract_participants(
    edia_logs_path: Path,
    participant_ids: Optional[List[str]] = None,
    exclude_practice: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Extract trial data from specified participants (or all if None).

    Args:
        edia_logs_path: Path to edia-logs folder
        participant_ids: Optional list of participant IDs (e.g., ['001', '002']) to process.
                         If None, processes all found participants.
        exclude_practice: Whether to exclude practice trials

    Returns:
        Tuple of (Combined DataFrame, Dictionary mapping ppid to acq_time)
    """
    all_participants = get_participant_folders(edia_logs_path)

    if participant_ids:
        filtered_participants = []
        for p_path in all_participants:
             if any(pid in p_path.name for pid in participant_ids):
                 filtered_participants.append(p_path)

        participants = filtered_participants
        if not participants:
            print(f"Warning: No matching folders found for IDs {participant_ids}")
            return pd.DataFrame(), {}
    else:
        participants = all_participants

    # Apply manual exclusions from EXCLUDE_PARTICIPANTS
    if EXCLUDE_PARTICIPANTS:
        before = len(participants)
        participants = [p for p in participants if not any(excl in p.name for excl in EXCLUDE_PARTICIPANTS)]
        excluded = before - len(participants)
        if excluded:
            print(f"Excluded {excluded} participant(s) (manual exclusion list: {EXCLUDE_PARTICIPANTS})")

    print(f"Found {len(participants)} participants to process")

    all_data = []
    acq_times = {}

    for participant_path in participants:
        print(f"Processing {participant_path.name}...")
        df, acq_time = extract_participant_data(participant_path)
        if df is not None:
            all_data.append(df)
            # Store acquisition time for later use in scans.tsv
            if not df.empty and 'ppid' in df.columns:
                ppid = str(df['ppid'].iloc[0])
                acq_times[ppid] = acq_time

    if not all_data:
        # Check if we should raise error or return empty.
        # If specific participants requested and not found/empty, empty return is safer than crash if we handle it in main.
        # But original raised ValueError.
        if participant_ids:
             print("No data extracted from requested participants.")
             return pd.DataFrame(), {}
        else:
             raise ValueError("No data extracted from any participant!")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal trials before filtering: {len(combined_df)}")

    # Exclude practice trials if requested
    if exclude_practice:
        combined_df = combined_df[combined_df['block_num'] != PRACTICE_BLOCK_NUM]
        print(f"Total trials after excluding practice: {len(combined_df)}")

    # Also exclude break trials
    combined_df = combined_df[combined_df['blockType'] == 'task']
    print(f"Total trials after excluding breaks: {len(combined_df)}")

    return combined_df, acq_times

def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics per participant."""

    agg_dict = {
        'distance_stopped': [('mean', 'mean'), ('SD', 'std'), ('min', 'min'), ('max', 'max')],
        'distance_travelled': [('mean', 'mean'), ('SD', 'std'), ('min', 'min'), ('max', 'max')],
        'current_discomfort': [('mean', 'mean'), ('SD', 'std')],
        'current_feeling': [('mean', 'mean'), ('SD', 'std')],
        'duration': [('mean', 'mean'), ('SD', 'std')],
        'approach_ratio': [('mean', 'mean'), ('SD', 'std')],
        'approach_duration': [('mean', 'mean'), ('SD', 'std')],
        'post_approach_duration': [('mean', 'mean'), ('SD', 'std')],
        'trial_num': [('count', 'count')]
    }

    summary = df.groupby('ppid').agg(agg_dict).round(3)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'trial_num_count': 'n_trials'})

    return summary

def create_bids_json_sidecar(columns: List[str], output_path: Path) -> None:
    """
    Create a JSON sidecar file with column descriptions for BIDS compliance.

    Args:
        columns: List of column names in the TSV file
        output_path: Path to save the JSON file (same name as TSV but .json extension)
    """
    sidecar = {}

    for col in columns:
        if col in COLUMN_DESCRIPTIONS:
            sidecar[col] = COLUMN_DESCRIPTIONS[col]
        else:
            # Add a placeholder description for undocumented columns
            sidecar[col] = {"Description": f"Column: {col}"}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, indent=2)

def save_participant_bids_files(df: pd.DataFrame, bids_path: Path, acq_times: Dict[str, str]) -> None:
    """
    Save individual participant files in BIDS format.

    Output structure:
        data/sub-{ID}/ses-001/
            beh/
                sub-{ID}_ses-001_task-VR_run-001_beh.tsv
                sub-{ID}_ses-001_task-VR_run-001_beh.json
            sub-{ID}_ses-001_scans.tsv
    """
    participants = df['ppid'].unique()

    for ppid in participants:
        # ppid can be int or str in dataframe
        ppid_str = str(ppid)

        sub_folder = bids_path / f"sub-{ppid_str}" / "ses-001" / "beh"
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Filter data for this participant
        participant_df = df[df['ppid'] == ppid].copy()

        base_filename = f"sub-{ppid_str}_ses-001_task-VR_run-001_beh"
        tsv_path = sub_folder / f"{base_filename}.tsv"
        json_path = sub_folder / f"{base_filename}.json"

        # Save TSV file (tab-separated)
        participant_df.to_csv(tsv_path, sep='\t', index=False)

        # Create corresponding JSON sidecar
        create_bids_json_sidecar(participant_df.columns.tolist(), json_path)

        print(f"   Saved: {tsv_path.name} + {json_path.name}")

        # Update scans.tsv in session folder (parent of beh)
        # Only add data files — JSON sidecars are metadata, not scans
        session_folder = sub_folder.parent

        # Get acquisition time
        acq_time = acq_times.get(ppid_str)

        # Add TSV data file
        relative_path_tsv = Path("beh") / tsv_path.name
        update_scans_tsv(session_folder, relative_path_tsv, ppid_str, acq_time)

    print(f"   Total: {len(participants)} participants saved to {bids_path}")

def update_scans_tsv(session_dir: Path, relative_file_path: Path, ppid: str, acq_time: str = None) -> None:
    """
    Update or create the scans.tsv file in the session directory.
    Appends the new file if not already present.
    """
    # Construct scans filename: sub-XX_ses-YY_scans.tsv
    # session_dir name is 'ses-001'
    session_id = session_dir.name
    scans_filename = f"sub-{ppid}_{session_id}_scans.tsv"
    scans_path = session_dir / scans_filename

    # BIDS requires forward slashes in TSV
    file_entry = str(relative_file_path).replace("\\", "/")

    # Prep new row data
    new_data = {'filename': file_entry}
    if acq_time:
        new_data['acq_time'] = acq_time

    if scans_path.exists():
        try:
            df = pd.read_csv(scans_path, sep='\t')

            # Check if filename column exists
            if 'filename' not in df.columns:
                df['filename'] = []

            # Ensure acq_time column exists if we have it
            if acq_time and 'acq_time' not in df.columns:
                df['acq_time'] = None

            # Check if entry already exists
            if file_entry not in df['filename'].values:
                new_row = pd.DataFrame([new_data])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(scans_path, sep='\t', index=False)
                print(f"   Updated scans.tsv: Added {file_entry}")
            else:
                # Already present
                pass

        except Exception as e:
            print(f"   Warning: Could not update {scans_path}: {e}")
    else:
        # Create new
        df = pd.DataFrame([new_data])
        df.to_csv(scans_path, sep='\t', index=False)
        print(f"   Created scans.tsv: {scans_path.name}")


# =============================================================================
# Main Execution
# =============================================================================

def update_summary_file(new_summary_df: pd.DataFrame, summary_path: Path) -> None:
    """
    Update the existing summary file with new participant data.
    If the file doesn't exist, it's created.
    If it exists, rows for participants in new_summary_df are updated (old rows removed, new added).
    """
    if not summary_path.exists():
        new_summary_df.to_csv(summary_path, sep='\t')
        print(f"   Created new summary statistics: {summary_path}")
        return

    # Load existing summary
    try:
        existing_df = pd.read_csv(summary_path, sep='\t', index_col=0) # Index is ppid
        # Attempt to handle if index is not ppid.
        # If 'ppid' is a column, set it as index.
        if 'ppid' in existing_df.columns:
            existing_df = existing_df.set_index('ppid')

        # Identify participants to update
        new_ppids = new_summary_df.index.tolist()

        # Cast indices to string to be safe
        existing_df.index = existing_df.index.astype(str)
        new_summary_df.index = new_summary_df.index.astype(str)

        # Drop existing rows for these participants
        filtered_existing = existing_df.drop(new_summary_df.index, errors='ignore')

        # Concatenate
        updated_df = pd.concat([filtered_existing, new_summary_df])

        # Sort by ppid
        updated_df = updated_df.sort_index()

        # Save
        updated_df.to_csv(summary_path, sep='\t')
        print(f"   Updated summary statistics: {summary_path} (Merged {len(filtered_existing)} existing + {len(new_summary_df)} new)")

    except Exception as e:
        print(f"   Error updating summary file {summary_path}: {e}")
        print("   Saving new summary to distinct file to avoid data loss.")
        backup_path = summary_path.parent / f"VR_stats_summary_new_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.tsv"
        new_summary_df.to_csv(backup_path, sep='\t')
        print(f"   Saved to {backup_path}")

def main(project_root: str = None, participant_id: str = None):
    """
    Main extraction pipeline.

    Args:
        project_root: Optional path to project root. If not provided, prompts user.
        participant_id: Optional single participant ID to process (e.g. '001', 'sub-001', '01A').
                        If None, processes all participants found in data folder.
    """
    # Set up paths (prompts user if not provided)
    setup_paths(project_root)

    print("\n" + "=" * 60)
    print("EDIA-LOGS DATA EXTRACTION")
    print("=" * 60)

    BIDS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print("\n1. Extracting trial data...")

    target_participants = [participant_id] if participant_id else None

    trial_df, acq_times = extract_participants(
        EDIA_LOGS_PATH,
        participant_ids=target_participants,
        exclude_practice=EXCLUDE_PRACTICE_TRIALS
    )

    if trial_df.empty:
        print("No data extracted. Exiting.")
        return None, None

    # Create participant summaries
    print("\n2. Creating summary statistics...")
    summary_df = create_summary_statistics(trial_df)

    # Save outputs
    print("\n3. Saving outputs...")

    # Per-participant files
    print("   Saving per-participant files...")
    save_participant_bids_files(trial_df, BIDS_OUTPUT_PATH, acq_times)

    # Summary statistics to derivatives folder
    derivatives_path = PROJECT_ROOT / "data" / "derivatives" / "vr-analysis"
    derivatives_path.mkdir(parents=True, exist_ok=True)

    # Save as TSV (BIDS-compliant)
    summary_tsv_path = derivatives_path / "VR_stats_summary.tsv"

    if participant_id:
        print(f"   Updating summary for participant {participant_id}...")
        update_summary_file(summary_df, summary_tsv_path)
    else:
        # Full run, overwrite
        summary_df.to_csv(summary_tsv_path, sep='\t')
        print(f"   Saved summary statistics: {summary_tsv_path}")

    summary_json_path = derivatives_path / "VR_stats_summary.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(SUMMARY_DESCRIPTIONS, f, indent=2)
    print(f"   Saved JSON sidecar: {summary_json_path}")

    # Print data overview
    print("\n" + "=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"\nParticipants: {trial_df['ppid'].nunique()}")
    print(f"Total trials: {len(trial_df)}")
    print(f"Trials per participant: {trial_df.groupby('ppid').size().to_dict()}")

    print("\nCondition distribution:")
    print(f"  Gender: {trial_df['gender'].value_counts().to_dict()}")
    print(f"  Actor: {trial_df['actor'].value_counts().to_dict()}")
    print(f"  Block: {trial_df['blockId'].value_counts().to_dict()}")

    print("\nKey variable ranges:")
    print(f"  distance_stopped: {trial_df['distance_stopped'].min():.2f} - {trial_df['distance_stopped'].max():.2f}")
    print(f"  distance_travelled: {trial_df['distance_travelled'].min():.2f} - {trial_df['distance_travelled'].max():.2f}")
    print(f"  current_discomfort: {trial_df['current_discomfort'].min()} - {trial_df['current_discomfort'].max()}")
    print(f"  current_feeling: {trial_df['current_feeling'].min()} - {trial_df['current_feeling'].max()}")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    return trial_df, summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EDIA VR data.")
    parser.add_argument("--root", type=str, help="Path to project root", default=None)
    parser.add_argument("--participant", type=str, help="Specific participant ID to process (e.g. '01A')", default=None)
    parser.add_argument("--exclude", type=str, nargs="+", help="Additional participant IDs to exclude (e.g. --exclude 19P 20P)", default=None)

    args = parser.parse_args()

    if args.exclude:
        EXCLUDE_PARTICIPANTS.extend(args.exclude)

    trial_data, summaries = main(project_root=args.root, participant_id=args.participant)
