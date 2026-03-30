# %%
"""
VR Summary Statistics Split by Speed
=====================================
Creates separate VR summary statistics for fast (speed > 2.0) vs slow (speed <= 2.0) trials.
Reads trial_results.csv files and aggregates participant-level metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# %%
# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SOURCEDATA_DIR = DATA_DIR / "sourcedata" / "unity-edia"
DERIVATIVES_DIR = DATA_DIR / "derivatives" / "vr-analysis"

SPEED_THRESHOLD = 2.0  # m/s
OUTPUT_PATH_FAST = DERIVATIVES_DIR / "VR_stats_summary_fast.tsv"
OUTPUT_PATH_SLOW = DERIVATIVES_DIR / "VR_stats_summary_slow.tsv"

# Participants whose VR questionnaire ratings are invalid — keep in sync with vr_preproc.py
# Spatial/distance metrics are retained; only current_discomfort and current_feeling are nulled
NULL_QUESTIONNAIRE_PARTICIPANTS: list[str] = [
    "19P",  # VR questionnaire data invalid; spatial metrics retained
]

# %%
def load_trial_results(sourcedata_dir=SOURCEDATA_DIR):
    """Load all trial_results.csv files and return combined dataframe"""
    all_data = []
    
    for csv_file in sorted(sourcedata_dir.rglob("trial_results.csv")):
        ppid = csv_file.parts[-3]

        try:
            df = pd.read_csv(csv_file)

            # Null questionnaire ratings for participants with invalid VR questionnaire data
            if any(excl in ppid for excl in NULL_QUESTIONNAIRE_PARTICIPANTS):
                for col in ("current_discomfort", "current_feeling"):
                    if col in df.columns:
                        df[col] = np.nan
                print(f"NOTE: Questionnaire columns nulled for {ppid} (invalid VR ratings)")
            
            # Filter to experiment trials only
            df = df[
                (df["blockType"] == "task") &
                (df["blockId"].str.contains("experiment", na=False))
            ].dropna(subset=["gender"])
            
            df['ppid'] = ppid
            all_data.append(df)
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No trial data found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} trials from {combined_df['ppid'].nunique()} participants")
    
    return combined_df

# %%
def create_summary_statistics(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Create summary statistics per participant"""
    
    # Calculate derived metrics (matching vr_preproc.py logic)
    if 'distance_travelled' in df.columns:
        df = df.rename(columns={'distance_travelled': 'distance_stopped'})
    
    if 'start_time' in df.columns:
        df = df.rename(columns={'start_time': 'onset'})
    
    # Calculate actual distance travelled
    if 'distance' in df.columns and 'distance_stopped' in df.columns:
        df['distance_travelled'] = df['distance'] - df['distance_stopped']
    
    # Approach ratio
    df['approach_ratio'] = np.where(
        df['distance'].isna() | (df['distance'] == 0),
        np.nan,
        df['distance_stopped'] / df['distance']
    )
    
    # Duration
    if 'end_time' in df.columns and 'onset' in df.columns:
        df['duration'] = df['end_time'] - df['onset']
    
    # Approach duration
    if 'speed' in df.columns and 'distance_travelled' in df.columns:
        df['approach_duration'] = df['distance_travelled'] / df['speed']
        if 'duration' in df.columns:
            df['post_approach_duration'] = df['duration'] - df['approach_duration']
    
    # Aggregate by participant
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
    
    print(f"{label}: {len(summary)} participants, mean trials per participant: {summary['n_trials'].mean():.1f}")
    
    return summary

# %%
# Load all trial data
print("\n" + "="*60)
print("Loading trial data...")
print("="*60)
df_all = load_trial_results()

# Split by speed
df_fast = df_all[df_all['speed'] > SPEED_THRESHOLD]
df_slow = df_all[df_all['speed'] <= SPEED_THRESHOLD]

print(f"\nFast trials (speed > {SPEED_THRESHOLD}): {len(df_fast)} trials")
print(f"Slow trials (speed <= {SPEED_THRESHOLD}): {len(df_slow)} trials")

# Create summaries
print("\n" + "="*60)
print("Creating summaries...")
print("="*60)

summary_fast = create_summary_statistics(df_fast, "Fast trials")
summary_slow = create_summary_statistics(df_slow, "Slow trials")

# Save to files
DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)

summary_fast.to_csv(OUTPUT_PATH_FAST, sep='\t')
print(f"\nSaved: {OUTPUT_PATH_FAST}")

summary_slow.to_csv(OUTPUT_PATH_SLOW, sep='\t')
print(f"Saved: {OUTPUT_PATH_SLOW}")

print("\n✓ VR summaries by speed completed successfully!")

# %%
