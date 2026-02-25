"""
Questionnaire Data Preprocessing & Scoring
==========================================

**Description:**
    This script processes raw questionnaire data exported from SoSci Survey (CSV format).
    It extracts data for specific scales (R-UCLA, CESD-R, DACOBS, BIS/BAS, MAIA-2, SSQ, VR Experience),
    computes subscale and total scores, and saves each questionnaire as a separate BIDS-compliant TSV file
    with a corresponding JSON sidecar.

**Key Features:**
    1.  **Automated Scoring**: Implements scoring logic for multiple standard psychological scales, including reverse coding and subscale aggregation.
    2.  **Participant Mapping**: Matches raw SoSci IDs to BIDS participant IDs (sub-XXX) by scanning the `sourcedata` directory.
    3.  **BIDS Compliance**: Saves outputs with standardized naming conventions and detailed metadata in JSON sidecars.

**Inputs:**
    -   Raw SoSci Data CSV: `data/sourcedata/questionnaire_data.csv` (or similar).
    -   Participant Folders: `data/sourcedata/sub-XXX` (used for ID matching).

**Outputs:**
    -   **Phenotype Data**: `data/derivatives/phenotype/*.tsv` (e.g., `cesdr_depression.tsv`, `rucla_loneliness.tsv`).
    -   **Metadata**: `data/derivatives/phenotype/*.json` (JSON sidecars for each TSV).

**Usage:**
    `python code/questionnaire_preproc.py`
    (Prompts for project root if not found automatically)

"""

import pandas as pd
import re
import os
import json
import glob
from pathlib import Path

# Questionnaire prefixes to extract
QUESTIONNAIRE_PREFIXES = ['BB', 'CE', 'DA', 'MA', 'SQ', 'UC', 'VR']

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_questionnaire_columns(df, prefix):
    """
    Extract columns that match pattern: 2 letters + 2 digits (e.g., BB01_01).
    This excludes columns like 'MAILSENT' that happen to start with 'MA'.
    """
    pattern = re.compile(rf'^{prefix}\d{{2}}')
    return [col for col in df.columns if pattern.match(col)]


def reverse_score(series, scale_min, scale_max):
    """Reverse score a series: new_value = scale_max + scale_min - original_value"""
    return scale_max + scale_min - series


def get_item_columns(df, prefix, item_numbers):
    """Get column names for specific item numbers within a questionnaire."""
    cols = []
    for item in item_numbers:
        # Find columns matching the pattern (e.g., BB01_03 for item 3)
        matching = [c for c in df.columns if re.match(rf'^{prefix}\d{{2}}_{item:02d}$', c)]
        cols.extend(matching)
    return cols


def find_participant_folder(output_dir, participant_id):
    """
    Find the participant folder in output_dir.

    Logic:
    1. Try exact string match (e.g., 'sub-01P').
    2. Fuzzy match: Extract numeric ID (e.g., '01') and look for any folder containing 'sub-01'.
       - If multiple matches found, ask user to disambiguate.

    Returns the folder path or None if not found/rejected.
    """
    # List all subdirectories in output_dir
    try:
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    except FileNotFoundError:
        print(f"Error: Output directory '{output_dir}' not found.")
        return None

    # Check for exact match
    if participant_id in subdirs:
        return os.path.join(output_dir, participant_id)

    # Extract number from participant_id (e.g., "sub-01B" -> "01")
    match = re.search(r'(\d+)', participant_id)
    if not match:
        print(f"Warning: Could not extract number from participant ID '{participant_id}'")
        return None

    participant_num = match.group(1)

    # Look for folders with the same number
    similar_folders = []
    for subdir in subdirs:
        if re.search(rf'sub-{participant_num}', subdir):
            similar_folders.append(subdir)

    if not similar_folders:
        print(f"Warning: No matching folder found for participant '{participant_id}'")
        return None

    if len(similar_folders) == 1:
        similar = similar_folders[0]
        response = input(f"Participant '{participant_id}' not found. Use '{similar}' instead? (y/n): ").strip().lower()
        if response == 'y':
            return os.path.join(output_dir, similar)
        else:
            print(f"Skipping participant '{participant_id}'")
            return None
    else:
        print(f"Multiple similar folders found for '{participant_id}': {similar_folders}")
        for i, folder in enumerate(similar_folders):
            print(f"  [{i+1}] {folder}")
        choice = input("Enter number to select, or 'n' to skip: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(similar_folders):
            return os.path.join(output_dir, similar_folders[int(choice)-1])
        else:
            print(f"Skipping participant '{participant_id}'")
            return None


# ============================================================
# SCORING FUNCTIONS FOR EACH QUESTIONNAIRE
# ============================================================

def score_uc_rucla(df):
    """
    R-UCLA Loneliness Scale (UC)
    - Scale: 1 (Never) to 4 (Often)
    - Reverse items: 1, 5, 6, 9, 10, 15, 16, 19, 20
    - Total score: Sum of all 20 items
    """
    uc_cols = get_questionnaire_columns(df, 'UC')
    uc_df = df[uc_cols].copy()

    reverse_items = [1, 5, 6, 9, 10, 15, 16, 19, 20]
    reverse_cols = get_item_columns(df, 'UC', reverse_items)

    for col in reverse_cols:
        if col in uc_df.columns:
            uc_df[col] = reverse_score(uc_df[col], 1, 4)

    result = pd.DataFrame()
    result['UC_Total'] = uc_df.sum(axis=1)

    return result


def score_ce_cesdr(df):
    """
    CESD-R Depression Scale (CE)

    Scoring Logic:
    - Original scale in SoSci is 0-4.
    - Standard CESD-R often uses 0-4, but some versions recode 4s to 3s.
    - Here, we RECODE values: 0->0, 1->1, 2->2, 3->3, 4->3.
      (This caps the max item score at 3).
    - Total score = Sum of 20 recoded items (Range 0-60).
    - Subscales are calculated using the original 0-4 values (sums of specific items).
    """
    ce_cols = get_questionnaire_columns(df, 'CE')
    ce_df = df[ce_cols].copy()

    # Recode for total score: 0->0, 1->1, 2->2, 3->3, 4->3 (only 4s become 3s)
    # This aligns the total score calculation with specific CESD-R variations.
    recode_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
    ce_recoded = ce_df.replace(recode_map)

    # Subscale definitions (item numbers)
    subscales = {
        'Sadness': [2, 4, 6],
        'Anhedonia': [8, 10],
        'Appetite': [1, 18],
        'Sleep': [5, 11, 19],
        'Thinking': [3, 20],
        'Guilt': [9, 17],
        'Tired': [7, 16],
        'Movement': [12, 13],
        'Suicidal': [14, 15]
    }

    result = pd.DataFrame()
    # Total score uses recoded values (0-3 scale)
    result['CE_Total'] = ce_recoded.sum(axis=1)

    # Subscales use original values (1-5 scale, no recoding)
    for subscale_name, items in subscales.items():
        subscale_cols = get_item_columns(df, 'CE', items)
        valid_cols = [c for c in subscale_cols if c in ce_df.columns]
        if valid_cols:
            result[f'CE_{subscale_name}'] = ce_df[valid_cols].sum(axis=1)

    return result


def score_da_dacobs(df):
    """
    DACOBS-42 Cognitive Biases Scale (DA)
    - Scale: 1-7
    - Subscales calculated by summing specific items
    """
    da_cols = get_questionnaire_columns(df, 'DA')
    da_df = df[da_cols].copy()

    # Subscale definitions
    subscales = {
        'JumpingToConclusions': [3, 8, 16, 18, 25, 30],
        'BeliefInflexibility': [13, 15, 26, 34, 38, 41],
        'AttentionForThreat': [1, 2, 6, 10, 20, 37],
        'ExternalAttribution': [7, 12, 17, 22, 24, 29],
        'SocialCognitionProblems': [4, 9, 11, 14, 19, 39],
        'SubjectiveCognitiveProblems': [5, 21, 28, 32, 36, 40],
        'SafetyBehaviors': [23, 27, 31, 33, 35, 42]
    }

    result = pd.DataFrame()

    for subscale_name, items in subscales.items():
        subscale_cols = get_item_columns(df, 'DA', items)
        valid_cols = [c for c in subscale_cols if c in da_df.columns]
        if valid_cols:
            result[f'DA_{subscale_name}'] = da_df[valid_cols].sum(axis=1)

    # Also calculate total
    result['DA_Total'] = da_df.sum(axis=1)

    return result


def score_bb_bisbas(df):
    """
    BIS/BAS Scale (BB)
    - Scale: 1 (very true) to 4 (very false)
    - Reverse score ALL items EXCEPT 2 and 22
    - Filler items (excluded): 1, 6, 11, 17
    - Subscales:
      - BAS Drive: 3, 9, 12, 21
      - BAS Fun Seeking: 5, 10, 15, 20
      - BAS Reward Responsiveness: 4, 7, 14, 18, 23
      - BIS: 2, 8, 13, 16, 19, 22, 24
    """
    bb_cols = get_questionnaire_columns(df, 'BB')
    bb_df = df[bb_cols].copy()

    # Filler items to exclude
    filler_items = [1, 6, 11, 17]

    # Items NOT to reverse: 2, 22
    no_reverse = [2, 22]

    # Reverse all items except 2 and 22
    all_items = list(range(1, 25))  # Items 1-24
    reverse_items = [i for i in all_items if i not in no_reverse and i not in filler_items]
    reverse_cols = get_item_columns(df, 'BB', reverse_items)

    for col in reverse_cols:
        if col in bb_df.columns:
            bb_df[col] = reverse_score(bb_df[col], 1, 4)

    # Subscale definitions
    subscales = {
        'BAS_Drive': [3, 9, 12, 21],
        'BAS_FunSeeking': [5, 10, 15, 20],
        'BAS_RewardResponsiveness': [4, 7, 14, 18, 23],
        'BIS': [2, 8, 13, 16, 19, 22, 24]
    }

    result = pd.DataFrame()

    for subscale_name, items in subscales.items():
        subscale_cols = get_item_columns(df, 'BB', items)
        valid_cols = [c for c in subscale_cols if c in bb_df.columns]
        if valid_cols:
            result[f'BB_{subscale_name}'] = bb_df[valid_cols].sum(axis=1)

    return result


def score_ma_maia(df):
    """
    MAIA-2 - Multidimensional Assessment of Interoceptive Awareness (MA)

    Scoring Logic:
    - The standard MAIA scale is 0 (Never) to 5 (Always).
    - SoSci Survey often uses 1-6. We must REMAP: 1->0, 2->1, ..., 6->5.
    - Subscales are calculated as the MEAN of their items (not Sum).
    - Reverse scoring applies to specific items (Not-Distracting, Not-Worrying).
    """
    ma_cols = get_questionnaire_columns(df, 'MA')
    ma_df = df[ma_cols].copy()

    # Remap from SoSci scale (1-6) to MAIA scale (0-5)
    # This normalization is crucial before any other calculations.
    remap_scale = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    ma_df = ma_df.replace(remap_scale)

    # Reverse score items 5-10 (Not-Distracting) and 11,12,15 (Not-Worrying)
    reverse_items = [5, 6, 7, 8, 9, 10, 11, 12, 15]
    reverse_cols = get_item_columns(df, 'MA', reverse_items)

    for col in reverse_cols:
        if col in ma_df.columns:
            ma_df[col] = reverse_score(ma_df[col], 0, 5)

    # MAIA-2 Subscale definitions (37 items) - scored as MEAN
    subscales = {
        'Noticing': [1, 2, 3, 4],
        'NotDistracting': [5, 6, 7, 8, 9, 10],       # all reversed
        'NotWorrying': [11, 12, 13, 14, 15],         # 11, 12, 15 reversed
        'AttentionRegulation': [16, 17, 18, 19, 20, 21, 22],
        'EmotionalAwareness': [23, 24, 25, 26, 27],
        'SelfRegulation': [28, 29, 30, 31],
        'BodyListening': [32, 33, 34],
        'Trusting': [35, 36, 37]
    }

    result = pd.DataFrame()

    for subscale_name, items in subscales.items():
        subscale_cols = get_item_columns(df, 'MA', items)
        valid_cols = [c for c in subscale_cols if c in ma_df.columns]
        if valid_cols:
            # MAIA uses MEAN, not sum
            result[f'MA_{subscale_name}'] = ma_df[valid_cols].mean(axis=1)

    return result


def extract_sq_ssq(df):
    """
    Simulator Sickness Questionnaire (SQ/SSQ)
    - Raw items only, no scoring
    - 6 items about simulator sickness symptoms
    """
    sq_cols = get_questionnaire_columns(df, 'SQ')
    if not sq_cols:
        return pd.DataFrame()

    result = df[sq_cols].copy()
    # Rename columns to more descriptive names
    rename_map = {
        'SQ01_01': 'SSQ_general_discomfort',
        'SQ01_02': 'SSQ_fatigue',
        'SQ01_03': 'SSQ_headache',
        'SQ01_04': 'SSQ_eye_strain',
        'SQ01_05': 'SSQ_difficulty_focusing',
        'SQ01_06': 'SSQ_nausea'
    }
    result = result.rename(columns=rename_map)
    return result


def extract_vr_experience(df):
    """
    VR Experience Questionnaire
    - Raw items only, no scoring
    - 2 items about prior VR experience
    """
    vr_cols = [col for col in df.columns if col.startswith('VR')]
    if not vr_cols:
        return pd.DataFrame()

    result = df[vr_cols].copy()
    # Rename columns to more descriptive names
    rename_map = {
        'VR01': 'VR_prior_experience',
        'VR02': 'VR_frequency_of_use'
    }
    result = result.rename(columns=rename_map)
    return result


def score_participant(df_row):
    """Score a single participant (single row DataFrame)."""
    # Create a single-row DataFrame for scoring functions
    df_single = pd.DataFrame([df_row])

    scores = {}

    # Add participant ID
    if 'DQ01_01' in df_single.columns:
        scores['participant_id'] = df_single['DQ01_01'].iloc[0]

    # UC (R-UCLA)
    uc_scores = score_uc_rucla(df_single)
    for col in uc_scores.columns:
        scores[col] = uc_scores[col].iloc[0]

    # CE (CESD-R)
    ce_scores = score_ce_cesdr(df_single)
    for col in ce_scores.columns:
        scores[col] = ce_scores[col].iloc[0]

    # DA (DACOBS)
    da_scores = score_da_dacobs(df_single)
    for col in da_scores.columns:
        scores[col] = da_scores[col].iloc[0]

    # BB (BIS/BAS)
    bb_scores = score_bb_bisbas(df_single)
    for col in bb_scores.columns:
        scores[col] = bb_scores[col].iloc[0]

    # MA (MAIA)
    ma_scores = score_ma_maia(df_single)
    for col in ma_scores.columns:
        scores[col] = ma_scores[col].iloc[0]

    return scores


# ============================================================
# JSON SIDECAR DEFINITIONS
# ============================================================

JSON_SIDECARS = {
    'UC': {
        "participant_id": {"Description": "Unique participant identifier"},
        "UC_Total": {"Description": "R-UCLA Loneliness Scale - Total Score", "LongName": "Loneliness Total"}
    },
    'CE': {
        "participant_id": {"Description": "Unique participant identifier"},
        "CE_Total": {"Description": "CESD-R Depression Scale - Total Score", "LongName": "Depression Total"},
        "CE_Sadness": {"Description": "CESD-R Sadness Subscale"},
        "CE_Anhedonia": {"Description": "CESD-R Anhedonia Subscale"},
        "CE_Appetite": {"Description": "CESD-R Appetite Subscale"},
        "CE_Sleep": {"Description": "CESD-R Sleep Subscale"},
        "CE_Thinking": {"Description": "CESD-R Thinking/Concentration Subscale"},
        "CE_Guilt": {"Description": "CESD-R Guilt/Worthlessness Subscale"},
        "CE_Tired": {"Description": "CESD-R Fatigue Subscale"},
        "CE_Movement": {"Description": "CESD-R Psychomotor Subscale"},
        "CE_Suicidal": {"Description": "CESD-R Suicidal Ideation Subscale"}
    },
    'DA': {
        "participant_id": {"Description": "Unique participant identifier"},
        "DA_JumpingToConclusions": {"Description": "Davos Assessment - Jumping to Conclusions Subscale"},
        "DA_BeliefInflexibility": {"Description": "Davos Assessment - Belief Inflexibility Subscale"},
        "DA_AttentionForThreat": {"Description": "Davos Assessment - Attention for Threat Subscale"},
        "DA_ExternalAttribution": {"Description": "Davos Assessment - External Attribution Subscale"},
        "DA_SocialCognitionProblems": {"Description": "Davos Assessment - Social Cognition Problems Subscale"},
        "DA_SubjectiveCognitiveProblems": {"Description": "Davos Assessment - Subjective Cognitive Problems Subscale"},
        "DA_SafetyBehaviors": {"Description": "Davos Assessment - Safety Behaviors Subscale"},
        "DA_Total": {"Description": "Davos Assessment - Total Score"}
    },
    'BB': {
        "participant_id": {"Description": "Unique participant identifier"},
        "BB_BAS_Drive": {"Description": "BIS/BAS Scale - BAS Drive Subscale"},
        "BB_BAS_FunSeeking": {"Description": "BIS/BAS Scale - BAS Fun Seeking Subscale"},
        "BB_BAS_RewardResponsiveness": {"Description": "BIS/BAS Scale - BAS Reward Responsiveness Subscale"},
        "BB_BIS": {"Description": "BIS/BAS Scale - Behavioral Inhibition System Subscale"}
    },
    'MA': {
        "participant_id": {"Description": "Unique participant identifier"},
        "MA_Noticing": {"Description": "MAIA - Noticing Subscale", "LongName": "Multidimensional Assessment of Interoceptive Awareness - Noticing"},
        "MA_NotDistracting": {"Description": "MAIA - Not Distracting Subscale"},
        "MA_NotWorrying": {"Description": "MAIA - Not Worrying Subscale"},
        "MA_AttentionRegulation": {"Description": "MAIA - Attention Regulation Subscale"},
        "MA_EmotionalAwareness": {"Description": "MAIA - Emotional Awareness Subscale"},
        "MA_SelfRegulation": {"Description": "MAIA - Self-Regulation Subscale"},
        "MA_BodyListening": {"Description": "MAIA - Body Listening Subscale"},
        "MA_Trusting": {"Description": "MAIA - Trusting Subscale"}
    },
    'SSQ': {
        "participant_id": {"Description": "Unique participant identifier"},
        "SSQ_general_discomfort": {"Description": "SSQ Item - General discomfort rating"},
        "SSQ_fatigue": {"Description": "SSQ Item - Fatigue rating"},
        "SSQ_headache": {"Description": "SSQ Item - Headache rating"},
        "SSQ_eye_strain": {"Description": "SSQ Item - Eye strain rating"},
        "SSQ_difficulty_focusing": {"Description": "SSQ Item - Difficulty focusing rating"},
        "SSQ_nausea": {"Description": "SSQ Item - Nausea rating"}
    },
    'VR': {
        "participant_id": {"Description": "Unique participant identifier"},
        "VR_prior_experience": {"Description": "Prior VR experience (1=yes, 2=no)"},
        "VR_frequency_of_use": {"Description": "Frequency of VR use"}
    }
}

QUESTIONNAIRE_NAMES = {
    'UC': 'RUCLA_loneliness',
    'CE': 'CESDR_depression',
    'DA': 'DACOBS_cognitive_biases',
    'BB': 'BISBAS_behavioral_activation',
    'MA': 'MAIA_interoception',
    'SSQ': 'SSQ_simulator_sickness',
    'VR': 'VR_experience'
}


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():
    # Ask for project root path
    # Auto-detect based on script location: <project_root>/code/questionnaire_preproc.py
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent

    print(f"Enter the project root path (where 'data' folder is located)")
    print(f"  [Default: {default_root}]")

    project_root_input = input("  Path: ").strip().strip('"').strip("'")
    if project_root_input:
        project_root = Path(project_root_input)
    else:
        project_root = default_root

    # Derive paths from project root
    # Handle case where user points directly to "data" folder
    if project_root.name == 'data':
        data_dir = project_root
    else:
        data_dir = project_root / "data"

    sourcedata_dir = data_dir / "sourcedata"
    derivatives_dir = data_dir / "derivatives"

    # Find questionnaire CSV file in sourcedata
    # Ensure directory exists
    if not sourcedata_dir.exists():
         print(f"Error: Directory not found: {sourcedata_dir}")
         return

    # Use Path.glob (returns generator, convert to list)
    questionnaire_files = [f for f in sourcedata_dir.glob("*.csv") if "questionnaire" in f.name.lower()]

    if not questionnaire_files:
        print(f"Error: No questionnaire CSV file found in {sourcedata_dir}")
        return

    if len(questionnaire_files) > 1:
        print(f"Found multiple questionnaire files: {questionnaire_files}")
        data_path = questionnaire_files[0]
        print(f"Using: {data_path}")
    else:
        data_path = questionnaire_files[0]

    # Output directory is the derivatives folder
    output_dir = derivatives_dir

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, sep=';')

    # Extract and display columns per questionnaire
    print("\n--- Questionnaire Column Counts ---")
    for prefix in QUESTIONNAIRE_PREFIXES:
        cols = get_questionnaire_columns(df, prefix)
        print(f"{prefix}: {len(cols)} items")

    # Create phenotype folder
    phenotype_folder = output_dir / 'phenotype'
    phenotype_folder.mkdir(exist_ok=True)

    # Build participant ID mapping (look in sourcedata for participant folders)
    print(f"\n--- Processing {len(df)} participants ---")
    participant_ids = []
    for idx, row in df.iterrows():
        raw_id = row.get('DQ01_01', f'unknown_{idx}')
        participant_folder = find_participant_folder(sourcedata_dir, raw_id)
        if participant_folder:
            folder_name = os.path.basename(participant_folder)
            participant_ids.append(folder_name)
        else:
            participant_ids.append(None)

    # Filter to valid participants only
    valid_mask = [pid is not None for pid in participant_ids]
    df_valid = df[valid_mask].copy()
    valid_participant_ids = [pid for pid in participant_ids if pid is not None]

    print(f"\nValid participants: {len(valid_participant_ids)}")

    # Process each questionnaire separately
    print("\n--- Saving individual questionnaire files ---")

    # UC (R-UCLA Loneliness)
    uc_data = score_uc_rucla(df_valid)
    uc_data.insert(0, 'participant_id', valid_participant_ids)
    save_questionnaire_file(uc_data, 'UC', phenotype_folder)

    # CE (CESD-R Depression)
    ce_data = score_ce_cesdr(df_valid)
    ce_data.insert(0, 'participant_id', valid_participant_ids)
    save_questionnaire_file(ce_data, 'CE', phenotype_folder)

    # DA (DACOBS)
    da_data = score_da_dacobs(df_valid)
    da_data.insert(0, 'participant_id', valid_participant_ids)
    save_questionnaire_file(da_data, 'DA', phenotype_folder)

    # BB (BIS/BAS)
    bb_data = score_bb_bisbas(df_valid)
    bb_data.insert(0, 'participant_id', valid_participant_ids)
    save_questionnaire_file(bb_data, 'BB', phenotype_folder)

    # MA (MAIA)
    ma_data = score_ma_maia(df_valid)
    ma_data.insert(0, 'participant_id', valid_participant_ids)
    save_questionnaire_file(ma_data, 'MA', phenotype_folder)

    # SSQ (Simulator Sickness)
    ssq_data = extract_sq_ssq(df_valid)
    ssq_data.insert(0, 'participant_id', valid_participant_ids)
    save_questionnaire_file(ssq_data, 'SSQ', phenotype_folder)

    # VR Experience
    vr_data = extract_vr_experience(df_valid)
    vr_data.insert(0, 'participant_id', valid_participant_ids)
    save_questionnaire_file(vr_data, 'VR', phenotype_folder)

    print(f"\n--- Summary ---")
    print(f"Processed: {len(valid_participant_ids)} participants")
    print(f"Saved {len(QUESTIONNAIRE_NAMES)} questionnaire files to: {phenotype_folder}")


def save_questionnaire_file(data_df, questionnaire_key, output_folder):
    """Save a questionnaire dataframe as TSV with JSON sidecar."""
    # Sort by participant ID
    data_df = data_df.copy()
    # Ensure participant_id is string type before using .str accessor
    data_df['participant_id'] = data_df['participant_id'].astype(str)
    data_df['_sort_key'] = data_df['participant_id'].str.extract(r'(\d+)').astype(int)
    data_df = data_df.sort_values('_sort_key').drop(columns=['_sort_key']).reset_index(drop=True)

    # Round float values
    for col in data_df.columns:
        if data_df[col].dtype == 'float64':
            data_df[col] = data_df[col].round(3)

    # Get file name
    file_name = QUESTIONNAIRE_NAMES.get(questionnaire_key, questionnaire_key.lower())

    # Save TSV
    tsv_path = os.path.join(output_folder, f'{file_name}.tsv')
    data_df.to_csv(tsv_path, sep='\t', index=False)
    print(f"  Saved: {file_name}.tsv")

    # Save JSON sidecar
    json_path = os.path.join(output_folder, f'{file_name}.json')
    json_sidecar = JSON_SIDECARS.get(questionnaire_key, {})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_sidecar, f, indent=4)
    print(f"  Saved: {file_name}.json")


if __name__ == '__main__':
    main()

