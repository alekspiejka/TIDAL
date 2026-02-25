"""
TIDAL Experiment - Randomized Trial Generator
=============================================

Description:
    This script is the master generator for experimental trials in the Tidal VR study.
    It manages the creation of participant data folders, generates randomized and balanced
    event lists for Unity, and aggregates all trial data into a master summary file.

Key Features:
    1. **Participant Management**:
       - Auto-detects existing participants and finds the next sequential ID.
       - Creates new participant folders by copying a specified template.
       - Updates session metadata (Site, Stage) in `session-info.json` for each participant.

    2. **Trial Generation**:
       - Generates pseudo-randomized trial lists for multiple blocks (e.g., Block 1, Block 2).
       - **Constraints**: Enforces strict rules:
            * No immediate repetition of (Gender, Actor) pairs.
            * Anti-streak rules (max 3 consecutive trials of same Gender or Speed).
       - **Balancing**: Ensures equal repetitions of all conditions (Gender x Actor x Speed) per block.

    3. **Data Aggregation**:
       - Outputs JSON files for Unity execution.
       - Appends generated trials to a master `all_trials_summary.csv` for analysis.
       - Can reconstruct the summary CSV from existing participant folders (Repair Mode).

Usage:
    Run the script and follow the interactive terminal prompts:
    `python generate_trials_tidal.py`

    Prompts will ask for:
    - Root directory for participants.
    - Template folder name.
    - Number of new participants to generate.
    - Site and Stage information.
"""
# %%
import json
import random
import os
import glob
import shutil
import datetime
import sys
import csv

# ==========================================
#              CONFIGURATION
# ==========================================

# File settings
# NOTE: ROOT_PATH will be provided via terminal input
ROOT_PATH = None  # Will be set via terminal input
PARTICIPANTS_ROOT = None  # Will be set via terminal input
FILENAME_TEMPLATE = "task-tidal_experiment_{}.json" # {} will be replaced by block index

# Logging
LOG_FILE = None  # Will be set after ROOT_PATH is provided

# Generation Flags
# NOTE: These will be provided via terminal input
USE_DETERMINISTIC_SEED = None   # If True, trials are reproducible per participant ID
ENABLE_OVERWRITE_CHECK = None   # If True, asks for confirmation before overwriting files

# Participant Generation Settings
# NOTE: Values below will be provided via terminal input (except PARTICIPANT_PREFIX)
TEMPLATE_PARTICIPANT = None  # Folder to copy from
TOTAL_PARTICIPANTS = None
PARTICIPANT_PREFIX = "sub-"  # Always the same
PARTICIPANT_SUFFIX = None

# Experiment Design Parameters
ACTOR_IDS = ["1", "2", "3", "4"]
GENDERS = ["M", "F"]
SPEED_RANGES = {
    "slow": (1.20, 1.40),
    "fast": (2.20, 2.40)
}
DISTANCE_RANGE = (10.0, 13.0)
REPS_PER_CONDITION_PER_BLOCK = 2
NUM_BLOCKS = 2

# Constraints
MAX_CONSTRUCTION_ATTEMPTS = 50 # Retries for the smart shuffle (usually succeeds on first try)

# Metadata/Messages
BLOCK_MESSAGES = [
    "Now we are starting the real experiment",
    "This is the second run of the same task."
]

# %%
# ==========================================
#           LOGGING UTILS
# ==========================================

def log(message, console=False):
    """Writes to log file and optionally to console."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted_msg + "\n")

    if console:
        print(message)

def init_log():
    """Initializes the log file."""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"--- Experiment Generation Log Started: {datetime.datetime.now()} ---\n")
        f.write(f"Configuration: Deterministic={USE_DETERMINISTIC_SEED}, Participants={TOTAL_PARTICIPANTS}\n\n")

# ==========================================
#           CORE LOGIC
# ==========================================

def get_unique_conditions(actor_ids, genders, speed_keys):
    conditions = []
    for gender in genders:
        for actor in actor_ids:
            for speed_type in speed_keys:
                conditions.append({
                    "gender": gender,
                    "actor": actor,
                    "speed_type": speed_type
                })
    return conditions

def detect_existing_participants(root_dir, prefix, suffix):
    """
    Scans the root directory for existing participant folders matching the pattern.
    Returns the highest existing participant number.
    """
    if not os.path.exists(root_dir):
        return 0

    highest_num = 0
    try:
        dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except OSError:
        return 0

    for d in dirs:
        if d.startswith(prefix) and (not suffix or d.endswith(suffix)):
            try:
                # Extract number part: sub-XXA -> XX
                # Remove prefix
                temp = d[len(prefix):]
                # Remove suffix if present
                if suffix and temp.endswith(suffix):
                    temp = temp[:-len(suffix)]

                num = int(temp)
                if num > highest_num:
                    highest_num = num
            except ValueError:
                continue

    return highest_num


def recreate_summary_csv(root_dir, csv_file, prefix, suffix):
    """
    Scans all participant folders, reads their block JSON files,
    and reconstructs the all_trials_summary.csv file.
    """
    log(f"Starting CSV Recreation from {root_dir}...", console=True)

    if not os.path.exists(root_dir):
        log(f"Error: Root directory {root_dir} does not exist.", console=True)
        return

    all_data = []

    # helper to find participant folders
    try:
        dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    except OSError:
        log("Error listing directories.", console=True)
        return

    for p_id in dirs:
        if p_id.startswith(prefix) and (not suffix or p_id.endswith(suffix)):
            p_path = os.path.join(root_dir, p_id)
            block_dir = os.path.join(p_path, "ses-001", "block-definitions")

            if not os.path.exists(block_dir):
                continue

            # Read all block files
            for i in range(NUM_BLOCKS):
                filename = FILENAME_TEMPLATE.format(i)
                filepath = os.path.join(block_dir, filename)

                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)

                        # Extract trials
                        # Structure: trialSettings -> valueList -> [ {values: [speed, dist, gender, actor]}, ... ]
                        if "trialSettings" in data and "valueList" in data["trialSettings"]:
                            trial_list = data["trialSettings"]["valueList"]

                            for t_idx, t in enumerate(trial_list):
                                vals = t["values"]
                                # Reconstruct row
                                # vals = [speed, dist, gender, actor]
                                # Note: speed_category is missing in JSON, we have to infer it or leave blank
                                # Using standard infer logic if standard ranges
                                speed_val = float(vals[0])
                                speed_category = "unknown"

                                # Simple inference based on current ranges
                                for k, v in SPEED_RANGES.items():
                                    if v[0] <= speed_val <= v[1]:
                                        speed_category = k
                                        break

                                row = {
                                    "participant_id": p_id,
                                    "block_index": i,
                                    "trial_index": t_idx + 1,
                                    "speed_val": vals[0],
                                    "distance_val": vals[1],
                                    "gender": vals[2],
                                    "actor": vals[3],
                                    "speed_category": speed_category
                                }
                                all_data.append(row)
                    except Exception as e:
                        log(f"Error reading {filepath}: {e}", console=True)

    # Write to CSV
    try:
        fieldnames = ["participant_id", "block_index", "trial_index", "speed_val", "distance_val", "gender", "actor", "speed_category"]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        log(f"SUCCESS: Recreated Summary CSV with {len(all_data)} rows.", console=True)
    except IOError as e:
        log(f"Error writing Recreated CSV: {e}", console=True)

def smart_shuffle_with_constraints(conditions, reps):
    """
    Constructs a trial list trial-by-trial to avoid immediate repetition of (Gender, Actor).
    Uses a Randomized Greedy approach which is much faster and cleaner than 'shuffle & check'.
    """
    # 1. Expand conditions into a pool of all trials needed
    pool = []
    for _ in range(reps):
        pool.extend(conditions)

    total_trials = len(pool)

    for attempt in range(MAX_CONSTRUCTION_ATTEMPTS):
        random.shuffle(pool) # Initial shuffle of the pool
        current_pool = pool[:]
        sequence = []

        success = True

        for _ in range(total_trials):
            # Identify valid candidates from current_pool
            if not sequence:
                 # First trial: everything is valid
                 candidates = [i for i in range(len(current_pool))]
            else:
                last_trial = sequence[-1]
                # Filter candidates based on constraints
                candidates = []
                for i, t in enumerate(current_pool):
                    is_valid = True

                    # 1. Immediate Repetition: Invalid if same Gender AND same Actor
                    if t["gender"] == last_trial["gender"] and t["actor"] == last_trial["actor"]:
                        is_valid = False

                    # 2. Streakiness: Prevent > 3 consecutive same Gender or Speed Type
                    if is_valid and len(sequence) >= 3:
                        last_three = sequence[-3:]
                        # Check Gender Streak
                        if all(x['gender'] == t['gender'] for x in last_three):
                            is_valid = False
                        # Check Speed Streak
                        if all(x['speed_type'] == t['speed_type'] for x in last_three):
                            is_valid = False

                    if is_valid:
                        candidates.append(i)

            if not candidates:
                # We got stuck (no valid moves left)
                success = False
                break

            # Pick a random candidate
            chosen_idx = random.choice(candidates)
            sequence.append(current_pool.pop(chosen_idx))

        if success:
            # Found a valid sequence
            return sequence

    # If we fall through here, we failed multiple times
    error_msg = f"ERROR: Could not generate valid sequence after {MAX_CONSTRUCTION_ATTEMPTS} attempts."
    log(error_msg, console=True)
    return None

def generate_values_for_trials(base_trials):
    """Assigns random numeric values (Speed, Distance) to the selected conditions."""
    final_trials = []
    for spec in base_trials:
        speed_val = random.uniform(*SPEED_RANGES[spec["speed_type"]])
        dist_val = random.uniform(*DISTANCE_RANGE)

        final_trials.append({
            "values": [
                f"{speed_val:.2f}",
                f"{dist_val:.2f}",
                spec["gender"],
                spec["actor"]
            ],
            # Keep metadata for CSV logging
            "meta": {
                "speed_type": spec["speed_type"],
                "gender": spec["gender"], # redundant but convenient
                "actor": spec["actor"]    # redundant but convenient
            }
        })
    return final_trials

def create_json_structure(block_id, intro_msg, trial_list):
    return {
        "type": "task",
        "subType": "tidal",
        "blockId": block_id,
        "settings": [],
        "messages": [
            { "key": "_intro", "value": intro_msg }
        ],
        "trialSettings": {
            "keys": ["speed", "distance", "gender", "actor"],
            "valueList": trial_list
        }
    }

def process_participant(participant_path, conditions):
    participant_id = os.path.basename(participant_path)
    target_dir = os.path.join(participant_path, "ses-001", "block-definitions")

    # Seeding
    if USE_DETERMINISTIC_SEED:
        # Use participant ID to create a unique, reproducible seed
        # We append a salt so it's not identical to other unrelated random calls
        seed_str = f"{participant_id}_tidal_experiment"
        random.seed(seed_str)
        log(f"{participant_id}: Random seed set to '{seed_str}'")
    else:
        # Ensure we are non-deterministic if flag is off
        random.seed()

    # Create directory
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            log(f"{participant_id}: Created directory {target_dir}")
        except OSError as e:
            log(f"{participant_id}: Error creating dir {target_dir} - {e}", console=True)
            return 0

    log(f"{participant_id}: Generating trials...")

    total_trials = 0

    all_trials_data = []

    for i in range(NUM_BLOCKS):
        # 1. Smart Shuffle conditions
        base_trials = smart_shuffle_with_constraints(conditions, REPS_PER_CONDITION_PER_BLOCK)
        if base_trials is None:
            log(f"{participant_id}: Skipping block {i} due to generation failure.")
            continue

        # 2. Assign float values
        trials = generate_values_for_trials(base_trials)
        total_trials += len(trials)

        # Collect data for CSV
        for t_idx, t in enumerate(trials):
            # t["values"] is [speed, dist, gender, actor]
            # t["meta"] has speed_type
            row = {
                "participant_id": participant_id,
                "block_index": i,
                "trial_index": t_idx + 1,
                "speed_val": t["values"][0],
                "distance_val": t["values"][1],
                "gender": t["values"][2],
                "actor": t["values"][3],
                "speed_category": t["meta"]["speed_type"]
            }
            all_trials_data.append(row)

        # 3. Create JSON (strip out metadata for JSON)
        clean_trials_for_json = [{"values": t["values"]} for t in trials]

        block_id_str = FILENAME_TEMPLATE.format(i).replace(".json", "")
        message = BLOCK_MESSAGES[i] if i < len(BLOCK_MESSAGES) else ""
        json_data = create_json_structure(block_id_str, message, clean_trials_for_json)

        # 4. Write
        filename = FILENAME_TEMPLATE.format(i)
        filepath = os.path.join(target_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
            log(f"{participant_id}: Wrote Block {i} ({len(trials)} trials) to {filename}")
        except IOError as e:
            log(f"{participant_id}: Error writing {filename} - {e}", console=True)

    return all_trials_data

def copy_and_update_session_info(participant_path, participant_id, site, stage):
    """
    Copies session-info.json from template and updates it with participant-specific data.
    """
    ses_dir = os.path.join(participant_path, "ses-001")
    session_info_path = os.path.join(ses_dir, "session-info.json")

    try:
        # Read the existing session-info.json
        if os.path.exists(session_info_path):
            with open(session_info_path, 'r') as f:
                session_data = json.load(f)
        else:
            # Create default structure if not found
            session_data = {
                "experiment": "TIDAL",
                "experimenter": "Unknown",
                "participant_details": []
            }

        # Update participant details
        # Clear existing details and add new ones with user-provided values
        session_data["participant_details"] = [
            {"key": "ID", "value": participant_id},
            {"key": "Site", "value": site},
            {"key": "Stage", "value": stage}
        ]

        # Write back to file
        with open(session_info_path, 'w') as f:
            json.dump(session_data, f, indent='\t')

        log(f"Updated session-info.json for {participant_id} (Site: {site}, Stage: {stage})")
        return True
    except Exception as e:
        log(f"Error updating session-info.json for {participant_id}: {e}", console=True)
        return False

def ensure_participants_exist(root_dir, template_name, start_idx, end_idx, prefix, suffix):
    """
    Checks if participant folders exist in the specified range.
    If a folder is missing, it creates it by copying the template folder.
    """
    template_path = os.path.join(root_dir, template_name)
    if not os.path.exists(template_path):
        log(f"CRITICAL: Template folder '{template_name}' not found in {root_dir}. Cannot create new participants.", console=True)
        return

    log(f"Checking structure for participants {start_idx} to {end_idx}...")

    created_count = 0
    for i in range(start_idx, end_idx + 1):
        p_name = f"{prefix}{i:02d}{suffix}"
        p_path = os.path.join(root_dir, p_name)

        if not os.path.exists(p_path):
            log(f"Creating new participant: {p_name}")
            try:
                shutil.copytree(template_path, p_path)
                created_count += 1
            except Exception as e:
                log(f"Failed to copy template to {p_name}: {e}", console=True)

    if created_count > 0:
        log(f"Created {created_count} new participant folders.", console=True)
    else:
        log("All participant folders already exist.", console=False)

def check_overwrite_safety(participant_folders):
    """Checks if ANY target files exist and asks user for confirmation."""
    if not ENABLE_OVERWRITE_CHECK:
        return True

    existing_files_found = 0
    for p_folder in participant_folders:
        for i in range(NUM_BLOCKS):
            filename = FILENAME_TEMPLATE.format(i)
            # Just check typical path
            path = os.path.join(p_folder, "ses-001", "block-definitions", filename)
            if os.path.exists(path):
                existing_files_found += 1
                break # Found one in this folder, essentially enough to trigger warning

    if existing_files_found > 0:
        log(f"Safety Check: Found existing trial files in {existing_files_found} participant folders.", console=False)
        print(f"\n[WARNING] Found existing experimental files in {existing_files_found} folders.")
        print("This operation will OVERWRITE them with new randomized data.")
        user_input = input("Are you sure you want to proceed? (type 'yes' to continue): ").strip().lower()

        if user_input == 'yes':
            log("User confirmed overwrite.")
            return True
        else:
            log("User aborted operation due to overwrite safety check.")
            print("Operation aborted.")
            return False

    return True

# %%
def main():
    global ROOT_PATH, PARTICIPANTS_ROOT, LOG_FILE, TEMPLATE_PARTICIPANT, TOTAL_PARTICIPANTS, PARTICIPANT_PREFIX, PARTICIPANT_SUFFIX, USE_DETERMINISTIC_SEED, ENABLE_OVERWRITE_CHECK

    print("\n" + "="*50)
    print("TIDAL Experiment - Trial Generation")
    print("="*50)

    # Input ROOT_PATH
    root_path_input = input("\nEnter ROOT_PATH (participants directory): ").strip()
    if not root_path_input:
        print("ERROR: ROOT_PATH is required.")
        return

    ROOT_PATH = root_path_input
    PARTICIPANTS_ROOT = ROOT_PATH
    LOG_FILE = os.path.join(os.path.dirname(ROOT_PATH), "generation_log.txt")

    # Initialize log file
    init_log()
    log("Script started.", console=True)

    # Input Participant Generation Settings
    template_input = input("\nEnter TEMPLATE_PARTICIPANT (e.g., sub-01A): ").strip()
    TEMPLATE_PARTICIPANT = template_input if template_input else "sub-01A"

    suffix_input = input("Enter PARTICIPANT_SUFFIX (e.g., B, P): ").strip()
    PARTICIPANT_SUFFIX = suffix_input if suffix_input else "B"

    # Auto-detect existing
    highest_existing = detect_existing_participants(PARTICIPANTS_ROOT, PARTICIPANT_PREFIX, PARTICIPANT_SUFFIX)
    print(f"\n[INFO] Found existing participants up to number: {highest_existing}")

    # Option to Recreate CSV
    if highest_existing > 0:
        recreate_input = input("\n[Repair] Do you want to RECREATE the summary CSV from all existing folders? (y/n): ").strip().lower()
        if recreate_input == 'y':
            csv_file = os.path.join(os.path.dirname(ROOT_PATH), "all_trials_summary.csv")
            recreate_summary_csv(PARTICIPANTS_ROOT, csv_file, PARTICIPANT_PREFIX, PARTICIPANT_SUFFIX)
            return  # Exit after recreation, or remove this return to continue generating more

    new_participants_input = input(f"How many NEW participants do you want to add? (Enter number): ").strip()
    try:
        num_new_participants = int(new_participants_input) if new_participants_input else 0
    except ValueError:
        print("Invalid number, defaulting to 0")
        num_new_participants = 0

    if num_new_participants <= 0:
        print("No new participants requested. Exiting.")
        return

    start_idx = highest_existing + 1
    end_idx = highest_existing + num_new_participants
    print(f"[INFO] Will generate participants from {start_idx} to {end_idx}")

    seed_input = input("\nUse DETERMINISTIC_SEED for reproducibility? (y/n, default n): ").strip().lower()
    USE_DETERMINISTIC_SEED = seed_input == 'y'

    overwrite_input = input("Enable OVERWRITE_CHECK? (y/n, default y): ").strip().lower()
    ENABLE_OVERWRITE_CHECK = overwrite_input != 'n'

    # Input session metadata
    site = input("\nEnter Site name (e.g., MPIB): ").strip()
    stage = input("Enter Stage name (e.g., Pilot): ").strip()

    if not site:
        site = "Unknown"
    if not stage:
        stage = "Unknown"

    log(f"Session metadata set: Site={site}, Stage={stage}")

    # 1. Ensure participant folders exist
    ensure_participants_exist(
        PARTICIPANTS_ROOT,
        TEMPLATE_PARTICIPANT,
        start_idx,
        end_idx,
        PARTICIPANT_PREFIX,
        PARTICIPANT_SUFFIX
    )

    # 2. Define conditions
    conditions = get_unique_conditions(ACTOR_IDS, GENDERS, list(SPEED_RANGES.keys()))

    # 3. Build participant folder paths directly
    participant_folders = [os.path.join(PARTICIPANTS_ROOT, f"{PARTICIPANT_PREFIX}{str(i).zfill(2)}{PARTICIPANT_SUFFIX}") for i in range(start_idx, end_idx + 1)]
    participant_folders = [f for f in participant_folders if os.path.isdir(f)]

    if not participant_folders:
        log("No participant folders found.", console=True)
        return

    # 4. Safety Check
    if not check_overwrite_safety(participant_folders):
        return

    # 5. Update session-info.json for all participants
    for p_folder in participant_folders:
        participant_id = os.path.basename(p_folder)
        copy_and_update_session_info(p_folder, participant_id, site, stage)

    # 6. Process
    print("\nProcessing participants... (See log file for details)")
    grand_total_trials = 0
    full_experiment_data = []

    for p_folder in participant_folders:
        p_data = process_participant(p_folder, conditions)
        if isinstance(p_data, list):
            full_experiment_data.extend(p_data)
            grand_total_trials += len(p_data)
        elif isinstance(p_data, int):
            # Fallback if specific return type changes (safety)
            grand_total_trials += p_data

    # 7. Generate Summary CSV
    csv_file = os.path.join(os.path.dirname(ROOT_PATH), "all_trials_summary.csv")
    if full_experiment_data:
        try:
            file_exists = os.path.exists(csv_file)
            mode = 'a' if file_exists else 'w'

            fieldnames = ["participant_id", "block_index", "trial_index", "speed_val", "distance_val", "gender", "actor", "speed_category"]

            with open(csv_file, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(full_experiment_data)

            log(f"Summary CSV updated: {csv_file} (Appended {len(full_experiment_data)} rows)", console=True)
        except IOError as e:
            log(f"Error writing CSV summary: {e}", console=True)

    log(f"Generation Complete. Total trials: {grand_total_trials}", console=True)
    print(f"Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
