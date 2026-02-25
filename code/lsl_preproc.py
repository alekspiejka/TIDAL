"""
LSL Raw Data Extractor & BIDS Converter
=======================================

**Description:**
    This script preprocesses and extracts individual data streams from raw LSL (Lab Streaming Layer) output files (.xdf).
    It converts complex, multi-stream XDF recordings into the BIDS (Brain Imaging Data Structure) standard,
    ensuring that each data source (e.g., ECG, EEG, Eye-Tracking, Motion) is separated into its own file with correct metadata.

**Key Features:**
    1.  **Stream Extraction**: Reads complex XDF files and isolates specific streams (Physio, Markers, Start/Stop events).
    2.  **Preprocessing**: Calculates effective sampling rates and synchronizes timestamps to the session start.
    3.  **BIDS Conversion**:
        -   Saves continuous data as compressed TSV files (`.tsv.gz`).
        -   Generates JSON sidecars with channel-specific metadata (Units, Type, Description).
        -   Organizes files into `sub-XXX/ses-XXX/beh/` directory structures.
    4.  **Batch Processing**: Can process entire directories of XDF files automatically.

**Inputs:**
    -   Raw XDF files (`.xdf`) containing multiple LSL streams.
    -   Structure: `data/sourcedata/sub-XXX/ses-XXX/beh/*.xdf` (preferred) or flat directory.

**Outputs:**
    -   **Physio Data**: `sub-XXX_ses-XXX_task-TIDAL_recording-streamName_physio.tsv.gz` (and `.json`)
    -   **Events**: `sub-XXX_ses-XXX_task-TIDAL_events.tsv` (and `.json`)
    -   **Scans Info**: `sub-XXX_ses-XXX_scans.tsv` listing all generated files.
    -   **Dataset Description**: `dataset_description.json` with summary statistics.

**Usage:**
    1.  **Interactive Mode** (Wizard):
        `python code/lsl_preproc.py`

    2.  **Single File CLI**:
        `python code/lsl_preproc.py --file "path/to/sub-01_ses-001.xdf" --root "data"`

    3.  **Batch Mode CLI**:
        `python code/lsl_preproc.py --sourcedata "data/sourcedata" --root "data"`

"""

import pyxdf
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import argparse
import re
# %%
# ================= HELPER FUNCTIONS =================

def parse_path_info(file_path):
    """
    Robust extraction of BIDS entities from file path.
    Prioritizes folder names, falls back to filename.
    """
    parts = file_path.parts
    sub = None
    ses = None
    run = None
    task = "TEST" # Default task if not found

    # 1. Check folder structure
    for p in parts:
        if p.startswith("sub-"):
            val = p.replace("sub-", "")
            sub = val.split("_")[0]
        if p.startswith("ses-"):
            val = p.replace("ses-", "")
            ses = val.split("_")[0]
        if p.startswith("run-"):
            val = p.replace("run-", "")
            run = val.split("_")[0]

    # 2. Fallback: Check filename
    filename = file_path.name
    if not sub:
        match = re.search(r"sub-([a-zA-Z0-9]+)", filename)
        if match: sub = match.group(1)
    if not ses:
        match = re.search(r"ses-([a-zA-Z0-9]+)", filename)
        if match: ses = match.group(1)
    if not run:
        match = re.search(r"run-([a-zA-Z0-9]+)", filename)
        if match: run = match.group(1)

    # 3. Check for Task in filename
    match_task = re.search(r"task-([a-zA-Z0-9]+)", filename)
    if match_task: task = match_task.group(1)

    # Defaults
    if not sub: sub = "01A"
    if not ses: ses = "001"
    if not run: run = "001"

    return sub, ses, run, task


def sanitize_stream_name(name):
    """
    Convert a stream name to a BIDS-safe identifier.
    BIDS labels must be alphanumeric. We normalize by lowercasing and removing special chars.
    Example: "BrainVision RDA" -> "brainvisionrda"
    """
    safe = name.lower()
    # Remove all non-alphanumeric characters (including underscores/dots)
    # This ensures the resulting string is a valid BIDS entity label.
    safe = re.sub(r'[^a-z0-9]', '', safe)
    return safe if safe else "unknown"


# =============================================================================
# CHANNEL METADATA LOOKUP
# =============================================================================
# Since the XDF streams from Unity/EDIA often omit unit information, we define
# correct units, descriptions, and BIDS types based on the channel name.
# Keys are lowercase channel names (as they appear after sanitize_stream_name).

CHANNEL_METADATA = {
    # --- Unity.Transform streams (HMD, Left, Right controllers) ---
    "posx":  {"Units": "m",  "Type": "POS",  "Description": "Position X (Unity world coordinates)"},
    "posy":  {"Units": "m",  "Type": "POS",  "Description": "Position Y (Unity world coordinates)"},
    "posz":  {"Units": "m",  "Type": "POS",  "Description": "Position Z (Unity world coordinates)"},
    "rotx":  {"Units": "n/a", "Type": "ORNT", "Description": "Rotation quaternion X component"},
    "roty":  {"Units": "n/a", "Type": "ORNT", "Description": "Rotation quaternion Y component"},
    "rotz":  {"Units": "n/a", "Type": "ORNT", "Description": "Rotation quaternion Z component"},
    "rotw":  {"Units": "n/a", "Type": "ORNT", "Description": "Rotation quaternion W component"},

    # --- EDIA Eye.Data stream (eye-tracking) ---
    "pitch":          {"Units": "degrees", "Type": "EYETRACK", "Description": "Eye gaze pitch angle"},
    "yaw":            {"Units": "degrees", "Type": "EYETRACK", "Description": "Eye gaze yaw angle"},
    "roll":           {"Units": "degrees", "Type": "EYETRACK", "Description": "Eye gaze roll angle"},
    "pupildiameter":  {"Units": "mm",      "Type": "EYETRACK", "Description": "Pupil diameter"},
    "openness":       {"Units": "a.u.",    "Type": "EYETRACK", "Description": "Eye openness (0 = closed, 1 = fully open)"},
    "confidence":     {"Units": "a.u.",    "Type": "EYETRACK", "Description": "Eye-tracking confidence (0-1)"},
    "timestampet":    {"Units": "s",       "Type": "MISC",     "Description": "Eye-tracker device timestamp"},

    # --- PolarBand ECG ---
    # (PolarBand already gets 'microvolts' from XDF metadata, but included as fallback)
    "ecg":            {"Units": "microvolts", "Type": "ECG",  "Description": "Electrocardiogram signal"},
}


def validate_bids_output(tsv_file, json_file=None):
    """
    Validates a generated BIDS file.
    - Checks if file exists and has content.
    - If .gz, checks if it can be decompressed and read.
    - If json_file provided, checks if column count matches.
    """
    try:
        if not tsv_file.exists():
            print(f"    [ERROR] File missing: {tsv_file.name}")
            return False

        if tsv_file.stat().st_size == 0:
            print(f"    [ERROR] File empty: {tsv_file.name}")
            return False

        # Try reading the file
        if tsv_file.suffix == '.gz':
            df = pd.read_csv(tsv_file, sep='\t', compression='gzip', header=None)
        else:
            df = pd.read_csv(tsv_file, sep='\t')

        if df.empty:
             print(f"    [WARN] Dataframe empty: {tsv_file.name}")
             return True # Not necessarily an error, but worth noting

        # Validate against JSON metadata if provided (for physio files)
        if json_file and json_file.exists():
            with open(json_file, 'r') as f:
                meta = json.load(f)

            # Physio files usually have no header in the file, but columns in JSON
            if "Columns" in meta:
                expected_cols = len(meta["Columns"])
                actual_cols = df.shape[1]
                if expected_cols != actual_cols:
                    print(f"    [FAIL] Column mismatch {tsv_file.name}: JSON says {expected_cols}, Found {actual_cols}")
                    return False

        print(f"    [VALIDATED] {tsv_file.name} (Shape: {df.shape})")
        return True

    except Exception as e:
        print(f"    [CRITICAL] Validation failed for {tsv_file.name}: {e}")
        return False


def process_file(xdf_file, bids_root, subject_id, session_id, run_id, task_name):
    """
    Core conversion logic for a single XDF file.
    Each continuous stream is saved as a SEPARATE physio file with its own sampling rate.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing: {xdf_file.name}")
    print(f"  -> Sub: {subject_id}, Ses: {session_id}, Run: {run_id}, Task: {task_name}")

    # Track Validation Results
    validation_passed = 0
    validation_failed = 0

    if not xdf_file.exists():
        print(f"[ERROR] Input file not found: {xdf_file}")
        return

    # Build Output Directory
    if session_id:
        subject_dir = bids_root / f"sub-{subject_id}" / f"ses-{session_id}" / "beh"
        base_fname = f"sub-{subject_id}_ses-{session_id}_task-{task_name}"
    else:
        subject_dir = bids_root / f"sub-{subject_id}" / "beh"
        base_fname = f"sub-{subject_id}_task-{task_name}"

    if run_id:
        base_fname += f"_run-{run_id}"

    subject_dir.mkdir(parents=True, exist_ok=True)

    # Load XDF
    try:
        data, header = pyxdf.load_xdf(xdf_file)
    except Exception as e:
        print(f"[ERROR] Failed to load XDF: {e}")
        return

    # Prepare Data Containers
    events_data = []
    generated_files = []

    # Track stream names to handle duplicates
    stream_name_counts = {}

    # Process Streams
    for stream in data:
        stream_name = stream['info']['name'][0]
        stream_type = stream['info']['type'][0]
        sampling_rate = float(stream['info']['nominal_srate'][0]) if stream['info']['nominal_srate'] else 0

        # Generate unique stream identifier
        safe_name = sanitize_stream_name(stream_name)
        if safe_name in stream_name_counts:
            stream_name_counts[safe_name] += 1
            stream_id = f"{safe_name}{stream_name_counts[safe_name]}"
        else:
            stream_name_counts[safe_name] = 1
            stream_id = safe_name

        if stream_type == 'Markers' or sampling_rate == 0:
            # Event Markers
            timestamps = stream['time_stamps']
            markers = [marker[0] if isinstance(marker, (list, tuple)) else str(marker)
                       for marker in stream['time_series']]

            for ts, marker in zip(timestamps, markers):
                events_data.append({
                    'onset': ts,
                    'duration': 0,
                    'trial_type': marker
                })
        else:
            # === CONTINUOUS PHYSIO: Save EACH stream separately ===
            time_series = stream['time_series']
            timestamps = stream['time_stamps']

            if len(timestamps) == 0:
                print(f"    [SKIP] {stream_name}: No data")
                continue

            start_time = float(timestamps[0])

            # Calculate Effective Sampling Rate
            # Nominal sampling rate (what the device claims) often differs from the actual rate.
            # We compute effective rate = (N_samples - 1) / Duration
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                effective_srate = (len(timestamps) - 1) / duration
            else:
                effective_srate = sampling_rate # Fallback for single-sample streams

            # Helper for XML
            def get_xml_val(node, path):
                try:
                    curr = node
                    for k in path: curr = curr[k][0]
                    return curr
                except: return None

            desc = stream['info']['desc'][0] if stream['info']['desc'] else {}
            manufacturer = get_xml_val(desc, ['hardware', 'manufacturer'])
            model = get_xml_val(desc, ['hardware', 'model'])
            serial = get_xml_val(desc, ['hardware', 'serial_number'])
            source_id = stream['info']['source_id'][0] if stream['info']['source_id'] else "unknown"

            # Channel Units
            channel_units = []
            channel_names = []
            try:
                chans = desc['channels'][0]['channel']
                channel_units = [get_xml_val(c, ['unit']) for c in chans]
                channel_names = [get_xml_val(c, ['label']) for c in chans]
            except: pass

            # Determine column names for this stream
            physio_columns = []
            if len(time_series.shape) == 1:
                # Single channel
                physio_columns = [stream_id]
                physio_df = pd.DataFrame({stream_id: time_series})
            else:
                # Multi-channel
                for i in range(time_series.shape[1]):
                    if i < len(channel_names) and channel_names[i]:
                        col_name = sanitize_stream_name(channel_names[i])
                    else:
                        col_name = f"{stream_id}_ch{i}"
                    physio_columns.append(col_name)
                physio_df = pd.DataFrame(time_series, columns=physio_columns)

            # === Write Physio TSV.GZ (one per stream) ===
            physio_tsv_file = subject_dir / f"{base_fname}_recording-{stream_id}_physio.tsv.gz"
            physio_df.to_csv(physio_tsv_file, sep='\t', index=False, header=False, compression='gzip', na_rep='n/a')
            generated_files.append(physio_tsv_file)
            print(f"    [SAVED] {physio_tsv_file.name} ({len(timestamps)} samples, Nominal: {sampling_rate:.2f}Hz, Effective: {effective_srate:.2f}Hz)")

            # === Write Physio JSON (one per stream) ===
            physio_json_file = subject_dir / f"{base_fname}_recording-{stream_id}_physio.json"
            json_content = {
                'SamplingFrequency': sampling_rate,
                'NominalSamplingFrequency': sampling_rate,
                'EffectiveSamplingFrequency': effective_srate,
                'StartTime': start_time,
                'Columns': physio_columns,
                'SourceStreamName': stream_name,
                'SourceStreamType': stream_type,
                'SourceID': source_id
            }
            if manufacturer: json_content['Manufacturer'] = manufacturer
            if model: json_content['ManufacturersModelName'] = model
            if serial: json_content['DeviceSerialNumber'] = serial

            # Add per-column metadata using CHANNEL_METADATA lookup
            for idx, col_name in enumerate(physio_columns):
                # 1. Try XDF-embedded unit first
                xdf_unit = channel_units[idx] if idx < len(channel_units) and channel_units[idx] else None

                # 2. Look up known channel metadata by name
                col_lower = col_name.lower()
                known = CHANNEL_METADATA.get(col_lower, {})

                # 3. Resolve unit: XDF metadata > lookup table > n/a
                unit = xdf_unit if xdf_unit else known.get('Units', 'n/a')

                # 4. Resolve BIDS type: lookup table > heuristic > MISC
                bids_type = known.get('Type', None)
                if not bids_type:
                    # Heuristic fallback for unknown channels
                    bids_type = "MISC"
                    if "ecg" in col_lower or "cardiac" in stream_type.lower(): bids_type = "ECG"
                    elif "ppg" in col_lower or "pulse" in col_lower: bids_type = "PPG"
                    elif "resp" in col_lower or "breathing" in col_lower: bids_type = "RESP"
                    elif "eda" in col_lower: bids_type = "EDA"
                    elif "eye" in col_lower or "gaze" in col_lower or "pupil" in col_lower: bids_type = "EYETRACK"

                # 5. Resolve description: lookup table > generic
                description = known.get('Description', f'Channel {idx} from {stream_name}')

                json_content[col_name] = {
                    'Description': description,
                    'Units': unit,
                    'Type': bids_type,
                    'Status': 'good'
                }

            with open(physio_json_file, 'w') as f:
                json.dump(json_content, f, indent=2)
            generated_files.append(physio_json_file)

            # === Validate Physio ===
            if validate_bids_output(physio_tsv_file, physio_json_file):
                validation_passed += 1
            else:
                validation_failed += 1

    # Write Events
    if events_data:
        events_df = pd.DataFrame(events_data)
        # Fix empty cells by replacing empty strings/NaN with 'n/a' (via na_rep or explicit fill)
        events_df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        events_tsv_file = subject_dir / f"{base_fname}_events.tsv"
        events_df.to_csv(events_tsv_file, sep='\t', index=False, na_rep='n/a')
        generated_files.append(events_tsv_file)
        print(f"    [SAVED] {events_tsv_file.name} ({len(events_data)} events)")

        events_json_file = subject_dir / f"{base_fname}_events.json"
        events_json = {
            'onset': {'Description': 'Onset time of event in seconds'},
            'duration': {'Description': 'Duration of event in seconds'},
            'trial_type': {'Description': 'Type of event or marker'}
        }
        with open(events_json_file, 'w') as f:
            json.dump(events_json, f, indent=2)
        generated_files.append(events_json_file)

    # Write Scans (only data files, not JSON sidecars)
    if session_id:
        scans_dir = bids_root / f"sub-{subject_id}" / f"ses-{session_id}"
    else:
        scans_dir = bids_root / f"sub-{subject_id}"

    scans_dir.mkdir(parents=True, exist_ok=True)
    scans_file = scans_dir / f"sub-{subject_id}_ses-{session_id}_scans.tsv" if session_id else scans_dir / f"sub-{subject_id}_scans.tsv"

    # Normalize acq_time to BIDS-compliant ISO 8601 (YYYY-MM-DDTHH:MM:SS, no timezone)
    try:
        raw_acq_time = header['info']['datetime'][0]
        # Parse any ISO 8601 format and output without timezone
        acq_time = datetime.fromisoformat(raw_acq_time).strftime('%Y-%m-%dT%H:%M:%S')
    except Exception:
        acq_time = datetime.fromtimestamp(os.path.getmtime(xdf_file)).strftime('%Y-%m-%dT%H:%M:%S')

    # Filter to data files only (exclude JSON sidecars)
    data_files = [f for f in generated_files if f.suffix != '.json']

    # Check if scans file exists (append mode support)
    existing_scans = []
    if scans_file.exists():
        try:
            existing_df = pd.read_csv(scans_file, sep='\t')
            existing_scans = existing_df.to_dict('records')
        except: pass

    for fpath in data_files:
        try:
            rel_path = fpath.relative_to(scans_dir).as_posix()
            entry = {"filename": rel_path, "acq_time": acq_time}
            # Avoid duplicates
            if not any(d['filename'] == rel_path for d in existing_scans):
                existing_scans.append(entry)
        except ValueError: pass

    if existing_scans:
        pd.DataFrame(existing_scans).to_csv(scans_file, sep='\t', index=False, na_rep='n/a')

    print(f"  -> [COMPLETE] Generated {len(generated_files)} files.")
    if validation_failed == 0 and validation_passed > 0:
        print(f"  -> [SUCCESS] All {validation_passed} physio files PASSED validation.")
    elif validation_failed > 0:
        print(f"  -> [WARNING] {validation_failed} files FAILED validation (Passed: {validation_passed}).")
# %%
# ================= MAIN EXECUTION =================

def main():
    parser = argparse.ArgumentParser(description="Universal XDF to BIDS Converter (Batch & Single Mode)")
    parser.add_argument("--file", required=False, help="Processing MODE: Specific Input XDF file. If OMITTED, runs in BATCH mode.")
    parser.add_argument("--sourcedata", required=False, help="Batch MODE: Input Directory containing XDF files (e.g., source/data)")
    parser.add_argument("--root", required=False, help="Output BIDS directory (Results folder)")
    parser.add_argument("--participants", nargs='+', help="Filter: List of subject IDs to process (e.g., 01A 02B). If OMITTED, processes ALL.")

    # Overrides for Single File Mode
    parser.add_argument("--subject", help="Override Subject ID")
    parser.add_argument("--session", help="Override Session ID")
    parser.add_argument("--task", help="Override Task Name")
    parser.add_argument("--run", help="Override Run ID")

    args = parser.parse_args()

    # === INTERACTIVE MODE ===
    # If key arguments are missing, switch to interactive input
    if not args.root and not args.file:
        print("\n=== BIDS XDF Converter V2 (Interactive Mode) ===")
        print("No arguments detected. Please answer the following:\n")

        # Define Defaults - Script-Relative
        # Assumes structure: repo_root/code/this_script.py -> repo_root/data
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent

        default_processed = project_root / "data"
        default_sourcedata = project_root / "data" / "sourcedata"

        # 1. Select Mode
        print("Select Mode:")
        print("  1. Process a SINGLE XDF file")
        print("  2. BATCH Process entire folder")
        while True:
            mode_sel = input("Enter choice (1 or 2): ").strip()
            if mode_sel in ['1', '2']:
                break
            print("  [!] Invalid choice.")

        # 2. Configure Mode
        if mode_sel == '1':
            # Single File Info
            while True:
                file_input = input("\nEnter path to XDF file: ").strip("'\" ")
                if file_input and Path(file_input).exists() and file_input.endswith('.xdf'):
                    args.file = file_input
                    break
                print("  [!] Error: Invalid file path or not an .xdf file.")

            # Output Root: Always use default (root folder)
            args.root = str(default_processed)
            print(f"\n[INFO] Output Directory: {default_processed}")

            # Optional Overrides
            print("\n(Optional) Press ENTER to auto-detect from filename, or type value to override:")
            sub_in = input("  Override Subject ID (e.g., 01P): ").strip()
            if sub_in: args.subject = sub_in

            ses_in = input("  Override Session ID (e.g., 001): ").strip()
            if ses_in: args.session = ses_in

            task_in = input("  Override Task Name (e.g., TIDAL): ").strip()
            if task_in: args.task = task_in

            run_in = input("  Override Run ID (e.g., 001): ").strip()
            if run_in: args.run = run_in

        else:
            # Batch Info
            # 1. Input Source (Auto-Detect)
            if default_sourcedata.exists():
                print(f"\n[INFO] Auto-detected Input Directory: {default_sourcedata}")
                args.sourcedata = str(default_sourcedata)
            else:
                while True:
                    src_input = input("\nEnter Input Sourcedata Directory (Where .xdf files are): ").strip("'\" ")
                    if src_input and Path(src_input).exists():
                        args.sourcedata = src_input
                        break
                    print("  [!] Error: Directory does not exist.")

            # Output Root: Always use default (root folder)
            args.root = str(default_processed)
            print(f"[INFO] Output Directory: {default_processed}")

            print("\n(Optional) Enter Subject IDs to filter (space separated, e.g., 01A 02B).")
            print("Press ENTER to process ALL subjects found in root.")
            part_in = input("Participants: ").strip()
            if part_in:
                args.participants = part_in.split()

    # Apply Cleanups
    if not args.root:
        print("[ERROR] Output Root directory is required.")
        return

    cleaned_root = args.root.strip("'\" ")
    bids_root = Path(cleaned_root).resolve()
    # Create output root if it doesn't exist
    bids_root.mkdir(parents=True, exist_ok=True)

    print(f"\nDEBUG: Output Path: '{bids_root}'")

    # === SINGLE FILE MODE ===
    if args.file:
        xdf_file = Path(args.file)
        print(f"--- SINGLE FILE MODE ---")

        # Determine entities (Arg override -> Parsing -> Default)
        p_sub, p_ses, p_run, p_task = parse_path_info(xdf_file)

        sub = args.subject if args.subject else p_sub
        ses = args.session if args.session else p_ses
        run = args.run if args.run else p_run
        task = args.task if args.task else p_task

        process_file(xdf_file, bids_root, sub, ses, run, task)
# %%
    # === BATCH MODE ===
    else:
        print(f"--- BATCH MODE ---")

        # Fallback: If CLI used old --root for input, we might need new logic.
        # But here we enforce explicit sourcedata path for scanning.
        sourcedata_path = None
        if args.sourcedata:
            sourcedata_path = Path(args.sourcedata.strip("'\" ")).resolve()

        if not sourcedata_path or not sourcedata_path.exists():
            # Try to guess sourcedata based on typical structure if not provided
            # e.g. if root is "data/processed", maybe "data/sourcedata"?
            possible_source = bids_root.parent / "sourcedata"
            if possible_source.exists():
                print(f"[INFO] Auto-detected sourcedata at: {possible_source}")
                sourcedata_path = possible_source
            else:
                # If CLI usage missed --sourcedata, check if user mistakenly pointed --root to input
                # This is tricky. Let's error and ask for input.
                print(f"[ERROR] Input Sourcedata directory not found. Please specify --sourcedata.")
                return

        print(f"Scanning for XDF files in: {sourcedata_path}")

        participants_filter = args.participants if args.participants else []
        if participants_filter:
            print(f"Filtering for participants: {participants_filter}")
        else:
            print(f"Processing ALL participants found.")

        # Scan for XDFs recursively in Input Directory
        xdf_files = list(sourcedata_path.rglob("*.xdf"))

        print(f"Found {len(xdf_files)} XDF files to process.")

        for xdf_file in xdf_files:
            # Parse info
            sub, ses, run, task = parse_path_info(xdf_file)

            # Check Filter
            if participants_filter and sub not in participants_filter:
                continue

            # Process (Saving to bids_root)
            process_file(xdf_file, bids_root, sub, ses, run, task)

        # === METADATA GENERATION (Batch Mode Only) ===
        print("\n--- GENERATING METADATA ---")

        # 1. Dataset Description
        desc_file = bids_root / "dataset_description.json"

        # Calculate Stats First
        subjects_found = set()
        modalities_found = set()

        for sub_dir in bids_root.glob("sub-*"):
             if sub_dir.is_dir():
                 s_id = sub_dir.name.replace("sub-", "").split("_")[0]
                 subjects_found.add(s_id)

        for p in bids_root.rglob("*"):
            if "eeg" in p.name: modalities_found.add("EEG")
            if "physio" in p.name: modalities_found.add("Physiology")
            if "events" in p.name: modalities_found.add("Events")

        summary_stats = {
            "ParticipantCount": len(subjects_found),
            "Modalities": sorted(list(modalities_found))
        }

        # Load existing or create new
        if desc_file.exists():
            try:
                with open(desc_file, 'r') as f:
                    desc = json.load(f)
            except:
                desc = {}
        else:
            desc = {
                "Name": "BIDS Dataset",
                "BIDSVersion": "1.10.1",
                "DatasetType": "raw",
                "License": "MIT",
                "Authors": ["Aleksandra Piejka", "Christian Abele"]
            }

        # Update with new stats
        desc["DatasetSummary"] = summary_stats

        # Ensure base fields if missing (fallback)
        if "Name" not in desc: desc["Name"] = "BIDS Dataset"
        if "BIDSVersion" not in desc: desc["BIDSVersion"] = "1.10.1"

        with open(desc_file, 'w') as f:
            json.dump(desc, f, indent=4)
        print(f"[UPDATED] dataset_description.json with Summary Stats")

if __name__ == "__main__":
    main()
