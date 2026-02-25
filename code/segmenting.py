"""
=================================================================================================
Script: Physiological Data Segmentation & Feature Extraction
=================================================================================================

Description:
    This script processes physiological and behavioral data for the TIDAL project.
    It synchronizes data streams from different sources (Unity, LSL, Polar, Eyetracker)
    to a common timeline and extracts specific time windows for analysis.

Inputs:
    1. ECG Peaks (.json):
       - Source: derivatives/ecg-preproc/.../ecg-preproc.json
       - Content: R-peak timestamps (Manual or Auto corrected).

    2. Heart Rate Data (.tsv.gz):
       - Source: derived from Polar band data.
       - Content: Beat-to-beat HR values.

    3. Eye-Tracking Data (.tsv.gz):
       - Source: data/sub-XXX/.../recording-ediaeyecentercenter_physio.tsv.gz
       - Content: Pupil diameter, eye openness, gaze confidence.

    4. Head Movement Data (.tsv.gz):
       - Source: data/sub-XXX/.../recording-hmd_physio.tsv.gz
       - Content: Position (x,y,z) and Rotation (x,y,z) of the HMD.

    5. Event Markers (.tsv):
       - Source: LSL event stream.
       - Used for synchronizing Unity events with real-world time.

    6. Unity Execution Order (.csv):
       - Source: sourcedata/unity-edia/.../executionOrder.csv
       - Content: Log of all events and states in the VR environment.

Actions:
    1. Time Alignment:
       - Synchronizes Unity execution logs with LSL timestamps using 'AgentStopped' events as anchors.
       - Aligns physiological data series (ECG, Eyetracking, HMD) to absolute timestamps using metadata start times.

    2. Feature Extraction:
       - Computes Heart Rate Variability (HRV) metrics (LF power, HF power) using Continuous Wavelet Transform (CWT).
       - Calculates composite Head Movement and Rotation vectors from raw coordinate data.

    3. Segmentation:
       - Extracts data for specific experimental Blocks (Block 1, Block 2).
       - Extracts data for individual Trials (filtering out practice and break trials).
       - Separates data into distinct phases relative to the 'AgentStopped' event:
         * Baseline: 3s before agent appears.
         * Approach: From agent appearance to stop.
         * Recovery: 3s after agent stops.

    4. Alignment Quality Check:
       - Runs a pre-alignment check (raw LSL vs Unity input consistency) and a
         post-alignment check (structural integrity, trial counts, delta preservation)
         via ``time_alignment_check.py``. Results are printed as a compact summary and
         saved as an ``alignment-report.txt`` alongside the segmented output.

Outputs:
    - JSON dictionary (gzipped) per Run containing segmented lists for:
      * Baseline, Approach, Recovery phases
      * Trials
      * Blocks
    - Each segment contains:
      * Metadata (Subject, Session, Run, Event/Phase Name)
      * Timestamps (Window Start, Window End, Event Time)
      * Physiological Data (ECG Peaks, HR, Pupil Diameter, Head Movement)
      * Calculated Features (LF/HF Power, HRV)

=================================================================================================
"""
#%%
# Import necessary libraries
import pandas as pd
import os
import numpy as np
from pathlib import Path
import json
from cwt import Cwt
from cwt import Cwt_fixed
import gzip
from time_alignment_check import verify_participant, validate_alignment, print_alignment_summary, save_alignment_report

#%%
# =================================================================================================
# Global Configuration & Path Setup
# =================================================================================================
# Set up the directory structure relative to the project root.
root = Path(__file__).parent.parent  # Move up two levels to reach the project root
data_dir = root / "data"             # Main data directory containing all subfolders
sourcedata_dir = data_dir / "sourcedata"
derivatives_dir = data_dir / "derivatives"

# Specific subdirectories for inputs
ecg_preproc_dir = derivatives_dir / "ecg-preproc"  # Location of preprocessed ECG peaks
unity_dir = sourcedata_dir / "unity-edia"          # Location of Unity execution logs

# ---------------------------------------------------------
# Subject & Session Definition
# ---------------------------------------------------------
# Define the list of subjects to process.
# pfxs contains the subject identifiers (e.g., "01P").
pfxs = ["01P", "02P", "03P", "04P", "05P", "06P", "07P", "08P", "09P", "10P", "11P", "12P", "13P", "14P", "15P", "16P", "17P", "18P"]
#pfxs = ["01P"]
subjects = [f"sub-{pfx}" for pfx in pfxs]

# Define the sessions to process (currently only session 001).
sessions = ["ses-001"]

#%% Define necessary functions for the processing (e.g., to adjust timing of execution order, to define windows, to extract data in the windows, etc.)

def execution_order_time_alignment(execution_order_path, events_path):
    """
    Synchronizes the Unity execution order events with real-world LSL timestamps.

    Logic:
    1. Matches 'AgentStopped' events from Unity logs with 'AgentStoppedWalking' events from LSL markers.
    2. Assigns the known LSL timestamps to these anchor points.
    3. Extrapolates timestamps for all other events (before and after anchors) based on
       the relative time differences recorded in the Unity logs.
    """
    # Recreating timestamps for the events in execution_order_data based on the timestamp of the AgentStopped synchronised with AgentStoppedWalking event from events_data that precedes them

    execution_order_data = pd.read_csv(execution_order_path)
    events_data = pd.read_csv(events_path, sep='\t')

    # Get the matching indices
    agent_stopped_mask = execution_order_data['executed'] == 'AgentStopped'
    onset_values = events_data[events_data['trial_type'] == 'AgentStoppedWalking']['onset'].values
    execution_order_data.loc[agent_stopped_mask, 'lsl_timestamp'] = onset_values

    # Recreating timestamps after each instance of "AgentStopped" in the execution_order_data using the preceeding lsl_timestamp
    for idx in range(len(execution_order_data)):
        if execution_order_data.at[idx, 'executed'] == 'AgentStopped':
            # Update the lsl_timestamp for the following events until the next "AgentStopped" event
            for idx2 in range(idx + 1, len(execution_order_data)):
                if execution_order_data.at[idx2, 'executed'] == 'AgentStopped':
                    break
                time_diff = execution_order_data.at[idx2, 'timestamp'] - execution_order_data.at[idx, 'timestamp']
                execution_order_data.at[idx2, 'lsl_timestamp'] = execution_order_data.at[idx, 'lsl_timestamp'] + time_diff

    # Recreating timestamps before each instance of "AgentStopped" in the execution_order_data using the following lsl_timestamp
    for idx in range(len(execution_order_data) - 1, -1, -1):
        if execution_order_data.at[idx, 'executed'] == 'AgentStopped':
            # Update the lsl_timestamp for the preceding events until the previous "AgentStopped" event
            for idx2 in range(idx - 1, -1, -1):
                if execution_order_data.at[idx2, 'executed'] == 'AgentStopped':
                    break
                time_diff = execution_order_data.at[idx2, 'timestamp'] - execution_order_data.at[idx, 'timestamp']
                execution_order_data.at[idx2, 'lsl_timestamp'] = execution_order_data.at[idx, 'lsl_timestamp'] + time_diff

    # Return the adjusted execution order data
    return execution_order_data

def rr_time_alignment(ecg_peaks_path, ecg_meta_path):
    """
    Loads ECG peak data and calculates RR intervals aligned with absolute timestamps.

    Processing Steps:
    1. Loads ECG peaks from JSON.
    2. Uses metadata 'StartTime' and 'EffectiveSamplingFrequency' to convert sample indices to timestamps.
    3. Selects the best available peak data (Manual Correction > Auto Correction > Uncorrected).
    4. Calculates RR intervals (time difference between consecutive peaks).
    5. Also includes RR intervals calculated using the BBSIG Pipeline as reference.
    """

    # Load the ECG peaks data from the .json file
    with open(ecg_peaks_path, 'r') as f:
        ecg_peaks_dic = json.load(f)

    # Load the ECG meta data from the .json file
    with open(ecg_meta_path, 'r') as f:
        ecg_meta_dic = json.load(f)

    # Extract Start_Time and EffectiveSamplingFrequency
    start_time = ecg_meta_dic.get('StartTime')
    sampling_rate = ecg_meta_dic.get('EffectiveSamplingFrequency')

    # Extract ECG peaks and RR intervals based on the available keys in the ecg_peaks_dic
    # Note: Check within the nested 'rpeaks' dictionary, not at the top level
    if 'ECG_R_Peaks_ManualCorr' in ecg_peaks_dic.get('rpeaks', {}).keys():
        ecg_peaks = ecg_peaks_dic.get('rpeaks').get('ECG_R_Peaks_ManualCorr')
        rr_bbsig = ecg_peaks_dic.get('rr_s').get('RR_s_ManualCorr')
    elif 'ECG_R_Peaks_AutoCorr' in ecg_peaks_dic.get('rpeaks', {}).keys():
        ecg_peaks = ecg_peaks_dic.get('rpeaks').get('ECG_R_Peaks_AutoCorr')
        rr_bbsig = ecg_peaks_dic.get('rr_s').get('RR_s_AutoCorr')
    else:
        # No corrected peaks available - ECG was not preprocessed
        print("ECG was not preprocessed - no auto-corrected or manually corrected peaks available")
        return pd.DataFrame()

    # Create timestamps for the ECG peaks based on the Start_Time and EffectiveSamplingFrequency and their sample indices
    ecg_peaks_timestamps = [start_time + (peak / sampling_rate) for peak in ecg_peaks]

    # Create RR intervals based on the timestamps of the ECG peaks
    rr_intervals = np.diff(ecg_peaks_timestamps)


    # Create a DataFrame to store the ECG peaks timestamps and RR intervals
    ecg_peaks_df = pd.DataFrame({
        'timestamp': ecg_peaks_timestamps[:-1],  # Exclude the last peak since it doesn't have a subsequent peak to calculate the RR interval
        'ecg_peak': ecg_peaks[:-1], # Exclude the last peak for the same reason as above
        'rr_calc': rr_intervals, # RR interval is the time difference between consecutive peaks
        'rr_bbsig': rr_bbsig # RR interval calculated using the amazing BBSIG pipeline for cardiac preprocessing, which is more robust to noise and artefacts than the simple calculation of RR intervals based on the timestamps of the ECG peaks

    })

    return ecg_peaks_df

def hr_data_time_alignment(hr_data_path, hr_meta_path, sampling_rate=1000):
    """
    Aligns Heart Rate (HR) data series with absolute timestamps.
    Using the start time from metadata, converts sample indices into LSL timestamps.
    """
    # Creating timestamps for the HR data based on the timestamp of the Start_Time and EffectiveSamplingFrequency saved in the .json file for the corresponding data
    if os.path.exists(hr_data_path) and os.path.exists(hr_meta_path):
        hr_data = pd.read_csv(hr_data_path, sep='\t', compression='gzip')
        with open(hr_meta_path, 'r') as f:
            hr_meta = json.load(f)
        start_time = hr_meta.get('StartTime')
        hr_data['lsl_timestamp'] = start_time + (hr_data.index / sampling_rate)
    else: hr_data = pd.DataFrame() # return an empty data frame if the file d
    return hr_data

def edia_time_alignment(data_path, meta_path):
    """
    Aligns Eye-Tracking or Head-Movement data with absolute timestamps.

    Steps:
    1. Loads raw TSV data and JSON metadata.
    2. Uses 'StartTime' and 'EffectiveSamplingFrequency' to generate an 'lsl_timestamp' column.
    3. Renames columns based on metadata if available.
    """
    # Creating timestamps for the eyetracking or head movement data based on the timestamp of the Start_Time and EffectiveSamplingFrequency saved in the .json file for the corresponding data
    if os.path.exists(data_path) and os.path.exists(meta_path):
        data = pd.read_csv(data_path, sep='\t', compression='gzip')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        start_time = meta.get('StartTime')
        sampling_rate = meta.get('EffectiveSamplingFrequency')
        data['lsl_timestamp'] = start_time + (data.index / sampling_rate)
        # Get column names from .json file
        column_names = meta.get('Columns')
        if column_names is not None and len(column_names) == len(data.columns) -1:
            data.columns = column_names + ['lsl_timestamp']
    else: data = pd.DataFrame() # return an empty data frame if the file does not exist
    return data

def extract_windows(execution_order_data, anchor, second_anchor=None, length_before=None, length_after=None):
    """
    Defines time windows around specific anchor events.
    Can be used fully modular based on the amount of desired phases and anchors.

    Can operate in two modes:
    1. Fixed Duration: Uses 'length_before' and 'length_after' (in seconds) around an 'anchor' event.
    2. Event-to-Event: Uses a 'second_anchor' to define the window duration dynamically.
    """
    # Filter events based on the anchor
    anchor_events = execution_order_data[execution_order_data['executed'] == anchor]['lsl_timestamp']
    anchor_events = anchor_events[2:].reset_index(drop=True) # Remove two first practice trials and reset index

    # If second_anchor is provided, calculate the delta time between the two anchors and create windows based on this delta time around the first anchor events
    if second_anchor is not None and length_before is None and length_after is None and second_anchor in execution_order_data['executed'].values :
        # Calculate delta time between the two anchors
        second_anchor_events = execution_order_data[execution_order_data['executed'] == second_anchor]['lsl_timestamp']
        second_anchor_events = second_anchor_events[2:].reset_index(drop=True) # Remove two first practice trials and reset index
        delta_time = second_anchor_events.values - anchor_events.values   # LSL timestamps already in seconds
        # Create windows for each event based on the delta time between the two anchors
        windows = pd.DataFrame({
            'window_start': anchor_events,
            'event_time': anchor_events,
            'window_end': anchor_events + delta_time
        }).reset_index(drop=True)  # Reset index to 0-based sequential

    # If length_before and length_after are provided, create windows based on these lengths around the anchor events
    elif length_before is not None and length_after is not None and second_anchor is None and anchor in execution_order_data['executed'].values:
        # Create windows for each event
        windows = pd.DataFrame({
            'window_start': anchor_events - length_before,
            'event_time': anchor_events,
            'window_end': anchor_events + length_after
        }).reset_index(drop=True)  # Reset index to 0-based sequential
    else:
        raise ValueError("Invalid combination of parameters. Please provide either length_before and length_after, or second_anchor.")

    return windows
#%%
#%%
# =================================================================================================
# Main Processing Loop
# =================================================================================================
# Iterates over all subjects, sessions, and runs to:
# 1. Import necessary physiological and behavioral files.
# 2. Align data to a common timeline.
# 3. Extract data for specific phases (Baseline, Approach, Recovery), Trials, and Blocks.
# 4. Compute CWT features (LF, HF Power).
# 5. Save the segmented data structures to JSON (gzipped).
#
# Files processed:
# - ECG peaks: .../ecg-preproc/..._peaks.csv
# - Eyetracking: .../beh/..._recording-ediaeyecentercenter_physio.tsv.gz
# - Head movement: .../beh/..._recording-hmd_physio.tsv.gz
# - Event logs: .../beh/..._events.tsv.gz
# - Execution order: .../unity-edia/.../executionOrder.csv

for subject in subjects:
    for session in sessions:
        runs = sorted(set(f"run-{f.name.split('run-')[1][:3]}" for f in ecg_preproc_dir.glob(f"**/{subject}_{session}_task-TIDAL_run-*") if 'run-' in f.name))
        print(f"Processing {subject} {session} with runs: {runs}")
        for run in runs:
            # ---------------------------------------------------------
            # 1. Define File Paths
            # ---------------------------------------------------------
            # Construct absolute paths for all required input files for the current run.
            ecg_peaks_path = ecg_preproc_dir / subject / session / "beh" / f"{subject}_{session}_task-TIDAL_{run}_recording-polarband_ecg-preproc.json"
            ecg_meta_path = data_dir / subject / session / "beh" / f"{subject}_{session}_task-TIDAL_{run}_recording-polarband_physio.json"
            hr_data_path = ecg_preproc_dir / subject / session / "beh" / f"{subject}_{session}_task-TIDAL_{run}_recording-polarband_hr-bpm-manualcorr.tsv.gz"
            eyetracking_path = data_dir / subject / session / "beh" / f"{subject}_{session}_task-TIDAL_{run}_recording-ediaeyecentercenter_physio.tsv.gz"
            eyetracking_json_path = str(eyetracking_path).replace('.tsv.gz', '.json')
            headmovement_path = data_dir / subject / session / "beh" / f"{subject}_{session}_task-TIDAL_{run}_recording-hmd_physio.tsv.gz"
            headmovement_json_path = str(headmovement_path).replace('.tsv.gz', '.json')
            events_path = data_dir / subject / session / "beh" / f"{subject}_{session}_task-TIDAL_{run}_events.tsv"
            execution_order_path = unity_dir / subject.split("-")[1] / "S001" / "other" / "executionOrder.csv"

            # ---------------------------------------------------------
            # 2. Load and Align Data
            # ---------------------------------------------------------
            # ── Pre-alignment check (informational) ──
            # Verifies the raw LSL-to-Unity mapping before time alignment.
            # Processing continues regardless of the result.
            pfx = subject.split("-")[1]
            pre_check = verify_participant(pfx, session="S001", skip_practice=True, verbose=False)

            # ── Failsafe: Detect AgentStopped count mismatch ──
            # For interrupted sessions (e.g., sub-12P), the raw BIDS events.tsv
            # contains events from both runs, causing a length mismatch with
            # the merged execution order. If detected, fall back to events_fixed.tsv.
            exec_order_df = pd.read_csv(execution_order_path)
            events_df_check = pd.read_csv(events_path, sep='\t')
            n_agent_stopped = (exec_order_df['executed'] == 'AgentStopped').sum()
            n_agent_stopped_walking = (events_df_check['trial_type'] == 'AgentStoppedWalking').sum()

            if n_agent_stopped != n_agent_stopped_walking:
                events_fixed_path = events_path.parent / events_path.name.replace('_events.tsv', '_events_fixed.tsv')

                if events_fixed_path.exists():
                    print(f"  ⚠ Event count mismatch for {subject} ({n_agent_stopped_walking} LSL vs {n_agent_stopped} Unity).")
                    print(f"    → Using fixed events file: {events_fixed_path.name}")
                    events_path = events_fixed_path
                else:
                    print(f"  ✗ Event count mismatch for {subject} ({n_agent_stopped_walking} LSL vs {n_agent_stopped} Unity).")
                    print(f"    No fixed events file found at: {events_fixed_path}")
                    print(f"    → Please run 'merge_interrupted_session.py' first to generate the fixed events file.")
                    print(f"    Skipping {subject} {session} {run}.")
                    continue

            # Load ECG Peaks and align timestamps
            ecg_peaks_data = rr_time_alignment(ecg_peaks_path, ecg_meta_path)

            # HR data
            if not Path(hr_data_path).exists():
                hr_data_path = str(hr_data_path).replace('manualcorr', 'autocorr')
                if not Path(hr_data_path).exists():
                    hr_data_path = str(hr_data_path).replace('autocorr', 'uncorr')
                    if not Path(hr_data_path).exists():
                        print(f"HR data file not found for {subject} {session} {run}: {hr_data_path}")

            hr_data = hr_data_time_alignment(hr_data_path, ecg_meta_path, sampling_rate=1000)

            # ---------------------------------------------------------
            # 3. Feature Extraction (CWT)
            # ---------------------------------------------------------
            # Use Continuous Wavelet Transform (CWT) to extract Low Frequency (LF) and High Frequency (HF)
            # power components from the RR intervals.
            # We compute this for both the 'bbsig' (robust) and 'calc' (simple) RR intervals.
            if not ecg_peaks_data.empty:
                #cwt_processor = Cwt()
                #power_lf_bbsig, power_hf_bbsig, power_hrv_bbsig, times_bbsig = cwt_processor.heart_extract(ecg_peaks_data, 'rr_bbsig')
                #power_lf_calc, power_hf_calc, power_hrv_calc, times_calc = cwt_processor.heart_extract(ecg_peaks_data, 'rr_calc')

                #start_time_offset = ecg_peaks_data['timestamp'].iloc[0]
                #times_bbsig_aligned = times_bbsig + start_time_offset
                #times_calc_aligned = times_calc + start_time_offset

                cwt_processor = Cwt_fixed()
                power_lf_bbsig, power_hf_bbsig, power_hrv_bbsig, times_bbsig_aligned = cwt_processor.heart_extract(ecg_peaks_data, 'rr_bbsig')
                power_lf_calc, power_hf_calc, power_hrv_calc, times_calc_aligned = cwt_processor.heart_extract(ecg_peaks_data, 'rr_calc')

            # Eyetracking data alignment
            eyetracking_data = edia_time_alignment(eyetracking_path, eyetracking_json_path)

            # Head movement data alignment
            headmovement_data = edia_time_alignment(headmovement_path, headmovement_json_path)

            # Extract the starting timestamp from the PolarBand channel in the XDF file
            execution_order_data = execution_order_time_alignment(execution_order_path, events_path)

            # Check which data frames are not empty and print a message if any of them is empty
            if ecg_peaks_data.empty:
                print(f"ECG peaks data frame is empty for {subject} {session} {run}")
            if hr_data.empty:
                print(f"HR data frame is empty for {subject} {session} {run}")
            if eyetracking_data.empty:
                print(f"Eyetracking data frame is empty for {subject} {session} {run}")
            if headmovement_data.empty:
                print(f"Head movement data frame is empty for {subject} {session} {run}")
            if execution_order_data.empty:
                print(f"Execution order data frame is empty for {subject} S001")

            # ---------------------------------------------------------
            # 4. Data Filtering & Metric Calculation
            # ---------------------------------------------------------
            # Eyetracking: Keep only essential columns (Timestamp, Pupil Diameter, Openness, Confidence)
            if not eyetracking_data.empty:
                eyetracking_data = eyetracking_data[['lsl_timestamp', 'pupildiameter', 'openness', 'confidence']]

            # Head Movement: Calculate composite movement vectors
            # - 'head_movement': Norm of position vector (x, y, z)
            # - 'head_rotation': Norm of rotation vector (rotx, roty, rotz)
            if not headmovement_data.empty:
                headmovement_data['head_movement'] = np.sqrt(headmovement_data['posx']**2 + headmovement_data['posy']**2 + headmovement_data['posz']**2)
                headmovement_data['head_rotation'] = np.sqrt(headmovement_data['rotx']**2 + headmovement_data['roty']**2 + headmovement_data['rotz']**2)
                headmovement_data = headmovement_data[['lsl_timestamp', 'head_movement', 'head_rotation']]

            # ---------------------------------------------------------
            # 5. Define Time Windows (Blocks & Trials)
            # ---------------------------------------------------------
            # Extract block timings based on the execution order data
            # Block 1
            b1_start_idx = execution_order_data[execution_order_data['executed'] == 'BlockContinueAfterIntro'].index[1]
            b1_end_idx = execution_order_data[execution_order_data['executed'] == 'BreakStep1'].index[0]
            b1_start_time = execution_order_data.loc[b1_start_idx, 'lsl_timestamp']
            b1_end_time = execution_order_data.loc[b1_end_idx, 'lsl_timestamp']

            # Block 2
            # Check if BreakStep2 exists (might be missing in interrupted sessions like sub-12P)
            break2_indices = execution_order_data[execution_order_data['executed'] == 'BreakStep2'].index

            if len(break2_indices) > 0:
                b2_break_idx = break2_indices[0]
            else:
                # Fallback: use BreakStep1 as the reference point
                # Make sure we use the first BreakStep1 (end of Block 1)
                b2_break_idx = execution_order_data[execution_order_data['executed'] == 'BreakStep1'].index[0]
                print(f"  ⚠ BreakStep2 not found. Using BreakStep1 (idx={b2_break_idx}) as reference for Block 2 start.")

            b2_end_idx = execution_order_data[execution_order_data['executed'] == 'OnTrialEnd'].index[-1]
            b2_start_time = execution_order_data[(execution_order_data['executed'] == 'OnTrialBeginUXF') & (execution_order_data.index > b2_break_idx)]['lsl_timestamp'].iloc[0]
            b2_end_time = execution_order_data.loc[b2_end_idx, 'lsl_timestamp']

            # Create a DataFrame to store the block timings
            block_timings = pd.DataFrame({
                'block': ['Block 1', 'Block 2'],
                'block_start': [b1_start_time, b2_start_time],
                'block_end': [b1_end_time, b2_end_time]
            })

            # Extract trial timings: Look for 'OnTrialBeginUXF' and 'OnTrialEnd' markers
            trial_starts = execution_order_data[execution_order_data['executed'] == 'OnTrialBeginUXF'][['lsl_timestamp']]
            trial_ends = execution_order_data[execution_order_data['executed'] == 'OnTrialEnd'][['lsl_timestamp']]

            # Filter Trials:
            # 1. Remove the first two practice trials
            trial_starts = trial_starts.iloc[2:].reset_index(drop=True)
            trial_ends = trial_ends.iloc[2:].reset_index(drop=True)

            # 2. Remove the break trial (index 32) that occurs between Block 1 and Block 2
            trial_starts = trial_starts.drop(trial_starts.index[32]).reset_index(drop=True)
            trial_ends = trial_ends.drop(trial_ends.index[32]).reset_index(drop=True)


            trial_timings = pd.concat([trial_starts.reset_index(drop=True), trial_ends.reset_index(drop=True)], axis=1)
            trial_timings['trial'] = [f"Trial_{i+1}" for i in range(len(trial_timings))]
            trial_timings.columns = ['trial_start', 'trial_end', 'trial_number']

            # ── Post-alignment check (informational) ──
            # Validates the aligned output: structural integrity, trial counts,
            # durations, block ordering, and delta preservation.
            # Processing continues regardless of the result.
            post_check = validate_alignment(
                execution_order_data, trial_timings, block_timings,
                subject=subject, verbose=False
            )

            # ── Combined alignment summary ──
            print_alignment_summary(subject, pre_check, post_check)
            report_path = save_alignment_report(
                subject, session, run, pre_check, post_check, derivatives_dir
            )
            print(f"    Report saved: {report_path.name}")

            # ---------------------------------------------------------
            # 6. Define Specific Phases (Window Extraction)
            # ---------------------------------------------------------
            # We define three key phases relative to the 'AgentStopped' event:
            # - Baseline: 3 seconds BEFORE the agent appears ('ShowAgent').
            # - Approach: The interval from 'ShowAgent' until 'AgentStopped'.
            # - Recovery: 3 seconds AFTER 'AgentStopped'.
            baseline = ['ShowAgent', None, 3, 0, 'baseline'] # 3 seconds before the anchor (ShowAgent)
            approach = ['ShowAgent', 'AgentStopped', None, None, 'approach'] # from the ShowAgent event to the AgentStopped event
            recovery = ['AgentStopped', None, 0, 3, 'recovery'] # 3 seconds after the anchor (AgentStopped)
            prestop = ['AgentStopped', None, 2, 0, 'prestop'] # 2 seconds before the anchor (AgentStopped)
            poststop = ['AgentStopped', None, 0, 2, 'poststop'] # 2 seconds after the anchor (AgentStopped)

            phases = [baseline, approach, recovery, prestop, poststop]
            phases_data = []
            for i in range(len(phases)):
                window_timings = extract_windows(execution_order_data, anchor=phases[i][0], second_anchor=phases[i][1], length_before=phases[i][2], length_after=phases[i][3])
                window_timings['phase'] = phases[i][4]

                # ---------------------------------------------------------
                # 6.1 Extract Data for Each Phase Window
                # ---------------------------------------------------------
                # Iterate through each defined window for the current phase (e.g., every Baseline window)
                for i in range(len(window_timings)):
                    window_data = []
                    window_start = window_timings.loc[i, 'window_start']
                    window_end = window_timings.loc[i, 'window_end']
                    event_time = window_timings.loc[i, 'event_time']
                    phase = window_timings.loc[i, 'phase']

                    # Extract ECG peaks in the window
                    ecg_peaks_window = ecg_peaks_data[(ecg_peaks_data['timestamp'] >= window_start) & (ecg_peaks_data['timestamp'] <= window_end)]

                    # Extract HR data in the window
                    if not hr_data.empty:
                        hr_data_window = hr_data[(hr_data['lsl_timestamp'] >= window_start) & (hr_data['lsl_timestamp'] <= window_end)]
                    else:
                        hr_data_window = pd.DataFrame()

                    # Extract eyetracking data in the window
                    if not eyetracking_data.empty:
                        eyetracking_window = eyetracking_data[(eyetracking_data['lsl_timestamp'] >= window_start) & (eyetracking_data['lsl_timestamp'] <= window_end)]
                    else:
                        eyetracking_window = pd.DataFrame()

                    # Extract head movement data in the window
                    headmovement_window = headmovement_data[(headmovement_data['lsl_timestamp'] >= window_start) & (headmovement_data['lsl_timestamp'] <= window_end)]

                    # Save as a dictionary:
                    window_data = {
                        'subject': subject,
                        'session': session,
                        'run': run,
                        'phase': phase,
                        'window_start': window_start,
                        'event_time': event_time,
                        'window_end': window_end,
                        'ecg_peaks': ecg_peaks_window.to_dict(orient='list'),
                        'hr_data': hr_data_window.to_dict(orient='list'),
                        'eyetracking_data': eyetracking_window.to_dict(orient='list'),
                        'headmovement_data': headmovement_window.to_dict(orient='list')
                    }

                    mask_bbsig = (times_bbsig_aligned >= window_start) & (times_bbsig_aligned <= window_end)
                    window_data.update({
                        'power_lf_bbsig': power_lf_bbsig[mask_bbsig].tolist(),
                        'power_hf_bbsig': power_hf_bbsig[mask_bbsig].tolist(),
                        'power_hrv_bbsig': power_hrv_bbsig[mask_bbsig].tolist(),
                        'times_bbsig': times_bbsig_aligned[mask_bbsig].tolist()
                    })

                    mask_calc = (times_calc_aligned >= window_start) & (times_calc_aligned <= window_end)
                    window_data.update({
                        'power_lf_calc': power_lf_calc[mask_calc].tolist(),
                        'power_hf_calc': power_hf_calc[mask_calc].tolist(),
                        'power_hrv_calc': power_hrv_calc[mask_calc].tolist(),
                        'times_calc': times_calc_aligned[mask_calc].tolist()
                    })

                    phases_data.append(window_data)

            # ---------------------------------------------------------
            # 6.2 Extract Data for Each Trial
            # ---------------------------------------------------------
            # Iterate through each full trial (Start to End) to extract physiological data.
            trials_data = []

            for i in range(len(trial_timings)):
                trial_start = trial_timings.loc[i, 'trial_start']
                trial_end = trial_timings.loc[i, 'trial_end']
                trial_number = trial_timings.loc[i, 'trial_number']

                # Extract ECG peaks in the trial
                ecg_peaks_trial = ecg_peaks_data[(ecg_peaks_data['timestamp'] >= trial_start) & (ecg_peaks_data['timestamp'] <= trial_end)]

                # Extract eyetracking data in the trial
                if not eyetracking_data.empty:
                    eyetracking_trial = eyetracking_data[(eyetracking_data['lsl_timestamp'] >= trial_start) & (eyetracking_data['lsl_timestamp'] <= trial_end)]
                else:
                    eyetracking_trial = pd.DataFrame()
                # Extract head movement data in the trial
                headmovement_trial = headmovement_data[(headmovement_data['lsl_timestamp'] >= trial_start) & (headmovement_data['lsl_timestamp'] <= trial_end)]

                # Save as a dictionary:
                trial_data = {
                    'subject': subject,
                    'session': session,
                    'run': run,
                    'trial': trial_number,
                    'event': 'Trial',
                    'trial_start': trial_start,
                    'trial_end': trial_end,
                    'ecg_peaks': ecg_peaks_trial.to_dict(orient='list'),
                    'hr_data': hr_data_window.to_dict(orient='list'),
                    'eyetracking_data': eyetracking_trial.to_dict(orient='list'),
                    'headmovement_data': headmovement_trial.to_dict(orient='list')
                }
                mask_bbsig = (times_bbsig_aligned >= trial_start) & (times_bbsig_aligned <= trial_end)
                trial_data.update({
                    'power_lf_bbsig': power_lf_bbsig[mask_bbsig].tolist(),
                    'power_hf_bbsig': power_hf_bbsig[mask_bbsig].tolist(),
                    'power_hrv_bbsig': power_hrv_bbsig[mask_bbsig].tolist(),
                    'times_bbsig': times_bbsig_aligned[mask_bbsig].tolist()
                })
                mask_calc = (times_calc_aligned >= trial_start) & (times_calc_aligned <= trial_end)
                trial_data.update({
                    'power_lf_calc': power_lf_calc[mask_calc].tolist(),
                    'power_hf_calc': power_hf_calc[mask_calc].tolist(),
                    'power_hrv_calc': power_hrv_calc[mask_calc].tolist(),
                    'times_calc': times_calc_aligned[mask_calc].tolist()
                })
                trials_data.append(trial_data)

            # ---------------------------------------------------------
            # 6.3 Extract Data for Each Block
            # ---------------------------------------------------------
            # Iterate through Block 1 and Block 2 to extract continuous data for the entire block duration.
            blocks_data = []

            for i in range(len(block_timings)):
                block_start = block_timings.loc[i, 'block_start']
                block_end = block_timings.loc[i, 'block_end']
                block_name = block_timings.loc[i, 'block']

                # Extract ECG peaks in the block
                ecg_peaks_block = ecg_peaks_data[(ecg_peaks_data['timestamp'] >= block_start) & (ecg_peaks_data['timestamp'] <= block_end)]

                # Extract eyetracking data in the block
                if not eyetracking_data.empty:
                    eyetracking_block = eyetracking_data[(eyetracking_data['lsl_timestamp'] >= block_start) & (eyetracking_data['lsl_timestamp'] <= block_end)]
                else:
                    eyetracking_block = pd.DataFrame()

                # Extract head movement data in the block
                headmovement_block = headmovement_data[(headmovement_data['lsl_timestamp'] >= block_start) & (headmovement_data['lsl_timestamp'] <= block_end)]

                # Save as a dictionary:
                block_data = {
                    'subject': subject,
                    'session': session,
                    'run': run,
                    'block': block_name,
                    'event': 'Block',
                    'block_start': block_start,
                    'block_end': block_end,
                    'ecg_peaks': ecg_peaks_block.to_dict(orient='list'),
                    'hr_data': hr_data_window.to_dict(orient='list'),
                    'eyetracking_data': eyetracking_block.to_dict(orient='list'),
                    'headmovement_data': headmovement_block.to_dict(orient='list')
                }
                mask_bbsig = (times_bbsig_aligned >= block_start) & (times_bbsig_aligned <= block_end)
                block_data.update({
                    'power_lf_bbsig': power_lf_bbsig[mask_bbsig].tolist(),
                    'power_hf_bbsig': power_hf_bbsig[mask_bbsig].tolist(),
                    'power_hrv_bbsig': power_hrv_bbsig[mask_bbsig].tolist(),
                    'times_bbsig': times_bbsig_aligned[mask_bbsig].tolist()
                })
                mask_calc = (times_calc_aligned >= block_start) & (times_calc_aligned <= block_end)
                block_data.update({
                    'power_lf_calc': power_lf_calc[mask_calc].tolist(),
                    'power_hf_calc': power_hf_calc[mask_calc].tolist(),
                    'power_hrv_calc': power_hrv_calc[mask_calc].tolist(),
                    'times_calc': times_calc_aligned[mask_calc].tolist()
                })
                blocks_data.append(block_data)

            # Extract different phases data
            baseline_data = [p for p in phases_data if p['phase'] == 'baseline']
            approach_data = [p for p in phases_data if p['phase'] == 'approach']
            recovery_data = [p for p in phases_data if p['phase'] == 'recovery']
            prestop_data = [p for p in phases_data if p['phase'] == 'prestop']
            poststop_data = [p for p in phases_data if p['phase'] == 'poststop']

            # ---------------------------------------------------------
            # 7. Save Segmented Data
            # ---------------------------------------------------------
            # Compile all extracted data into a structured dictionary and save as gzipped JSON.
            # Output Path: derivatives/segmenting/.../extracted_data.json
            segmented_data = {
                'baseline': baseline_data,
                'approach': approach_data,
                'recovery': recovery_data,
                'prestop': prestop_data,
                'poststop': poststop_data,
                'trials': trials_data,
                'blocks': blocks_data
            }
            segmented_data_path = derivatives_dir / "segmenting" / subject / session / f"{subject}_{session}_task-TIDAL_{run}_segmented_data.json"
            os.makedirs(segmented_data_path.parent, exist_ok=True) # create the directory if it does not exist
            with gzip.open(segmented_data_path, 'wt', encoding='utf-8') as f:
                json.dump(segmented_data, f, indent=4)

            # Print summary of what was extracted
            print(f"Extracted data for {subject} {session} {run}:")
            print(f"- Number of baseline phases: {len(baseline_data)}")
            print(f"- Number of approach phases: {len(approach_data)}")
            print(f"- Number of recovery phases: {len(recovery_data)}")
            print(f"- Number of prestop phases: {len(prestop_data)}")
            print(f"- Number of poststop phases: {len(poststop_data)}")
            print(f"- Number of trials: {len(trial_timings)}")
            print(f"- Number of blocks: {len(block_timings)}")
            # Print the final path where the segmented data is saved
            print(f"- Segmented data saved to: {segmented_data_path}")
            #====End of processing for the current subject, session, and run

# %%
