"""
Interrupted Session Repair Utility
==================================

**Description:**
    This script is a generalized utility for repairing and merging interrupted experimental blocks.
    It is designed to handle cases where a participant's session was stopped (e.g., due to a crash or technical issue)
    and restarted as a new block, resulting in split data files.

**Features:**
    -   Merges two separate run folders (e.g., "Block1" and "Block2") into a single session folder (e.g., "S001").
    -   Stitches trial data (`trial_results.csv`) and execution logs (`executionOrder.csv`) with correct time offsets.
    -   Cleans and merges LSL event streams, preserving absolute timestamps for BIDS compatibility.
    -   Generates a BIDS-compatible `_events_fixed.tsv` for use in downstream pipelines.

**Usage:**
    Run this script from the terminal with the required arguments.

    **Example: Fixing sub-12P (Standard Case)**
    Participant 12P had a crash after Block 1. The session was restarted, creating "Block2".
    We want to merge "Block1" (all trials) with "Block2" (starting effectively from Block 2).

    ```bash
    python merge_interrupted_session.py --subject sub-12P --run1 Block1 --run2 Block2 --run2-start-block 2
    ```

    **Arguments:**
    -   `--subject`: Participant ID (e.g., `sub-12P`).
    -   `--session`: Session ID (default: `ses-001`).
    -   `--run1`: Name of the folder containing the first run (e.g., `Block1`).
    -   `--run2`: Name of the folder containing the second run (e.g., `Block2`).
    -   `--run2-start-block`: The block number in Run 2 that contains valid data (default: `2`).
                             This renumbers the trials to continue sequentially from Run 1.
    -   `--lsl-skip-start`: Number of LSL events to skip at the start of Run 2 (default: `2` for practice trials).
    -   `--lsl-skip-end`: Number of events to skip at the end of Run 2 (default: `2`).

**Output:**
    Creates a merged session folder (e.g., `S001`) containing:
    -   `trial_results.csv`: Unified trial data.
    -   `other/executionOrder.csv`: Unified execution log.
    -   `lsl_events_cleaned.csv`: Cleaned LSL events.
    -   Allows `segmenting.py` to process the data as a single continuous session.
    Also creates:
    -   `.../beh/..._events_fixed.tsv`: BIDS-compatible events file in the BIDS directory.

"""

import pandas as pd
import numpy as np
import pyxdf
from pathlib import Path
import argparse


def load_trial_data(block1_path, block2_path):
    """Load and inspect trial data from both blocks."""
    if not block1_path.exists():
        raise FileNotFoundError(f"Block 1 path not found: {block1_path}")
    if not block2_path.exists():
        raise FileNotFoundError(f"Block 2 path not found: {block2_path}")

    df_block1 = pd.read_csv(block1_path)
    df_block2 = pd.read_csv(block2_path)

    print(f"\n  Block1: {len(df_block1)} trials")
    for block_num in sorted(df_block1['block_num'].unique()):
        block_data = df_block1[df_block1['block_num'] == block_num]
        block_id = block_data['blockId'].iloc[0]
        print(f"    Block {block_num} ({block_id}): {len(block_data)} trials")

    print(f"\n  Block2: {len(df_block2)} trials")
    for block_num in sorted(df_block2['block_num'].unique()):
        block_data = df_block2[df_block2['block_num'] == block_num]
        block_id = block_data['blockId'].iloc[0]
        print(f"    Block {block_num} ({block_id}): {len(block_data)} trials")

    return df_block1, df_block2


def extract_blocks_for_merging(df_block1, df_block2, run2_start_block):
    """Extract relevant blocks: all of Block1 and valid blocks from Block2."""
    # All of Block1
    df_block1_all = df_block1.copy()

    # Block2: Extract from the specified start block
    # For sub-12P, run2_start_block is 2 (experiment_0)
    df_block2_valid = df_block2[df_block2['block_num'] == run2_start_block].reset_index(drop=True)

    print(f"\n  Extracting:")
    print(f"    Block1 (all): {len(df_block1_all)} trials")
    print(f"      Time: {df_block1_all['start_time'].min():.3f}s - {df_block1_all['end_time'].max():.3f}s")
    print(f"    Block2 (starting block {run2_start_block}): {len(df_block2_valid)} trials")
    print(f"      Time: {df_block2_valid['start_time'].min():.3f}s - {df_block2_valid['end_time'].max():.3f}s")

    return df_block1_all, df_block2_valid


def calculate_timing_offset(df_block1_all, df_block2_valid):
    """Calculate offset to make Block2 follow Block1 seamlessly."""
    block1_end = df_block1_all['end_time'].max()
    block2_start = df_block2_valid['start_time'].min()
    offset = block1_end - block2_start

    print(f"\n  Timing offset calculation:")
    print(f"    Block1 end: {block1_end:.3f}s")
    print(f"    Block2 start (local): {block2_start:.3f}s")
    print(f"    Offset: {offset:.3f}s")

    return offset


def adjust_and_renumber_block2(df_block2_valid, offset, run2_start_block):
    """Adjust timings and renumber Block2 trials."""
    df_adjusted = df_block2_valid.copy()

    # Adjust timestamps
    df_adjusted['start_time'] += offset
    df_adjusted['end_time'] += offset

    # Renumber blocks and IDs
    # Assuming standard structure: Practice=1, Block1=2, Break=3, Block2=4
    # If run2_start_block is 2 (experiment_0), it should likely map to 4 (experiment_1)
    # This logic generalizes "next block" mapping.
    # For 12P: Block 1 is block_num 2. Block 2 should be block_num 4.
    current_block_num = df_block2_valid['block_num'].iloc[0]
    new_block_num = current_block_num + 2 # e.g., 2 -> 4

    # Update block metadata
    df_adjusted['block_num'] = new_block_num

    # Update blockId strings (e.g., task-tidal_experiment_0 -> task-tidal_experiment_1)
    # This replaces the last digit with the incremented digit
    df_adjusted['blockId'] = df_adjusted['blockId'].apply(
        lambda x: x[:-1] + str(int(x[-1]) + 1) if x[-1].isdigit() else x
    )

    print(f"\n  Block2 adjusted:")
    print(f"    New time range: {df_adjusted['start_time'].min():.3f}s - {df_adjusted['end_time'].max():.3f}s")
    print(f"    Renumbered: block_num {current_block_num} -> {new_block_num}")
    print(f"    Block IDs updated (e.g., {df_block2_valid['blockId'].iloc[0]} -> {df_adjusted['blockId'].iloc[0]})")

    return df_adjusted


def merge_trial_results(df_block1_all, df_block2_adjusted, output_path):
    """Merge trial data and renumber sequentially."""
    df_merged = pd.concat([df_block1_all, df_block2_adjusted], ignore_index=True)

    # Renumber trial_num sequentially
    df_merged['trial_num'] = range(1, len(df_merged) + 1)

    # Renumber trial_num_in_block within each block
    for block_num in sorted(df_merged['block_num'].unique()):
        block_mask = df_merged['block_num'] == block_num
        df_merged.loc[block_mask, 'trial_num_in_block'] = range(1, block_mask.sum() + 1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)

    print(f"\n  Merged trial_results.csv:")
    print(f"    Total trials: {len(df_merged)}")
    print(f"    Saved to: {output_path}")

    return df_merged


def merge_execution_order(block1_exec_path, block2_exec_path,
                         df_block2_valid, offset, output_path):
    """Merge executionOrder.csv files with timing adjustment."""
    df_exec_block1 = pd.read_csv(block1_exec_path)
    df_exec_block2 = pd.read_csv(block2_exec_path)

    # Extract execution events for Block2 valid range
    block2_start = df_block2_valid['start_time'].min()
    block2_end = df_block2_valid['end_time'].max()

    df_exec_block2_valid = df_exec_block2[
        (df_exec_block2['timestamp'] >= block2_start) &
        (df_exec_block2['timestamp'] <= block2_end)
    ].copy()

    # Adjust timestamps to match the merged timeline
    df_exec_block2_valid['timestamp'] += offset

    # Merge and sort
    df_exec_merged = pd.concat([df_exec_block1, df_exec_block2_valid],
                               ignore_index=True).sort_values('timestamp').reset_index(drop=True)

    # Cleanup artifact: Check for last trial having identical start/end timestamps (0 duration)
    # This prevents segmenting.py failures for sub-12P and similar cases where interruption causes ghost trials
    try:
        if not df_exec_merged.empty:
            last_start_idx = df_exec_merged[df_exec_merged['executed'] == 'OnTrialBeginUXF'].last_valid_index()
            last_end_idx = df_exec_merged[df_exec_merged['executed'] == 'OnTrialEnd'].last_valid_index()

            if last_start_idx is not None and last_end_idx is not None:
                last_start_ts = df_exec_merged.at[last_start_idx, 'timestamp']
                last_end_ts = df_exec_merged.at[last_end_idx, 'timestamp']

                # If they are effectively equal (0 duration)
                if abs(last_end_ts - last_start_ts) < 1e-6:
                     print(f"  Warning: Detected 0-duration trial artifact at {last_start_ts:.4f}s. Removing...")
                     # Remove events starting from this trial
                     drop_thresh = min(last_start_idx, last_end_idx)
                     df_exec_merged = df_exec_merged.loc[:drop_thresh-1].copy()
    except Exception as e:
        print(f"  Warning: Cleanup of artifact failed: {e}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_exec_merged.to_csv(output_path, index=False)

    print(f"\n  Merged executionOrder.csv:")
    print(f"    Total events: {len(df_exec_merged)}")
    print(f"    Saved to: {output_path}")

    return df_exec_merged


def clean_lsl_events(xdf_path, output_path, skip_start=2, skip_end=2):
    """Load and clean LSL AgentStopped events."""
    if not xdf_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_path}")

    print(f"  Loading XDF: {xdf_path.name}")
    streams, _ = pyxdf.load_xdf(str(xdf_path))

    # Find Unity event stream
    event_stream = None
    for stream in streams:
        stream_name = stream['info'].get('name', [''])[0]
        if 'Unity.Event' in stream_name or 'Marker' in stream_name:
            event_stream = stream
            break

    if event_stream is None:
        raise ValueError("Could not find Unity.Event stream in XDF file")

    event_times = event_stream['time_stamps']
    event_data = event_stream['time_series']

    # Create events dataframe
    # 'timestamp' = relative (zero-based, for the cleaned CSV output)
    # 'lsl_onset' = absolute LSL timestamps (for BIDS events_fixed.tsv)
    events_df = pd.DataFrame({
        'timestamp': event_times - event_times[0],
        'lsl_onset': event_times,
        'event': [str(e[0]) if isinstance(e, (list, np.ndarray)) else str(e)
                 for e in event_data]
    })

    # Filter AgentStopped events
    events_df = events_df[events_df['event'].str.contains('AgentStopped',
                                                          case=False, na=False)]

    # Split into runs (Run 1 and Run 2) based on large time gap
    # Assuming >100s gap indicates the break/restart
    event_diffs = np.diff(events_df['timestamp'].values)
    big_gaps = np.where(event_diffs > 100)[0]

    if len(big_gaps) == 0:
        print("  Warning: No large gap found in LSL stream. Assuming continuous recording or single run.")
        # Determine behavior? For now, return all events
        return events_df

    split_idx = big_gaps[0] + 1
    events_run1 = events_df.iloc[:split_idx].copy()
    events_run2 = events_df.iloc[split_idx:].copy()

    print(f"\n  LSL events:")
    print(f"    Run 1: {len(events_run1)} events")
    print(f"    Run 2: {len(events_run2)} events (before cleaning)")

    # Clean Run 2 events
    # Default: remove first 2 (practice) and last 2 (extraneous)
    if len(events_run2) > (skip_start + skip_end):
        events_run2_cleaned = events_run2.iloc[skip_start:-skip_end].copy().reset_index(drop=True)
        print(f"    Run 2: {len(events_run2_cleaned)} events (after removing {skip_start} start + {skip_end} end)")
    else:
        print(f"    Warning: Run 2 has too few events ({len(events_run2)}) to skip {skip_start}+{skip_end}. Using all.")
        events_run2_cleaned = events_run2

    # Combine
    events_cleaned = pd.concat([events_run1, events_run2_cleaned], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_cleaned.to_csv(output_path, index=False)

    print(f"    Total cleaned: {len(events_cleaned)} events")
    print(f"    Saved to: {output_path}")

    return events_cleaned


def verify_output(df_merged, events_cleaned, base_path):
    """Verify merged data against reference participants (01P, 02P, 03P)."""
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")

    # Check block structure
    print(f"\n  Block structure:")
    for block_num in sorted(df_merged['block_num'].unique()):
        block_data = df_merged[df_merged['block_num'] == block_num]
        block_id = block_data['blockId'].iloc[0]
        print(f"    Block {block_num} ({block_id}): {len(block_data)} trials")

    print(f"\n  Totals:")
    print(f"    Trials: {len(df_merged)}")
    print(f"    LSL events: {len(events_cleaned)}")

    # Compare with reference participants
    print(f"\n  Comparison with reference participants:")
    ref_participants = ["01P", "02P", "03P"]
    all_match = True
    for ref_pid in ref_participants:
        ref_trial_path = base_path / "unity-edia" / ref_pid / "S001" / "trial_results.csv"
        if ref_trial_path.exists():
            ref_df = pd.read_csv(ref_trial_path)
            # We don't necessarily expect exact match if block lengths differ, but checking is good
            match = len(ref_df) == len(df_merged)
            status = "[OK]" if match else "[FAIL]"
            print(f"    {status} {ref_pid}: {len(ref_df)} trials")
        else:
            print(f"    - {ref_pid}: not found")


def generate_bids_events_fixed(events_cleaned, bids_events_path):
    """Generate a BIDS-compatible _events_fixed.tsv from cleaned LSL events."""
    bids_events = pd.DataFrame({
        'onset': events_cleaned['lsl_onset'].values,
        'duration': 0.0,
        'trial_type': events_cleaned['event'].values
    })

    bids_events_path.parent.mkdir(parents=True, exist_ok=True)
    bids_events.to_csv(bids_events_path, sep='\t', index=False)

    print(f"\n  BIDS-compatible events file generated:")
    print(f"    {len(bids_events)} events -> {bids_events_path}")

    return bids_events


def main():
    """Main execution function with CLI args."""
    parser = argparse.ArgumentParser(description="Merge interrupted TIDAL sessions.")
    parser.add_argument("--subject", type=str, default="sub-12P", help="Subject ID (e.g., sub-12P)")
    parser.add_argument("--session", type=str, default="ses-001", help="Session ID (e.g., ses-001)")
    parser.add_argument("--run1", type=str, default="Block1", help="Folder name for Run 1")
    parser.add_argument("--run2", type=str, default="Block2", help="Folder name for Run 2")
    parser.add_argument("--run2-start-block", type=int, default=2, help="Block number in Run 2 to start merging from (default: 2)")
    parser.add_argument("--lsl-skip-start", type=int, default=2, help="LSL events to skip at start of Run 2 (default: 2)")
    parser.add_argument("--lsl-skip-end", type=int, default=2, help="LSL events to skip at end of Run 2 (default: 2)")

    args = parser.parse_args()

    # Configuration
    root = Path(__file__).parent.parent  # tidal/ project root
    data_dir = root / "data"
    sourcedata_dir = data_dir / "sourcedata"

    subject = args.subject
    pfx = subject.replace("sub-", "")
    session = args.session

    print(f"\n{'='*80}")
    print(f"MERGE INTERRUPTED SESSION: {subject} {session}")
    print(f"Run 1: {args.run1} | Run 2: {args.run2} (Start Block: {args.run2_start_block})")
    print(f"{'='*80}")

    # Paths
    # Input
    xdf_path = sourcedata_dir / subject / session / "beh" / f"{subject}_{session}_task-TIDAL_run-001_beh.xdf"
    block1_dir = sourcedata_dir / "unity-edia" / pfx / args.run1
    block2_dir = sourcedata_dir / "unity-edia" / pfx / args.run2

    block1_path = block1_dir / "trial_results.csv"
    block2_path = block2_dir / "trial_results.csv"
    block1_exec_path = block1_dir / "other" / "executionOrder.csv"
    block2_exec_path = block2_dir / "other" / "executionOrder.csv"

    # Output
    output_dir = sourcedata_dir / "unity-edia" / pfx / "S001"
    output_trial_path = output_dir / "trial_results.csv"
    output_exec_path = output_dir / "other" / "executionOrder.csv"
    output_events_path = output_dir / "lsl_events_cleaned.csv"
    bids_events_fixed_path = data_dir / subject / session / "beh" / f"{subject}_{session}_task-TIDAL_run-001_events_fixed.tsv"

    # Step 1: Load trial data
    print(f"\n[1/8] Loading trial data...")
    df_block1, df_block2 = load_trial_data(block1_path, block2_path)

    # Step 2: Extract blocks for merging
    print(f"\n[2/8] Extracting blocks...")
    df_block1_all, df_block2_valid = extract_blocks_for_merging(df_block1, df_block2, args.run2_start_block)

    # Step 3: Calculate timing offset
    print(f"\n[3/8] Calculating timing offset...")
    timing_offset = calculate_timing_offset(df_block1_all, df_block2_valid)

    # Step 4: Adjust and renumber Block2
    print(f"\n[4/8] Adjusting Block2...")
    df_block2_adjusted = adjust_and_renumber_block2(df_block2_valid, timing_offset, args.run2_start_block)

    # Step 5: Merge trial results
    print(f"\n[5/8] Merging trial_results.csv...")
    df_merged = merge_trial_results(df_block1_all, df_block2_adjusted, output_trial_path)

    # Step 6: Merge execution order
    print(f"\n[6/8] Merging executionOrder.csv...")
    df_exec_merged = merge_execution_order(block1_exec_path, block2_exec_path,
                                          df_block2_valid, timing_offset, output_exec_path)

    # Step 7: Clean LSL events
    print(f"\n[7/8] Cleaning LSL events...")
    events_cleaned = clean_lsl_events(xdf_path, output_events_path, args.lsl_skip_start, args.lsl_skip_end)

    # Step 8: Generate BIDS-compatible fixed events file
    print(f"\n[8/8] Generating BIDS-compatible events_fixed.tsv...")
    generate_bids_events_fixed(events_cleaned, bids_events_fixed_path)

    # Verification
    verify_output(df_merged, events_cleaned, sourcedata_dir)
    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
