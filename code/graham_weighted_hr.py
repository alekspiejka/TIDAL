"""
Graham's (1978) Weighted IBI Method for Event-Related Heart Rate Analysis
=========================================================================

Implements second-by-second (or half-second) weighted heart rate computation
following Graham (1978, Psychophysiology, 15, 492-498), and extraction of
four primary cardiac metrics for VR stop-distance paradigms.

The weighted method assigns each IBI's instantaneous heart rate (60/IBI_s)
to time bins in proportion to the fraction of each bin that the IBI occupies.
This avoids bias from arithmetic averaging of IBIs within arbitrary windows.

References
----------
- Graham, F.K. (1978). Constraints on measuring heart rate and period
  sequentially through real and cardiac time. Psychophysiology, 15, 492-498.
- Perakakis, P., Joffily, M., Taylor, M., Guerra, P., & Vila, J. (2010).
  KARDIA: A Matlab software for the analysis of cardiac interbeat interval data.
  Computer Methods and Programs in Biomedicine, 98, 83-89.
- Quigley, K.S., et al. (2024). Publication guidelines for human heart rate
  and heart rate variability studies in psychophysiology. Psychophysiology, 61, e14604.

"""
#%%
import numpy as np
from typing import Tuple, Dict, Optional
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

#%%
def graham_weighted_hr(
    ibi_s: np.ndarray,
    ibi_times: np.ndarray,
    epoch_start: float,
    epoch_end: float,
    bin_width: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted heart rate in fixed time bins using Graham's (1978) method.

    Each IBI spans a time interval [ibi_onset, ibi_offset]. For each time bin,
    the instantaneous HR (60 / IBI_duration) of every overlapping IBI is weighted
    by the proportion of the bin that IBI occupies. If an IBI covers the entire
    bin, its weight = 1.0. If it covers half, weight = 0.5. Weights within a bin
    sum to 1.0 (assuming no gaps in the IBI series).

    Parameters
    ----------
    ibi_s : np.ndarray, shape (n_ibis,)
        IBI durations in seconds.
    ibi_times : np.ndarray, shape (n_ibis,)
        Timestamp of each IBI. Convention: the time of the SECOND R-peak
        defining each IBI (i.e., the end of the interval). This means the
        IBI spanning [t_Rpeak_i, t_Rpeak_{i+1}] has ibi_times = t_Rpeak_{i+1}.
    epoch_start : float
        Start time of the epoch (absolute time, same clock as ibi_times).
    epoch_end : float
        End time of the epoch.
    bin_width : float
        Width of each time bin in seconds. Default 0.5s (half-second bins).

    Returns
    -------
    bin_centers : np.ndarray, shape (n_bins,)
        Center time of each bin (absolute time).
    weighted_hr : np.ndarray, shape (n_bins,)
        Weighted heart rate (bpm) for each bin. NaN if no IBI data covers the bin.

    Notes
    -----
    The IBI onset for each interval is computed as: ibi_onset = ibi_times - ibi_s.
    This requires that ibi_times marks the end of each IBI (second R-peak).

    If your ibi_times instead mark the START of each IBI (first R-peak), set:
        ibi_times_end = ibi_times + ibi_s
    and pass ibi_times_end as ibi_times.
    """

    # Derive IBI onset and offset times
    ibi_offsets = ibi_times.copy()
    ibi_onsets = ibi_times - ibi_s

    # Instantaneous HR for each IBI
    ibi_hr = 60.0 / ibi_s

    # Create time bins
    bin_edges = np.arange(epoch_start, epoch_end + bin_width * 0.01, bin_width)
    n_bins = len(bin_edges) - 1
    bin_centers = bin_edges[:-1] + bin_width / 2.0

    weighted_hr = np.full(n_bins, np.nan)

    for b in range(n_bins):
        b_start = bin_edges[b]
        b_end = bin_edges[b + 1]

        # Find IBIs that overlap with this bin
        # An IBI overlaps if its onset < bin_end AND its offset > bin_start
        overlap_mask = (ibi_onsets < b_end) & (ibi_offsets > b_start)

        if not np.any(overlap_mask):
            continue

        # Compute overlap fraction for each contributing IBI
        overlap_starts = np.maximum(ibi_onsets[overlap_mask], b_start)
        overlap_ends = np.minimum(ibi_offsets[overlap_mask], b_end)
        overlap_durations = overlap_ends - overlap_starts

        # Weight = fraction of the bin covered by this IBI
        weights = overlap_durations / bin_width
        hrs = ibi_hr[overlap_mask]

        # Weighted HR for this bin
        # If IBIs fully tile the bin, weights sum to 1.0
        # If there's a gap (artifact rejection), weights sum to < 1.0
        # Normalize by sum of weights to handle partial coverage
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weighted_hr[b] = np.sum(weights * hrs) / weight_sum

    return bin_centers, weighted_hr


def extract_cardiac_metrics(
    bin_centers: np.ndarray,
    weighted_hr: np.ndarray,
    event_time: float,
    baseline_window: Tuple[float, float] = (-3.0, 0.0),
    approach_window: Tuple[float, float] = (0.0, 6.0),
    poststop_window: Optional[Tuple[float, float]] = (6.0, 9.0),
    showagent_time: Optional[float] = None,
    agentstopped_time: Optional[float] = None,
) -> Dict[str, float]:
    """
    Extract four primary cardiac metrics from a weighted HR epoch.

    All windows are specified RELATIVE to event_time (trial onset).

    Parameters
    ----------
    bin_centers : np.ndarray
        Absolute times of bin centers (from graham_weighted_hr).
    weighted_hr : np.ndarray
        Weighted HR values (bpm) per bin.
    event_time : float
        Absolute time of the event (trial onset / avatar render).
    baseline_window : tuple of (float, float)
        (start, end) in seconds relative to event_time for baseline computation.
        Default (-3.0, 0.0) = last 3s of rating period before avatar renders.
    approach_window : tuple of (float, float)
        (start, end) relative to event_time for approach phase.
        Default (0.0, 6.0). Set end to (button_press - event_time) per trial.
    poststop_window : tuple or None
        (start, end) relative to event_time for post-stop recovery.
        Set start to (button_press - event_time) per trial.
    showagent_time : float or None
        Absolute timestamp of the ShowAgent event (avatar appearance).
    agentstopped_time : float or None
        Absolute timestamp of the AgentStopped event (avatar stops).

    Returns
    -------
    metrics : dict with keys:
        'baseline_hr'         : mean HR during baseline window (bpm)
        'peak_decel_amplitude': maximum HR decrease from baseline (bpm, negative = deceleration)
        'peak_decel_latency'  : time of peak deceleration relative to event_time (seconds)
        'peak_accel_amplitude': maximum HR increase from baseline (bpm, positive = acceleration)
        'peak_accel_latency'  : time of peak acceleration relative to event_time (seconds)
        'mean_hr_change_approach': mean delta HR during approach (bpm relative to baseline)
        'mean_hr_change_poststop': mean delta HR during post-stop (bpm relative to baseline)
                                   NaN if poststop_window is None or no data.
        'hr_timecourse'       : array of delta HR values (baseline-subtracted) for all bins
        'bin_times_relative'  : array of bin times relative to event_time
        'showagent_time'      : absolute timestamp of ShowAgent event (if provided)
        'agentstopped_time'   : absolute timestamp of AgentStopped event (if provided)
    """
    # Convert bin_centers to relative time
    rel_times = bin_centers - event_time

    # --- Baseline ---
    bl_mask = (rel_times >= baseline_window[0]) & (rel_times < baseline_window[1])
    bl_mask &= ~np.isnan(weighted_hr)

    if np.sum(bl_mask) == 0:
        raise ValueError(
            f"No valid HR data in baseline window {baseline_window} relative to event."
        )

    baseline_hr = np.nanmean(weighted_hr[bl_mask])

    # --- Delta HR (baseline-subtracted) ---
    delta_hr = weighted_hr - baseline_hr

    # --- Approach window ---
    app_mask = (rel_times >= approach_window[0]) & (rel_times < approach_window[1])
    app_mask &= ~np.isnan(weighted_hr)

    if np.sum(app_mask) == 0:

        mean_hr_change_approach = np.nan
    else:
        # Peak deceleration = minimum delta HR (most negative value)
        # Peak acceleration = maximum delta HR (most positive value)

        mean_hr_change_approach = np.nanmean(weighted_hr[app_mask]) - baseline_hr

    # --- Trial metrics (for deceleration, covering approach + poststop)
    trial_mask = (rel_times >= approach_window[0]) & (rel_times < poststop_window[1])
    trial_mask &= ~np.isnan(weighted_hr)

    if np.sum(trial_mask) == 0:
        peak_decel_amplitude = np.nan
        peak_decel_latency = np.nan
    else:
        trial_delta_hr = delta_hr[trial_mask]
        trial_rel_times = rel_times[trial_mask]

        if np.all(trial_delta_hr >= 0):
            peak_decel_amplitude = 0.0
            peak_decel_latency = 0.0
        else:
            peak_decel_amplitude = np.min(trial_delta_hr)
            peak_decel_latency = trial_rel_times[np.argmin(trial_delta_hr)]

    # --- Approach metrics (for acceleration, covering ONLY the approach window)
    if np.sum(app_mask) == 0:
        peak_accel_amplitude = np.nan
        peak_accel_latency = np.nan
    else:
        app_delta_hr = delta_hr[app_mask]
        app_rel_times = rel_times[app_mask]

        if np.all(app_delta_hr <= 0):
            peak_accel_amplitude = 0.0
            peak_accel_latency = 0.0
        else:
            peak_accel_amplitude = np.max(app_delta_hr)
            peak_accel_latency = app_rel_times[np.argmax(app_delta_hr)]


    # --- find_peaks Prominence-based extraction ---
    peak_accel_amplitude_fp = np.nan
    peak_accel_latency_fp = np.nan
    peak_decel_amplitude_fp = np.nan
    peak_decel_latency_fp = np.nan

    clean_delta_hr = np.nan_to_num(delta_hr, nan=0.0)
    smoothed_hr = uniform_filter1d(clean_delta_hr, size=3)

    if np.sum(app_mask) > 0:
        s_app = smoothed_hr[app_mask]
        t_app = rel_times[app_mask]
        approach_end_t = approach_window[1]

        peaks, _ = find_peaks(s_app, prominence=0.3, width=1)
        candidates = list(peaks)

        # Include the last bin as a candidate when the signal is still rising
        # at the agent-stop boundary (find_peaks cannot detect boundary samples).
        # This covers the case where the true acceleration peak lands exactly at
        # the moment the agent stops.
        last_idx = len(s_app) - 1
        if last_idx not in candidates and last_idx > 0:
            if s_app[last_idx] >= s_app[last_idx - 1]:
                candidates.append(last_idx)

        if len(candidates) > 0:
            # Select the prominent peak closest to the agent-stop event.
            lats = t_app[np.array(candidates)]
            best = candidates[int(np.argmin(np.abs(lats - approach_end_t)))]
            peak_accel_amplitude_fp = s_app[best]
            peak_accel_latency_fp = t_app[best]

    # fp decel: search starts after the accel peak so mid-approach dips cannot
    # be selected as the deceleration trough.  Falls back to agent-stop time.
    fp_decel_start = (
        peak_accel_latency_fp
        if not np.isnan(peak_accel_latency_fp)
        else approach_window[1]
    )
    fp_trial_mask = (rel_times >= fp_decel_start) & (rel_times < poststop_window[1])
    fp_trial_mask &= ~np.isnan(weighted_hr)

    if np.sum(fp_trial_mask) > 0:
        s_trial = smoothed_hr[fp_trial_mask]
        t_trial = rel_times[fp_trial_mask]
        approach_end_t = approach_window[1]

        inv_s_trial = -s_trial
        peaks, _ = find_peaks(inv_s_trial, prominence=0.3, width=1)
        candidates = list(peaks)

        # Boundary candidate at END of search window only: find_peaks never
        # detects the last bin, so include it if the signal is still falling
        # there (trough extends beyond the post-stop window).
        # NOTE: no stop-time boundary candidate here — adding it would always
        # win the closest-to-stop competition when the signal is still falling
        # at stop, masking the true post-stop deceleration trough.
        last_idx = len(s_trial) - 1
        if last_idx not in candidates and last_idx > 0:
            if s_trial[last_idx] <= s_trial[last_idx - 1]:
                candidates.append(last_idx)

        if len(candidates) > 0:
            # Select the deceleration trough closest to the agent-stop event.
            lats = t_trial[np.array(candidates)]
            best = candidates[int(np.argmin(np.abs(lats - approach_end_t)))]
            peak_decel_amplitude_fp = s_trial[best]
            peak_decel_latency_fp = t_trial[best]


    # --- Post-stop window ---
    if poststop_window is not None:
        ps_mask = (rel_times >= poststop_window[0]) & (rel_times < poststop_window[1])
        ps_mask &= ~np.isnan(weighted_hr)
        if np.sum(ps_mask) > 0:
            mean_hr_change_poststop = np.nanmean(delta_hr[ps_mask])
        else:
            mean_hr_change_poststop = np.nan
    else:
        mean_hr_change_poststop = np.nan

    # Replace NaN in hr_timecourse with 0 (no measured change)
    delta_hr_output = delta_hr.copy()
    delta_hr_output[np.isnan(delta_hr_output)] = 0.0

    return {
        "baseline_hr": baseline_hr,
        "peak_decel_amplitude": peak_decel_amplitude,
        "peak_decel_latency": peak_decel_latency,
        "peak_accel_amplitude": peak_accel_amplitude,
        "peak_accel_latency": peak_accel_latency,
        "peak_decel_amplitude_fp": peak_decel_amplitude_fp,
        "peak_decel_latency_fp": peak_decel_latency_fp,
        "peak_accel_amplitude_fp": peak_accel_amplitude_fp,
        "peak_accel_latency_fp": peak_accel_latency_fp,
        "mean_hr_change_approach": mean_hr_change_approach,
        "mean_hr_change_poststop": mean_hr_change_poststop,
        "hr_timecourse": delta_hr_output,
        "bin_times_relative": rel_times,
        "showagent_time": showagent_time,
        "agentstopped_time": agentstopped_time,
    }



def process_all_trials(
    ibi_s: np.ndarray,
    ibi_times: np.ndarray,
    trial_onsets: np.ndarray,
    button_presses: np.ndarray,
    baseline_duration: float = 3.0,
    poststop_duration: float = 3.0,
    bin_width: float = 0.5,
    max_artifact_fraction: float = 0.20,
) -> list:
    """
    Process all trials for a single participant, returning per-trial metrics.

    Parameters
    ----------
    ibi_s : np.ndarray
        Full-session IBI durations in seconds.
    ibi_times : np.ndarray
        Full-session IBI timestamps (second R-peak of each interval).
    trial_onsets : np.ndarray, shape (n_trials,)
        Absolute time of avatar render for each trial.
    button_presses : np.ndarray, shape (n_trials,)
        Absolute time of button press (stop) for each trial.
    baseline_duration : float
        Seconds before trial onset to use as baseline. Default 3.0.
    poststop_duration : float
        Seconds after button press to include. Default 3.0.
    bin_width : float
        Time bin width in seconds. Default 0.5.
    max_artifact_fraction : float
        Maximum fraction of NaN bins allowed in approach window before
        rejecting the trial. Default 0.20.

    Returns
    -------
    trial_results : list of dict
        Each dict contains the metrics from extract_cardiac_metrics plus:
        - 'trial_idx': trial index (0-based)
        - 'trial_duration': approach duration in seconds
        - 'rejected': bool, True if trial was rejected due to artifacts
        - 'n_bins_approach': number of approach bins
        - 'n_nan_bins_approach': number of NaN bins in approach window
    """
    n_trials = len(trial_onsets)
    trial_results = []

    for i in range(n_trials):
        onset = trial_onsets[i]
        stop = button_presses[i]
        trial_dur = stop - onset

        epoch_start = onset - baseline_duration
        epoch_end = stop + poststop_duration

        # Compute weighted HR for this epoch
        bin_centers, weighted_hr = graham_weighted_hr(
            ibi_s, ibi_times, epoch_start, epoch_end, bin_width
        )

        # Check artifact fraction in approach window
        rel_times = bin_centers - onset
        app_mask = (rel_times >= 0) & (rel_times < trial_dur)
        n_approach_bins = np.sum(app_mask)
        n_nan_approach = np.sum(np.isnan(weighted_hr[app_mask])) if n_approach_bins > 0 else 0
        nan_fraction = n_nan_approach / max(n_approach_bins, 1)

        rejected = nan_fraction > max_artifact_fraction

        if rejected:
            trial_results.append({
                "trial_idx": i,
                "trial_duration": trial_dur,
                "rejected": True,
                "n_bins_approach": int(n_approach_bins),
                "n_nan_bins_approach": int(n_nan_approach),
                "baseline_hr": np.nan,
                "peak_decel_amplitude": np.nan,
                "peak_decel_latency": np.nan,
                "peak_accel_amplitude": np.nan,
                "peak_accel_latency": np.nan,
                "peak_decel_amplitude_fp": np.nan,
                "peak_decel_latency_fp": np.nan,
                "peak_accel_amplitude_fp": np.nan,
                "peak_accel_latency_fp": np.nan,
                "mean_hr_change_approach": np.nan,
                "mean_hr_change_poststop": np.nan,
                "hr_timecourse": None,
                "bin_times_relative": None,
            })
            continue

        # Extract metrics
        try:
            metrics = extract_cardiac_metrics(
                bin_centers,
                weighted_hr,
                event_time=onset,
                baseline_window=(-baseline_duration, 0.0),
                approach_window=(0.0, trial_dur),
                poststop_window=(trial_dur, trial_dur + poststop_duration),
            )
        except ValueError:
            # No baseline data (e.g., first trial with no prior rating period)
            metrics = {
                "baseline_hr": np.nan,
                "peak_decel_amplitude": np.nan,
                "peak_decel_latency": np.nan,
                "peak_accel_amplitude": np.nan,
                "peak_accel_latency": np.nan,
                "peak_decel_amplitude_fp": np.nan,
                "peak_decel_latency_fp": np.nan,
                "peak_accel_amplitude_fp": np.nan,
                "peak_accel_latency_fp": np.nan,
                "mean_hr_change_approach": np.nan,
                "mean_hr_change_poststop": np.nan,
                "hr_timecourse": None,
                "bin_times_relative": None,
            }

        metrics["trial_idx"] = i
        metrics["trial_duration"] = trial_dur
        metrics["rejected"] = False
        metrics["n_bins_approach"] = int(n_approach_bins)
        metrics["n_nan_bins_approach"] = int(n_nan_approach)
        trial_results.append(metrics)

    return trial_results


def compute_condition_rmssd(
    ibi_s: np.ndarray,
    ibi_times: np.ndarray,
    trial_onsets: np.ndarray,
    button_presses: np.ndarray,
    condition_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute condition-level lnRMSSD by pooling IBIs within each condition.

    Only successive differences WITHIN trials are used (not across trial
    boundaries) to avoid contamination from inter-trial intervals.

    Parameters
    ----------
    ibi_s : np.ndarray
        Full-session IBI durations in seconds.
    ibi_times : np.ndarray
        Full-session IBI timestamps (second R-peak).
    trial_onsets : np.ndarray
        Trial onset times.
    button_presses : np.ndarray
        Button press times.
    condition_labels : np.ndarray of str
        Condition label for each trial (e.g., 'male_fast', 'female_slow').

    Returns
    -------
    condition_rmssd : dict
        Keys are unique condition labels, values are lnRMSSD.
    """
    unique_conditions = np.unique(condition_labels)
    condition_rmssd = {}

    for cond in unique_conditions:
        cond_mask = condition_labels == cond
        successive_diffs = []

        for i in np.where(cond_mask)[0]:
            # Find IBIs within this trial's approach window
            trial_mask = (ibi_times > trial_onsets[i]) & (ibi_times <= button_presses[i])
            trial_ibis = ibi_s[trial_mask]

            if len(trial_ibis) >= 2:
                # Successive differences in ms
                diffs = np.diff(trial_ibis) * 1000.0  # convert to ms
                successive_diffs.extend(diffs.tolist())

        if len(successive_diffs) >= 3:
            successive_diffs = np.array(successive_diffs)
            rmssd = np.sqrt(np.mean(successive_diffs ** 2))
            condition_rmssd[cond] = np.log(rmssd)  # lnRMSSD
        else:
            condition_rmssd[cond] = np.nan

    return condition_rmssd

# %%
# ─────────────────────────────────────────────────────────────
# TIDAL Pipeline — Event-Related Cardiac Metrics Extraction
# ─────────────────────────────────────────────────────────────
#
# Produces TWO sets of output files per run:
#
# 1. ShowAgent-anchored (original):
#    *_graham-cardiac_bbsig.json / *_graham-cardiac_calc.json
#    t=0 = avatar appearance (ShowAgent)
#    Epoch: baseline (3s pre-ShowAgent) → approach → recovery
#
# 2. AgentStopped-anchored (new):
#    *_graham-cardiac_agentstopped_bbsig.json / *_agentstopped_calc.json
#    t=0 = avatar stops (AgentStopped)
#    Epoch: prestop (2s pre-stop) + poststop (2s post-stop)
#    Baseline: same 3s pre-ShowAgent window as above
#
# Outputs are saved as JSON files under:
#   data/derivatives/graham-cardiac/sub-XXX/ses-XXX/
# ─────────────────────────────────────────────────────────────
import json
import gzip
from pathlib import Path

# ========== CONFIGURATION ==========

root = Path(__file__).parent.parent  # project root (tidal/)
data_dir = root / "data"
derivatives_dir = data_dir / "derivatives"
segmenting_dir = derivatives_dir / "segmenting"
output_base_dir = derivatives_dir / "graham-cardiac"

# Define subjects and sessions
pfxs = ["01P", "02P", "03P", "04P", "05P", "06P", "07P", "08P", "09P", "10P", "11P", "12P", "13P", "14P", "15P", "16P","17P","18P","19P","20P"]
subjects = [f"sub-{pfx}" for pfx in pfxs]
sessions = ["ses-001"]
task_name = "TIDAL"

# Graham HR settings
bin_width = 0.5              # Time bin width in seconds (half-second bins)
baseline_duration = 3.0      # Baseline window: 3 s before ShowAgent
poststop_duration = 3.0      # Post-stop window: 3 s after AgentStopped (original epoch)
agentstopped_pre  = 2.0      # Seconds before AgentStopped for stop-centred epoch
agentstopped_post = 2.0      # Seconds after AgentStopped for stop-centred epoch


# ========== HELPER FUNCTIONS ==========

def nan_safe(val):
    """Convert numpy NaN / inf to None for JSON serialization."""
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


def load_segmented_data(segmented_data_path):
    """Load gzip-compressed segmented data JSON."""
    if not segmented_data_path.exists():
        print(f"  Warning: File not found: {segmented_data_path}")
        return None
    with gzip.open(segmented_data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def save_results(results, output_dir, subject, session, run, rr_label):
    """Save Graham cardiac metrics to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{subject}_{session}_task-{task_name}_{run}_graham-cardiac_{rr_label}.json"
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    return filepath


# ========== MAIN PROCESSING LOOP ==========

if __name__ == "__main__":

    print("=" * 80)
    print("GRAHAM WEIGHTED HR — Event-Related Cardiac Metrics")
    print("=" * 80)
    print(f"Bin width: {bin_width} s | Baseline: {baseline_duration} s | Post-stop: {poststop_duration} s")
    print("=" * 80)

    for subject in subjects:
        for session in sessions:
            subject_dir = segmenting_dir / subject / session

            if not subject_dir.exists():
                continue

            segmented_files = sorted(subject_dir.glob(
                f"{subject}_{session}_task-{task_name}_run-*_segmented_data.json"
            ))

            if not segmented_files:
                continue

            print(f"\n{subject} {session}:")

            for segmented_file in segmented_files:
                run = segmented_file.name.split(f'task-{task_name}_')[1].split('_segmented')[0]

                # Load segmented data
                segmented_data = load_segmented_data(segmented_file)
                if segmented_data is None:
                    continue

                # Check that all original phases exist
                required_phases = ['baseline', 'approach', 'recovery']
                missing = [p for p in required_phases if p not in segmented_data or len(segmented_data[p]) == 0]
                if missing:
                    print(f"  {run}: Missing phases {missing}, skipping.")
                    continue

                # Check availability of stop-centred segments (optional — warn but don't skip)
                stop_phases = ['prestop', 'poststop']
                missing_stop = [p for p in stop_phases if p not in segmented_data or len(segmented_data[p]) == 0]
                has_stop_segments = len(missing_stop) == 0
                if missing_stop:
                    print(f"  {run}: Missing stop-centred segments {missing_stop} — skipping AgentStopped epoch.")

                n_trials = min(len(segmented_data[p]) for p in required_phases)
                print(f"  {run}: Processing {n_trials} trials...")

                graham_results_bbsig = []
                graham_results_calc = []
                # Combined outputs with timestamps
                graham_results_combined_bbsig = []
                graham_results_combined_calc = []
                graham_results_agentstopped_combined_bbsig = []
                graham_results_agentstopped_combined_calc = []

                for i in range(n_trials):
                    bl = segmented_data['baseline'][i]
                    ap = segmented_data['approach'][i]
                    rc = segmented_data['recovery'][i]

                    # Get AgentStopped time if available
                    agentstopped_time = None
                    po = None
                    if has_stop_segments:
                        ps = segmented_data['prestop'][i]
                        po = segmented_data['poststop'][i]
                        agentstopped_time = ps['event_time']

                    for rr_source, results_list in [('rr_bbsig', graham_results_bbsig),
                                                    ('rr_calc', graham_results_calc)]:
                        # Concatenate timestamps and RR intervals across all three phases
                        timestamps = (bl['ecg_peaks']['timestamp']
                                    + ap['ecg_peaks']['timestamp']
                                    + rc['ecg_peaks']['timestamp'])
                        rr_vals    = (bl['ecg_peaks'][rr_source]
                                    + ap['ecg_peaks'][rr_source]
                                    + rc['ecg_peaks'][rr_source])

                        ibi_s = np.array(rr_vals)

                        # Adjust timestamps: segmenting.py marks IBI START (first R-peak),
                        # but Graham method expects IBI END (second R-peak)
                        ibi_times = np.array(timestamps) + ibi_s

                        if len(ibi_s) < 3:
                            print(f"    Trial {i} ({rr_source}): Too few IBIs ({len(ibi_s)}), skipping.")
                            continue

                        # Define the epoch and event anchors
                        event_time  = ap['event_time']       # ShowAgent timestamp
                        epoch_start = bl['window_start']     # Start of baseline
                        epoch_end   = rc['window_end']       # End of recovery
                        approach_dur = ap['window_end'] - event_time  # Duration of approach phase

                        # Compute weighted HR in fixed time bins
                        bin_centers, weighted_hr = graham_weighted_hr(
                            ibi_s, ibi_times, epoch_start, epoch_end, bin_width=bin_width
                        )

                        # Extract cardiac metrics
                        try:
                            metrics = extract_cardiac_metrics(
                                bin_centers, weighted_hr,
                                event_time=event_time,
                                baseline_window=(-baseline_duration, 0.0),
                                approach_window=(0.0, approach_dur),
                                poststop_window=(approach_dur, approach_dur + poststop_duration),
                                showagent_time=event_time,
                                agentstopped_time=agentstopped_time
                            )
                        except ValueError as e:
                            print(f"    Trial {i} ({rr_source}): {e}")
                            continue

                        # Build original result dict (convert numpy arrays/NaN for JSON)
                        result = {
                            'subject': subject,
                            'session': session,
                            'run': run,
                            'trial_idx': i,
                            'rr_source': rr_source,
                            'baseline_hr': nan_safe(float(metrics['baseline_hr'])),
                            'peak_decel_amplitude': nan_safe(float(metrics['peak_decel_amplitude'])),
                            'peak_decel_latency': nan_safe(float(metrics['peak_decel_latency'])),
                            'peak_accel_amplitude': nan_safe(float(metrics['peak_accel_amplitude'])),
                            'peak_accel_latency': nan_safe(float(metrics['peak_accel_latency'])),
                            'peak_decel_amplitude_fp': nan_safe(float(metrics['peak_decel_amplitude_fp'])),
                            'peak_decel_latency_fp': nan_safe(float(metrics['peak_decel_latency_fp'])),
                            'peak_accel_amplitude_fp': nan_safe(float(metrics['peak_accel_amplitude_fp'])),
                            'peak_accel_latency_fp': nan_safe(float(metrics['peak_accel_latency_fp'])),
                            'mean_hr_change_approach': nan_safe(float(metrics['mean_hr_change_approach'])),
                            'mean_hr_change_poststop': nan_safe(float(metrics['mean_hr_change_poststop'])),
                            'approach_duration': approach_dur,
                            'hr_timecourse': [nan_safe(float(v)) for v in metrics['hr_timecourse']],
                            'bin_times_relative': metrics['bin_times_relative'].tolist(),
                        }
                        results_list.append(result)

                        # Build combined result dict WITH timestamps for new outputs
                        combined_list = graham_results_combined_bbsig if rr_source == 'rr_bbsig' else graham_results_combined_calc
                        result_combined = result.copy()
                        result_combined['showagent_time'] = nan_safe(metrics['showagent_time'])
                        result_combined['agentstopped_time'] = nan_safe(metrics['agentstopped_time'])
                        combined_list.append(result_combined)

                    # ── AgentStopped-centred epoch ────────────────────────────────────────
                    if not has_stop_segments:
                        continue

                    # ps and po already loaded above
                    stop_time = agentstopped_time
                    showagent_time_for_stop = ap['event_time']  # ShowAgent event for reference

                    for rr_source, stop_list in [
                        ('rr_bbsig', graham_results_agentstopped_combined_bbsig),
                        ('rr_calc',  graham_results_agentstopped_combined_calc),
                    ]:
                        # No baseline subtraction for the agentstopped epoch.
                        # We plot raw HR centred on the stop event.
                        # Only prestop + poststop peaks needed.
                        timestamps_stop = (
                            ps['ecg_peaks']['timestamp']
                            + po['ecg_peaks']['timestamp']
                        )
                        rr_vals_stop = (
                            ps['ecg_peaks'][rr_source]
                            + po['ecg_peaks'][rr_source]
                        )

                        ibi_s_stop     = np.array(rr_vals_stop)
                        ibi_times_stop = np.array(timestamps_stop) + ibi_s_stop

                        if len(ibi_s_stop) < 3:
                            continue

                        # Epoch: exactly the ±2 s window around AgentStopped
                        epoch_start_stop = ps['window_start']  # stop_time - 2s
                        epoch_end_stop   = po['window_end']    # stop_time + 2s

                        bin_centers_stop, weighted_hr_stop = graham_weighted_hr(
                            ibi_s_stop, ibi_times_stop,
                            epoch_start_stop, epoch_end_stop,
                            bin_width=bin_width
                        )

                        # bin_times_relative: centred on stop_time
                        bin_times_rel = bin_centers_stop - stop_time

                        # Phase means: raw HR (no baseline subtraction)
                        pre_mask  = (bin_times_rel >= -agentstopped_pre) & (bin_times_rel < 0.0)
                        post_mask = (bin_times_rel >= 0.0) & (bin_times_rel <= agentstopped_post)
                        mean_hr_prestop  = float(np.nanmean(weighted_hr_stop[pre_mask]))  if pre_mask.any()  else float('nan')
                        mean_hr_poststop = float(np.nanmean(weighted_hr_stop[post_mask])) if post_mask.any() else float('nan')

                        result_stop = {
                            'subject':                subject,
                            'session':                session,
                            'run':                    run,
                            'trial_idx':              i,
                            'rr_source':              rr_source,
                            'anchor':                 'AgentStopped',
                            'showagent_time':         nan_safe(showagent_time_for_stop),
                            'agentstopped_time':      nan_safe(stop_time),
                            'mean_hr_prestop':        nan_safe(mean_hr_prestop),
                            'mean_hr_poststop':       nan_safe(mean_hr_poststop),
                            'approach_duration':      approach_dur,
                            'hr_timecourse':          [nan_safe(float(v)) for v in weighted_hr_stop],
                            'bin_times_relative':     bin_times_rel.tolist(),
                        }
                        stop_list.append(result_stop)

                # Save results — ShowAgent anchor
                output_dir = output_base_dir / subject / session
                if graham_results_bbsig:
                    fp = save_results(graham_results_bbsig, output_dir, subject, session, run, 'bbsig')
                    print(f"    Saved: {fp.name}  ({len(graham_results_bbsig)} trials)")
                if graham_results_calc:
                    fp = save_results(graham_results_calc, output_dir, subject, session, run, 'calc')
                    print(f"    Saved: {fp.name}  ({len(graham_results_calc)} trials)")

                # Save results — Combined with timestamps
                if graham_results_combined_bbsig:
                    fp = save_results(graham_results_combined_bbsig, output_dir, subject, session, run, 'combined_bbsig')
                    print(f"    Saved: {fp.name}  ({len(graham_results_combined_bbsig)} trials)")
                if graham_results_combined_calc:
                    fp = save_results(graham_results_combined_calc, output_dir, subject, session, run, 'combined_calc')
                    print(f"    Saved: {fp.name}  ({len(graham_results_combined_calc)} trials)")

    print("\n" + "=" * 80)
    print("GRAHAM CARDIAC EXTRACTION COMPLETE!")
    print("=" * 80)

#%%
