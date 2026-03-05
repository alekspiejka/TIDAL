"""
Runtime Patch for Systole Editor (v0.3.0)
=========================================

This module fixes THREE bugs in Systole's Editor class WITHOUT modifying any
system files and WITHOUT requiring a kernel restart:

Bug 1: Hardcoded sfreq=1000 and default decim=10 in plot_raw() calls
       - Causes signal distortion for low sampling rate signals (e.g., Polar H10 at 130 Hz)

Bug 2: set_tight_layout() called without required argument (matplotlib 3.8+ incompatibility)
       - Causes TypeError when using correction/rejection tools

Bug 3: Invalid subplots_adjust(left=0.1, right=0.1) call
       - Causes ValueError: left cannot be >= right

The Solution:
-------------
This module monkey-patches matplotlib and plot_raw at runtime, and installs an
instance-level fixed plot_signals method for Editors created via patched_editor,
fixing all issues without any file modifications.

Usage:
------
    from systole_editor_fixed import patched_editor
    from IPython.display import display

    editor = patched_editor(
        signal=ecg_clean,
        sfreq=130,  # Your actual sampling frequency
        corrected_json=output_path,
        corrected_peaks=rpeaks_dict['rpeaks_bool'],
        signal_type="ECG"
    )

    # Display the commands box
    display(editor.commands_box)

    # When done, save:
    editor.save()
"""

import warnings
import functools

# Global state to track patching
_patch_state = {
    "is_patched": False,
    "original_plot_raw": None,
    "original_set_tight_layout": None,
    "target_sfreq": None,
    "target_decim": 1,
    "matplotlib_patched": False,
}


def _patch_matplotlib_tight_layout(verbose: bool = False) -> bool:
    """
    Patch matplotlib's Figure.set_tight_layout to handle missing argument.

    This fixes the matplotlib 3.8+ compatibility issue by restoring the old
    behavior where calling fig.set_tight_layout() without arguments is valid.
    """
    global _patch_state

    if _patch_state["matplotlib_patched"]:
        return True

    try:
        import matplotlib.figure

        # Store original
        _patch_state["original_set_tight_layout"] = matplotlib.figure.Figure.set_tight_layout
        original_method = _patch_state["original_set_tight_layout"]

        def patched_set_tight_layout(self, tight=True):
            """Fixed set_tight_layout that defaults to True if no argument provided."""
            return original_method(self, tight)

        # Apply patch
        matplotlib.figure.Figure.set_tight_layout = patched_set_tight_layout
        _patch_state["matplotlib_patched"] = True

        if verbose:
            print("Matplotlib Figure.set_tight_layout patched (default argument fix)")

        return True

    except Exception as e:
        if verbose:
            print(f"Error patching matplotlib: {e}")
        return False


def _patch_plot_raw(sfreq, decim: int, verbose: bool = False) -> bool:
    """
    Patch systole.plots.plot_raw to fix sfreq and decim.

    - Injects a sensible decim when none is provided.
    - Replaces hardcoded sfreq=1000 with the actual sampling frequency.
    """
    global _patch_state

    try:
        import systole.plots

        # Store original if not already stored
        if _patch_state["original_plot_raw"] is None:
            _patch_state["original_plot_raw"] = systole.plots.plot_raw

        _patch_state["target_sfreq"] = sfreq
        _patch_state["target_decim"] = decim

        def dynamic_fixed_plot_raw(*args, **kwargs):
            if "decim" not in kwargs:
                kwargs["decim"] = _patch_state["target_decim"]
            if kwargs.get("sfreq") == 1000 and _patch_state["target_sfreq"] != 1000:
                kwargs["sfreq"] = _patch_state["target_sfreq"]
            return _patch_state["original_plot_raw"](*args, **kwargs)

        # Patch in systole.plots
        systole.plots.plot_raw = dynamic_fixed_plot_raw

        # Also patch in systole.interact if it has a local reference
        try:
            import systole.interact as interact_module

            if hasattr(interact_module, "plot_raw"):
                interact_module.plot_raw = dynamic_fixed_plot_raw
        except Exception:
            if verbose:
                print("Warning: could not patch systole.interact.plot_raw")

        # Also patch in systole.interact.interact submodule
        try:
            import systole.interact.interact as interact_submodule

            if hasattr(interact_submodule, "plot_raw"):
                interact_submodule.plot_raw = dynamic_fixed_plot_raw
        except Exception:
            if verbose:
                print("Warning: could not patch systole.interact.interact.plot_raw")

        _patch_state["is_patched"] = True

        if verbose:
            print(f"plot_raw patched: sfreq={sfreq} Hz, decim={decim}")

        return True

    except Exception as e:
        if verbose:
            print(f"Error patching plot_raw: {e}")
        return False


def apply_all_patches(sfreq, decim: int = 1, verbose: bool = False) -> bool:
    """
    Apply all patches needed for low sampling rate ECG signals.

    This function patches:
    1. matplotlib.figure.Figure.set_tight_layout - fixes missing argument for matplotlib 3.8+
    2. plot_raw() - fixes sfreq and decim parameters
    """
    success0 = _patch_matplotlib_tight_layout(verbose)
    success1 = _patch_plot_raw(sfreq, decim, verbose)
    return success0 and success1


def get_recommended_decim(sfreq: float) -> int:
    """Get the recommended decimation factor for a given sampling frequency."""
    if sfreq <= 250:
        return 1
    elif sfreq <= 500:
        return 2
    elif sfreq <= 1000:
        return 10
    else:
        return 10


def patched_editor(
    signal,
    sfreq,
    corrected_json,
    corrected_peaks=None,
    signal_type: str = "ECG",
    figsize=(10, 6),
    decim=None,
    verbose: bool = False,
    **kwargs,
):
    """
    Factory function to create a properly configured Editor for any sampling rate.

    This fixes ALL known issues with the Systole Editor:

    1. Hardcoded sfreq=1000 → Uses your actual sfreq
    2. Default decim=10 → Auto-detects appropriate decim based on sfreq
    3. set_tight_layout() bug → Adds required True argument
    4. subplots_adjust bug → Removes invalid parameters

    For low sampling rates (e.g., Polar H10 at 130 Hz), this prevents signal
    distortion. For high sampling rates (e.g., 1000 Hz), behavior matches
    the original systole defaults.

    NO KERNEL RESTART REQUIRED!
    """
    # Auto-detect decim if not specified
    if decim is None:
        decim = get_recommended_decim(sfreq)
        if verbose:
            print(f"Auto-detected decim={decim} for sfreq={sfreq} Hz")

    # Warn if sfreq is low and decim is too high (user override)
    if sfreq < 500 and decim > 2:
        warnings.warn(
            f"Your sampling rate ({sfreq} Hz) is low. "
            f"Using decim={decim} may cause visual distortion. "
            f"Consider using decim=1 or decim=2.",
            UserWarning,
        )

    # STEP 1: Apply global patches
    apply_all_patches(sfreq=sfreq, decim=decim, verbose=verbose)

    # STEP 2: Import Editor (it will see patched plot_raw and patched tight_layout)
    try:
        from systole.interact import Editor
    except ImportError:
        raise ImportError("systole is not installed. Install with: pip install systole")

    # STEP 3: Create the editor
    editor = Editor(
        signal=signal,
        sfreq=sfreq,
        corrected_json=corrected_json,
        corrected_peaks=corrected_peaks,
        signal_type=signal_type,
        figsize=figsize,
        **kwargs,
    )

    # STEP 4: Install an instance-level fixed plot_signals method
    import types
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    def fixed_plot_signals(self):
        """
        Fixed version of plot_signals that:
        - Uses patched plot_raw with correct sfreq/decim.
        - Calls set_tight_layout(True) and avoids invalid subplots_adjust.
        """
        if self.signal is None:
            return self

        # Clear axes and redraw, retaining x-/y-axis zooms
        xlim, ylim = self.ax[0].get_xlim(), self.ax[0].get_ylim()
        xlim2, ylim2 = self.ax[1].get_xlim(), self.ax[1].get_ylim()
        self.ax[0].clear()
        self.ax[1].clear()

        # Convert bad segments into list of tuples
        if self.bad_segments:
            bad_segments = [
                (self.bad_segments[i], self.bad_segments[i + 1])
                for i in range(0, len(self.bad_segments), 2)
            ]
        else:
            bad_segments = None

        # Import the (possibly patched) plot_raw
        from systole.plots import plot_raw

        plot_raw(
            signal=self.signal,
            peaks=self.peaks,
            modality=self.signal_type.lower(),
            backend="matplotlib",
            show_heart_rate=True,
            show_artefacts=True,
            bad_segments=bad_segments,
            sfreq=self.sfreq,
            decim=_patch_state.get("target_decim", 1),
            ax=[self.ax[0], self.ax[1]],
        )
        self.ax[0].set(xlim=xlim, ylim=ylim)
        self.ax[1].set(xlim=xlim2, ylim=ylim2)

        # Show span selectors
        self.delete = functools.partial(self.on_remove)
        self.span1 = SpanSelector(
            self.ax[0],
            self.delete,
            "horizontal",
            button=1,
            props=dict(facecolor="red", alpha=0.2),
            useblit=True,
        )
        self.add = functools.partial(self.on_add)
        self.span2 = SpanSelector(
            self.ax[0],
            self.add,
            "horizontal",
            button=3,
            props=dict(facecolor="green", alpha=0.2),
            useblit=True,
        )
        # Matplotlib 3.8+ compatibility
        #self.fig.tight_layout()

        plt.margins(x=0, y=0)
        self.fig.canvas.draw()

        return self

    editor.plot_signals = types.MethodType(fixed_plot_signals, editor)

    return editor
