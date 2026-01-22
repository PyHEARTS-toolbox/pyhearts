"""
Smoke tests for `pyhearts.plots`.

These tests ensure plotting helpers can be imported and called in a headless
environment. They do not validate plot aesthetics.
"""

import numpy as np


def test_plots_smoke(monkeypatch, simple_ecg_signal, sampling_rate, sample_epoch_df):
    # Ensure non-interactive backend before importing pyplot-heavy modules.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Prevent any blocking show() calls from older plot helpers.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    from pyhearts.plots import (
        plot_detrended_cycle,
        plot_dynamic_offset,
        plot_epochs,
        plot_fit,
        plot_fwhm,
        plot_labeled_peaks,
        plot_rise_decay,
        plot_rpeaks,
    )

    sig = np.asarray(simple_ecg_signal, dtype=float)
    xs = np.arange(sig.size)

    # Functions that return Axes (newer API)
    ax = plot_rpeaks(sig, float(sampling_rate), r_peaks=[int(np.argmax(sig))], show=False)
    assert ax is not None

    ax2 = plot_labeled_peaks(
        xs=xs,
        signal=sig,
        peak_data={"R": {"center_idx": int(np.argmax(sig))}},
        show=False,
    )
    assert ax2 is not None

    ax3 = plot_rise_decay(
        xs=xs,
        sig=sig,
        peak_data={"R": {"le_idx": 10, "center_idx": int(np.argmax(sig)), "ri_idx": 20}},
        show=False,
    )
    assert ax3 is not None

    ax4 = plot_fwhm(
        xs=xs,
        signal=sig,
        peak_inds={"R": int(np.argmax(sig))},
        fwhm_results={"R": {"fwhm_left": 10, "fwhm_right": 20}},
        show=False,
    )
    assert ax4 is not None

    # Older helpers (do not return axes; always call show()).
    # If these raise, we still want to know.
    plot_fit(xs=xs, sig_detrended=sig, fit=sig)
    plot_epochs(all_cycles=[sig[:200], sig[:200]], x_vals=np.arange(200))
    plot_detrended_cycle(xs=xs, sig=sig, sig_corrected=sig, cycle=0)
    plot_dynamic_offset(
        xs=xs,
        sig=sig,
        r_center_idx=int(np.argmax(sig)),
        r_left_idx=max(0, int(np.argmax(sig)) - 5),
        r_right_idx=min(len(sig) - 1, int(np.argmax(sig)) + 5),
        q_min_idx=max(0, int(np.argmax(sig)) - 10),
        s_max_idx=min(len(sig) - 1, int(np.argmax(sig)) + 10),
    )


