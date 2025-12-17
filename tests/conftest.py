"""
Pytest configuration and shared fixtures for PyHEARTS tests.
"""

import numpy as np
import pandas as pd
import pytest



@pytest.fixture
def sampling_rate() -> float:
    """Standard sampling rate for tests (1000 Hz)."""
    return 1000.0


@pytest.fixture
def human_sampling_rate() -> float:
    """Standard human ECG sampling rate (500 Hz)."""
    return 500.0


@pytest.fixture
def mouse_sampling_rate() -> float:
    """Standard mouse ECG sampling rate (2000 Hz)."""
    return 2000.0


@pytest.fixture
def simple_ecg_signal(sampling_rate: float) -> np.ndarray:
    """
    Generate a simple synthetic ECG-like signal for testing.
    
    Creates a signal with clear R-peaks at regular intervals,
    suitable for testing R-peak detection and epoching.
    """
    duration = 10  # seconds
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Create baseline
    signal = np.zeros(n_samples)
    
    # Add R-peaks (sharp Gaussian peaks) at regular intervals (~60 bpm)
    heart_rate = 60  # bpm
    rr_interval = 60 / heart_rate  # seconds
    peak_times = np.arange(0.5, duration - 0.5, rr_interval)
    
    for peak_time in peak_times:
        # R-peak: tall, narrow Gaussian
        r_peak = 1.0 * np.exp(-((t - peak_time) ** 2) / (2 * 0.01 ** 2))
        
        # P-wave: small positive, before R
        p_wave = 0.15 * np.exp(-((t - (peak_time - 0.16)) ** 2) / (2 * 0.02 ** 2))
        
        # Q-wave: small negative, just before R
        q_wave = -0.1 * np.exp(-((t - (peak_time - 0.04)) ** 2) / (2 * 0.01 ** 2))
        
        # S-wave: negative, just after R
        s_wave = -0.2 * np.exp(-((t - (peak_time + 0.04)) ** 2) / (2 * 0.01 ** 2))
        
        # T-wave: positive, after QRS
        t_wave = 0.3 * np.exp(-((t - (peak_time + 0.25)) ** 2) / (2 * 0.04 ** 2))
        
        signal += r_peak + p_wave + q_wave + s_wave + t_wave
    
    return signal


@pytest.fixture
def noisy_ecg_signal(simple_ecg_signal: np.ndarray) -> np.ndarray:
    """Add realistic noise to the simple ECG signal."""
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(simple_ecg_signal))
    return simple_ecg_signal + noise


@pytest.fixture
def rr_intervals_ms() -> np.ndarray:
    """Sample RR intervals in milliseconds for HRV testing."""
    # Typical human RR intervals around 1000ms (60 bpm) with some variability
    np.random.seed(42)
    base_rr = 1000  # ms
    variability = 50  # ms
    n_intervals = 100
    return base_rr + np.random.normal(0, variability, n_intervals)


@pytest.fixture
def rr_intervals_with_nan(rr_intervals_ms: np.ndarray) -> np.ndarray:
    """RR intervals with some NaN values for edge case testing."""
    rr_with_nan = rr_intervals_ms.copy()
    rr_with_nan[5] = np.nan
    rr_with_nan[20] = np.nan
    rr_with_nan[50] = np.nan
    return rr_with_nan


@pytest.fixture
def sample_epoch_df(sampling_rate: float) -> pd.DataFrame:
    """Create a sample epoch DataFrame for testing."""
    n_samples = 500  # About half a second at 1000 Hz
    t = np.arange(n_samples)
    
    # Create a single cycle waveform
    signal = np.zeros(n_samples)
    center = n_samples // 2
    
    # R-peak at center
    r_idx = center
    signal += 1.0 * np.exp(-((t - r_idx) ** 2) / (2 * 10 ** 2))
    
    # P-wave
    p_idx = center - 160
    signal += 0.15 * np.exp(-((t - p_idx) ** 2) / (2 * 20 ** 2))
    
    # Q-wave
    q_idx = center - 40
    signal += -0.1 * np.exp(-((t - q_idx) ** 2) / (2 * 10 ** 2))
    
    # S-wave
    s_idx = center + 40
    signal += -0.2 * np.exp(-((t - s_idx) ** 2) / (2 * 10 ** 2))
    
    # T-wave
    t_idx = center + 200
    signal += 0.3 * np.exp(-((t - t_idx) ** 2) / (2 * 40 ** 2))
    
    return pd.DataFrame({
        "time": t / sampling_rate,
        "signal": signal,
        "global_idx": t + 1000,  # Simulating offset in full signal
        "cycle": 0,
    })


@pytest.fixture
def default_config():
    """Default ProcessCycleConfig for testing."""
    from pyhearts.config import ProcessCycleConfig
    return ProcessCycleConfig()


@pytest.fixture
def human_config():
    """Human preset ProcessCycleConfig for testing."""
    from pyhearts.config import ProcessCycleConfig
    return ProcessCycleConfig.for_human()


@pytest.fixture
def mouse_config():
    """Mouse preset ProcessCycleConfig for testing."""
    from pyhearts.config import ProcessCycleConfig
    return ProcessCycleConfig.for_mouse()

