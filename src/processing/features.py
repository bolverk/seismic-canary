"""Spectral analysis and seismic feature extraction.

Extracts additional seismic features beyond P/S ratio for use in
anomaly detection. Each feature function returns (value, uncertainty, quality).
"""
import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)

# Quality flags
QUALITY_GOOD = "good"
QUALITY_MARGINAL = "marginal"
QUALITY_POOR = "poor"
QUALITY_UNAVAILABLE = "unavailable"


@dataclass
class FeatureMeasurement:
    """A single feature measurement with metadata."""
    name: str
    value: Optional[float]
    uncertainty: Optional[float]
    quality: str
    station: Optional[str] = None
    method: Optional[str] = None


def compute_mb_ms(mb: Optional[float], ms: Optional[float]) -> FeatureMeasurement:
    """Compute mb - Ms discriminant.

    The mb-Ms criterion is one of the most established explosion
    discriminants. Explosions typically have mb-Ms > 1.0, while
    earthquakes have mb-Ms < 0.5.

    Args:
        mb: Body-wave magnitude.
        ms: Surface-wave magnitude.

    Returns:
        FeatureMeasurement with mb-Ms value.
    """
    if mb is None or ms is None:
        return FeatureMeasurement(
            name="mb_ms", value=None, uncertainty=None,
            quality=QUALITY_UNAVAILABLE, method="catalog"
        )

    value = mb - ms
    # Typical uncertainty in magnitude is ~0.2-0.3
    uncertainty = 0.3  # conservative estimate

    quality = QUALITY_GOOD if abs(mb) > 0 and abs(ms) > 0 else QUALITY_MARGINAL

    return FeatureMeasurement(
        name="mb_ms", value=value, uncertainty=uncertainty,
        quality=quality, method="catalog"
    )


def compute_spectral_slope(
    data: np.ndarray,
    sampling_rate: float,
    pick_sample: int,
    window_seconds: float = 3.0,
) -> FeatureMeasurement:
    """Compute spectral slope from the P-wave window.

    Explosions tend to have steeper spectral falloff at high frequencies
    compared to earthquakes.

    Args:
        data: Waveform data array.
        sampling_rate: Sampling rate in Hz.
        pick_sample: Sample index of P-wave arrival.
        window_seconds: Analysis window length.

    Returns:
        FeatureMeasurement with spectral slope in dB/octave.
    """
    window_samples = int(window_seconds * sampling_rate)
    start = pick_sample
    end = min(start + window_samples, len(data))

    if end - start < int(sampling_rate):
        return FeatureMeasurement(
            name="spectral_slope", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="fft"
        )

    segment = data[start:end]

    # Apply Hanning window
    window = np.hanning(len(segment))
    windowed = segment * window

    # FFT
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0/sampling_rate)

    # Fit slope in log-log space (1-10 Hz range)
    mask = (freqs >= 1.0) & (freqs <= 10.0) & (spectrum > 0)
    if np.sum(mask) < 5:
        return FeatureMeasurement(
            name="spectral_slope", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="fft"
        )

    log_freqs = np.log10(freqs[mask])
    log_spectrum = np.log10(spectrum[mask])

    # Linear regression in log-log space
    coeffs = np.polyfit(log_freqs, log_spectrum, 1)
    slope = coeffs[0]

    # Estimate uncertainty from residuals
    fitted = np.polyval(coeffs, log_freqs)
    residuals = log_spectrum - fitted
    uncertainty = float(np.std(residuals))

    quality = QUALITY_GOOD if uncertainty < 0.5 else QUALITY_MARGINAL

    return FeatureMeasurement(
        name="spectral_slope", value=float(slope), uncertainty=uncertainty,
        quality=quality, method="fft_linear_fit"
    )


def compute_corner_frequency(
    data: np.ndarray,
    sampling_rate: float,
    pick_sample: int,
    window_seconds: float = 5.0,
) -> FeatureMeasurement:
    """Estimate corner frequency from P-wave spectrum.

    The corner frequency is related to source size. Explosions tend
    to have higher corner frequencies than earthquakes of similar
    magnitude.

    Args:
        data: Waveform data array.
        sampling_rate: Sampling rate in Hz.
        pick_sample: P-wave arrival sample.
        window_seconds: Analysis window.

    Returns:
        FeatureMeasurement with corner frequency in Hz.
    """
    window_samples = int(window_seconds * sampling_rate)
    start = pick_sample
    end = min(start + window_samples, len(data))

    if end - start < int(sampling_rate * 2):
        return FeatureMeasurement(
            name="corner_frequency", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="spectral_fit"
        )

    segment = data[start:end]
    window = np.hanning(len(segment))
    windowed = segment * window

    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0/sampling_rate)

    # Find corner frequency: frequency where spectrum drops to
    # half of its low-frequency plateau (in amplitude)
    mask = (freqs >= 0.5) & (freqs <= sampling_rate/4) & (spectrum > 0)
    if np.sum(mask) < 10:
        return FeatureMeasurement(
            name="corner_frequency", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="spectral_fit"
        )

    valid_freqs = freqs[mask]
    valid_spectrum = spectrum[mask]

    # Low-frequency level: average of lowest 20% of frequencies
    n_low = max(3, len(valid_spectrum) // 5)
    low_freq_level = np.mean(valid_spectrum[:n_low])

    if low_freq_level <= 0:
        return FeatureMeasurement(
            name="corner_frequency", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="spectral_fit"
        )

    # Find where spectrum drops below half the plateau
    half_level = low_freq_level * 0.5
    below_half = np.where(valid_spectrum < half_level)[0]

    if len(below_half) == 0:
        # Spectrum never drops - corner is above Nyquist
        return FeatureMeasurement(
            name="corner_frequency", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="spectral_fit"
        )

    corner_idx = below_half[0]
    corner_freq = float(valid_freqs[corner_idx])

    # Uncertainty: ± 1 frequency bin
    freq_resolution = valid_freqs[1] - valid_freqs[0] if len(valid_freqs) > 1 else 1.0
    uncertainty = float(freq_resolution)

    quality = QUALITY_GOOD if 1.0 < corner_freq < 15.0 else QUALITY_MARGINAL

    return FeatureMeasurement(
        name="corner_frequency", value=corner_freq, uncertainty=uncertainty,
        quality=quality, method="spectral_half_level"
    )


def compute_dominant_frequency(
    data: np.ndarray,
    sampling_rate: float,
    pick_sample: int,
    window_seconds: float = 3.0,
) -> FeatureMeasurement:
    """Compute dominant frequency of the P-wave window.

    Args:
        data: Waveform data.
        sampling_rate: Sampling rate in Hz.
        pick_sample: P-wave arrival sample.
        window_seconds: Analysis window.

    Returns:
        FeatureMeasurement with dominant frequency in Hz.
    """
    window_samples = int(window_seconds * sampling_rate)
    start = pick_sample
    end = min(start + window_samples, len(data))

    if end - start < int(sampling_rate):
        return FeatureMeasurement(
            name="dominant_frequency", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="fft_peak"
        )

    segment = data[start:end]
    window = np.hanning(len(segment))
    windowed = segment * window

    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0/sampling_rate)

    # Find peak in 0.5 - Nyquist/2 range
    mask = (freqs >= 0.5) & (freqs <= sampling_rate/4)
    if np.sum(mask) < 3:
        return FeatureMeasurement(
            name="dominant_frequency", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="fft_peak"
        )

    valid_spectrum = spectrum[mask]
    valid_freqs = freqs[mask]

    peak_idx = np.argmax(valid_spectrum)
    dominant_freq = float(valid_freqs[peak_idx])

    # Uncertainty: width of peak at half max
    half_max = valid_spectrum[peak_idx] / 2
    above_half = valid_spectrum >= half_max
    freq_resolution = valid_freqs[1] - valid_freqs[0] if len(valid_freqs) > 1 else 1.0
    uncertainty = float(np.sum(above_half) * freq_resolution / 2)

    quality = QUALITY_GOOD if dominant_freq > 0.5 else QUALITY_MARGINAL

    return FeatureMeasurement(
        name="dominant_frequency", value=dominant_freq, uncertainty=uncertainty,
        quality=quality, method="fft_peak"
    )


def compute_snr(
    data: np.ndarray,
    sampling_rate: float,
    pick_sample: int,
    signal_window: float = 3.0,
    noise_window: float = 5.0,
) -> FeatureMeasurement:
    """Compute signal-to-noise ratio.

    SNR = RMS(signal window) / RMS(noise window before P).

    Args:
        data: Waveform data.
        sampling_rate: Sampling rate.
        pick_sample: P-wave arrival sample.
        signal_window: Signal window in seconds.
        noise_window: Noise window in seconds.

    Returns:
        FeatureMeasurement with SNR value.
    """
    sig_samples = int(signal_window * sampling_rate)
    noise_samples = int(noise_window * sampling_rate)

    # Signal window: from P pick
    sig_start = pick_sample
    sig_end = min(sig_start + sig_samples, len(data))

    # Noise window: before P pick
    noise_end = max(0, pick_sample - int(sampling_rate))  # 1s gap
    noise_start = max(0, noise_end - noise_samples)

    if sig_end - sig_start < int(sampling_rate) or noise_end - noise_start < int(sampling_rate):
        return FeatureMeasurement(
            name="snr", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="rms_ratio"
        )

    signal_rms = np.sqrt(np.mean(data[sig_start:sig_end] ** 2))
    noise_rms = np.sqrt(np.mean(data[noise_start:noise_end] ** 2))

    if noise_rms <= 0:
        return FeatureMeasurement(
            name="snr", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="rms_ratio"
        )

    snr_value = float(signal_rms / noise_rms)
    # Uncertainty estimate (rough)
    uncertainty = snr_value * 0.2  # ~20% relative uncertainty

    if snr_value > 10:
        quality = QUALITY_GOOD
    elif snr_value > 3:
        quality = QUALITY_MARGINAL
    else:
        quality = QUALITY_POOR

    return FeatureMeasurement(
        name="snr", value=snr_value, uncertainty=uncertainty,
        quality=quality, method="rms_ratio"
    )


def compute_duration(
    data: np.ndarray,
    sampling_rate: float,
    pick_sample: int,
    threshold_factor: float = 3.0,
    max_duration: float = 30.0,
) -> FeatureMeasurement:
    """Compute signal duration above noise threshold.

    Duration is measured from P-pick until signal amplitude drops
    below threshold_factor * noise_rms.

    Args:
        data: Waveform data.
        sampling_rate: Sampling rate.
        pick_sample: P-wave arrival sample.
        threshold_factor: Multiple of noise RMS for threshold.
        max_duration: Maximum duration to measure (seconds).

    Returns:
        FeatureMeasurement with duration in seconds.
    """
    noise_samples = int(5.0 * sampling_rate)
    noise_end = max(0, pick_sample - int(sampling_rate))
    noise_start = max(0, noise_end - noise_samples)

    if noise_end - noise_start < int(sampling_rate):
        return FeatureMeasurement(
            name="duration", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="threshold"
        )

    noise_rms = np.sqrt(np.mean(data[noise_start:noise_end] ** 2))
    threshold = threshold_factor * noise_rms

    if threshold <= 0:
        return FeatureMeasurement(
            name="duration", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="threshold"
        )

    # Find where envelope drops below threshold
    max_samples = int(max_duration * sampling_rate)
    end_search = min(pick_sample + max_samples, len(data))
    envelope = np.abs(data[pick_sample:end_search])

    # Use a running average for smoother envelope
    window_size = int(0.5 * sampling_rate)
    if len(envelope) < window_size:
        return FeatureMeasurement(
            name="duration", value=None, uncertainty=None,
            quality=QUALITY_POOR, method="threshold"
        )

    smoothed = np.convolve(envelope, np.ones(window_size)/window_size, mode="valid")
    below_threshold = np.where(smoothed < threshold)[0]

    if len(below_threshold) == 0:
        duration = max_duration
        quality = QUALITY_MARGINAL
    else:
        duration = float(below_threshold[0]) / sampling_rate
        quality = QUALITY_GOOD if duration > 1.0 else QUALITY_MARGINAL

    uncertainty = 0.5  # ~0.5 second uncertainty

    return FeatureMeasurement(
        name="duration", value=duration, uncertainty=uncertainty,
        quality=quality, method="threshold"
    )


def aggregate_feature(
    measurements: List[FeatureMeasurement],
) -> FeatureMeasurement:
    """Aggregate a feature across multiple stations.

    Uses median for robustness. Computes IQR as uncertainty.

    Args:
        measurements: List of measurements from different stations.

    Returns:
        Single aggregated FeatureMeasurement.
    """
    if not measurements:
        return FeatureMeasurement(
            name="unknown", value=None, uncertainty=None,
            quality=QUALITY_UNAVAILABLE
        )

    name = measurements[0].name

    # Filter to good/marginal quality
    valid = [m for m in measurements if m.value is not None and m.quality in (QUALITY_GOOD, QUALITY_MARGINAL)]

    if not valid:
        return FeatureMeasurement(
            name=name, value=None, uncertainty=None,
            quality=QUALITY_UNAVAILABLE, method="aggregated"
        )

    values = [m.value for m in valid]
    median_value = float(np.median(values))

    if len(values) > 1:
        q75, q25 = np.percentile(values, [75, 25])
        uncertainty = float(q75 - q25)
    else:
        uncertainty = valid[0].uncertainty

    if len(valid) >= 5:
        quality = QUALITY_GOOD
    elif len(valid) >= 3:
        quality = QUALITY_MARGINAL
    else:
        quality = QUALITY_POOR

    return FeatureMeasurement(
        name=name, value=median_value, uncertainty=uncertainty,
        quality=quality, method="median_aggregate"
    )
