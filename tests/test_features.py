"""Tests for spectral analysis and feature extraction."""
import numpy as np
import pytest

from src.processing.features import (
    compute_mb_ms,
    compute_spectral_slope,
    compute_dominant_frequency,
    compute_snr,
    compute_duration,
    aggregate_feature,
    FeatureMeasurement,
    QUALITY_GOOD,
    QUALITY_MARGINAL,
    QUALITY_POOR,
    QUALITY_UNAVAILABLE,
)


class TestMbMs:
    def test_normal_earthquake(self):
        result = compute_mb_ms(mb=4.5, ms=4.3)
        assert result.value == pytest.approx(0.2)
        assert result.quality == QUALITY_GOOD

    def test_explosion_like(self):
        result = compute_mb_ms(mb=4.5, ms=3.0)
        assert result.value == pytest.approx(1.5)

    def test_missing_mb(self):
        result = compute_mb_ms(mb=None, ms=4.0)
        assert result.value is None
        assert result.quality == QUALITY_UNAVAILABLE

    def test_missing_ms(self):
        result = compute_mb_ms(mb=4.5, ms=None)
        assert result.value is None
        assert result.quality == QUALITY_UNAVAILABLE

    def test_both_missing(self):
        result = compute_mb_ms(mb=None, ms=None)
        assert result.quality == QUALITY_UNAVAILABLE


class TestSpectralSlope:
    def _make_signal(self, sr=100.0, freq=5.0, duration=5.0):
        """Create a clean sinusoidal signal."""
        t = np.arange(int(sr * duration)) / sr
        # Signal with known frequency content
        return np.sin(2 * np.pi * freq * t) + 0.01 * np.random.normal(0, 1, len(t))

    def test_clean_signal(self):
        sr = 100.0
        data = self._make_signal(sr=sr)
        result = compute_spectral_slope(data, sr, pick_sample=0)
        assert result.value is not None
        assert result.quality in (QUALITY_GOOD, QUALITY_MARGINAL)

    def test_too_short(self):
        sr = 100.0
        data = np.zeros(50)  # 0.5 seconds
        result = compute_spectral_slope(data, sr, pick_sample=0)
        assert result.quality == QUALITY_POOR

    def test_noise_only(self):
        sr = 100.0
        data = np.random.normal(0, 0.001, int(sr * 5))
        result = compute_spectral_slope(data, sr, pick_sample=0)
        # Should still compute something (flat spectrum)
        assert result.value is not None or result.quality == QUALITY_POOR


class TestDominantFrequency:
    def test_known_frequency(self):
        sr = 100.0
        duration = 5.0
        freq = 5.0
        t = np.arange(int(sr * duration)) / sr
        data = np.sin(2 * np.pi * freq * t)

        result = compute_dominant_frequency(data, sr, pick_sample=0)
        assert result.value is not None
        assert abs(result.value - freq) < 1.0  # within 1 Hz
        assert result.quality == QUALITY_GOOD

    def test_low_frequency(self):
        sr = 100.0
        duration = 10.0
        freq = 1.0
        t = np.arange(int(sr * duration)) / sr
        data = np.sin(2 * np.pi * freq * t)

        result = compute_dominant_frequency(data, sr, pick_sample=0)
        assert result.value is not None
        assert abs(result.value - freq) < 0.5

    def test_too_short(self):
        sr = 100.0
        data = np.zeros(50)
        result = compute_dominant_frequency(data, sr, pick_sample=0)
        assert result.quality == QUALITY_POOR


class TestSNR:
    def test_high_snr(self):
        sr = 100.0
        # 10 seconds noise + 5 seconds signal
        noise = np.random.normal(0, 0.01, int(10 * sr))
        signal = np.sin(2 * np.pi * 5.0 * np.arange(int(5 * sr)) / sr) * 1.0
        data = np.concatenate([noise, signal])

        pick_sample = int(10 * sr)
        result = compute_snr(data, sr, pick_sample)

        assert result.value is not None
        assert result.value > 10  # should be high SNR
        assert result.quality == QUALITY_GOOD

    def test_low_snr(self):
        sr = 100.0
        # Noise level similar to signal
        data = np.random.normal(0, 1.0, int(15 * sr))
        # Small signal added at pick
        pick_sample = int(10 * sr)
        data[pick_sample:pick_sample + int(3*sr)] += 0.5

        result = compute_snr(data, sr, pick_sample)
        assert result.value is not None
        assert result.value < 5

    def test_too_short_noise_window(self):
        sr = 100.0
        data = np.ones(int(2 * sr))  # only 2 seconds
        result = compute_snr(data, sr, pick_sample=10)
        assert result.quality == QUALITY_POOR


class TestDuration:
    def test_impulsive_signal(self):
        sr = 100.0
        # Noise background
        data = np.random.normal(0, 0.01, int(30 * sr))
        # Add impulsive signal at sample 1000 that decays quickly
        pick = 1000
        t = np.arange(int(2 * sr)) / sr
        impulse = np.exp(-3 * t) * np.sin(2 * np.pi * 5 * t) * 5.0
        data[pick:pick+len(impulse)] += impulse

        result = compute_duration(data, sr, pick_sample=pick)
        assert result.value is not None
        assert result.value > 0
        assert result.value < 10  # should be short duration

    def test_insufficient_data(self):
        sr = 100.0
        data = np.zeros(50)
        result = compute_duration(data, sr, pick_sample=40)
        assert result.quality == QUALITY_POOR


class TestAggregateFeature:
    def test_empty(self):
        result = aggregate_feature([])
        assert result.quality == QUALITY_UNAVAILABLE

    def test_single_measurement(self):
        measurements = [
            FeatureMeasurement(name="snr", value=10.0, uncertainty=2.0, quality=QUALITY_GOOD)
        ]
        result = aggregate_feature(measurements)
        assert result.value == 10.0
        assert result.quality == QUALITY_POOR  # only 1 station

    def test_multiple_measurements(self):
        measurements = [
            FeatureMeasurement(name="snr", value=8.0, uncertainty=1.0, quality=QUALITY_GOOD),
            FeatureMeasurement(name="snr", value=10.0, uncertainty=1.5, quality=QUALITY_GOOD),
            FeatureMeasurement(name="snr", value=12.0, uncertainty=2.0, quality=QUALITY_GOOD),
            FeatureMeasurement(name="snr", value=9.0, uncertainty=1.0, quality=QUALITY_MARGINAL),
            FeatureMeasurement(name="snr", value=11.0, uncertainty=1.5, quality=QUALITY_GOOD),
        ]
        result = aggregate_feature(measurements)
        assert result.value == pytest.approx(10.0)  # median
        assert result.quality == QUALITY_GOOD  # 5 stations
        assert result.uncertainty is not None

    def test_poor_excluded(self):
        measurements = [
            FeatureMeasurement(name="snr", value=10.0, uncertainty=1.0, quality=QUALITY_GOOD),
            FeatureMeasurement(name="snr", value=999.0, uncertainty=0.0, quality=QUALITY_POOR),
            FeatureMeasurement(name="snr", value=None, uncertainty=None, quality=QUALITY_UNAVAILABLE),
        ]
        result = aggregate_feature(measurements)
        assert result.value == 10.0  # only the good one
