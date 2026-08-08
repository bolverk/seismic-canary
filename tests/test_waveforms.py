"""Tests for waveform retrieval and P/S measurement."""
import numpy as np
import pytest
from datetime import datetime, timezone

from src.processing.waveforms import (
    is_eligible_for_waveforms,
    process_waveform,
    aggregate_picks,
    PickResult,
    WaveformResult,
)


class TestEligibility:
    def test_above_magnitude_threshold(self):
        assert is_eligible_for_waveforms(magnitude=4.0, depth_km=15.0) is True

    def test_below_magnitude_threshold(self):
        assert is_eligible_for_waveforms(magnitude=2.0, depth_km=15.0) is False

    def test_shallow_depth(self):
        assert is_eligible_for_waveforms(magnitude=2.0, depth_km=2.0) is True

    def test_none_magnitude_shallow(self):
        assert is_eligible_for_waveforms(magnitude=None, depth_km=1.0) is True

    def test_none_magnitude_deep(self):
        assert is_eligible_for_waveforms(magnitude=None, depth_km=20.0) is False

    def test_both_none(self):
        assert is_eligible_for_waveforms(magnitude=None, depth_km=None) is False

    def test_exact_threshold(self):
        assert is_eligible_for_waveforms(magnitude=3.0, depth_km=15.0) is True

    def test_custom_thresholds(self):
        assert is_eligible_for_waveforms(
            magnitude=2.5, depth_km=15.0, min_magnitude=2.0
        ) is True


class TestProcessWaveform:
    def _make_synthetic_trace(self, sr=100.0, duration=60.0, p_time=10.0, s_time=20.0):
        """Create a synthetic trace with P and S arrivals."""
        from obspy import Trace, UTCDateTime
        from obspy.core import Stats

        npts = int(sr * duration)
        t = np.arange(npts) / sr

        # Start with noise
        data = np.random.normal(0, 0.01, npts)

        # Add P arrival (impulsive)
        p_sample = int(p_time * sr)
        p_signal = np.exp(-2.0 * (t[p_sample:p_sample+int(3*sr)] - t[p_sample])) * \
                   np.sin(2 * np.pi * 5.0 * (t[p_sample:p_sample+int(3*sr)] - t[p_sample]))
        data[p_sample:p_sample+len(p_signal)] += p_signal * 5.0

        # Add S arrival (larger amplitude, lower frequency)
        s_sample = int(s_time * sr)
        remaining = min(int(5*sr), npts - s_sample)
        s_signal = np.exp(-1.0 * (t[s_sample:s_sample+remaining] - t[s_sample])) * \
                   np.sin(2 * np.pi * 2.0 * (t[s_sample:s_sample+remaining] - t[s_sample]))
        data[s_sample:s_sample+remaining] += s_signal * 10.0

        stats = Stats()
        stats.network = "II"
        stats.station = "TEST"
        stats.channel = "BHZ"
        stats.sampling_rate = sr
        stats.npts = npts
        stats.starttime = UTCDateTime(2024, 1, 15, 12, 30, 0)

        tr = Trace(data=data.astype(np.float64), header=stats)
        return tr

    def test_synthetic_trace_processing(self):
        """Process a synthetic trace and verify P/S ratio is computed."""
        tr = self._make_synthetic_trace()
        event_time = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

        result = process_waveform(tr, event_time)

        assert result.station == "TEST"
        assert result.network == "II"
        assert result.channel == "BHZ"
        # Should detect something (P pick)
        assert result.p_time is not None
        assert result.quality != "unavailable"

    def test_noise_only_trace(self):
        """A noise-only trace should get poor quality."""
        from obspy import Trace, UTCDateTime
        from obspy.core import Stats

        data = np.random.normal(0, 0.001, 6000)
        stats = Stats()
        stats.network = "II"
        stats.station = "NOISE"
        stats.channel = "BHZ"
        stats.sampling_rate = 100.0
        stats.npts = 6000
        stats.starttime = UTCDateTime(2024, 1, 15, 12, 30, 0)

        tr = Trace(data=data.astype(np.float64), header=stats)
        event_time = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

        result = process_waveform(tr, event_time)
        assert result.quality in ("poor", "marginal")

    def test_short_trace(self):
        """A too-short trace should return poor quality."""
        from obspy import Trace, UTCDateTime
        from obspy.core import Stats

        data = np.random.normal(0, 0.01, 500)  # 5 seconds at 100Hz
        stats = Stats()
        stats.network = "II"
        stats.station = "SHORT"
        stats.channel = "BHZ"
        stats.sampling_rate = 100.0
        stats.npts = 500
        stats.starttime = UTCDateTime(2024, 1, 15, 12, 30, 0)

        tr = Trace(data=data.astype(np.float64), header=stats)
        event_time = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

        result = process_waveform(tr, event_time)
        assert result.quality == "poor"


class TestAggregatePicks:
    def test_no_picks(self):
        result = aggregate_picks([])
        assert result.station_count == 0
        assert result.quality == "unavailable"
        assert result.median_p_s_ratio is None

    def test_single_good_pick(self):
        picks = [PickResult(station="S1", network="II", channel="BHZ",
                           p_s_ratio=0.5, quality="good")]
        result = aggregate_picks(picks)
        assert result.successful_stations == 1
        assert result.median_p_s_ratio == 0.5

    def test_multiple_picks(self):
        picks = [
            PickResult(station="S1", network="II", channel="BHZ",
                      p_s_ratio=0.3, quality="good"),
            PickResult(station="S2", network="II", channel="BHZ",
                      p_s_ratio=0.5, quality="good"),
            PickResult(station="S3", network="II", channel="BHZ",
                      p_s_ratio=0.7, quality="marginal"),
        ]
        result = aggregate_picks(picks)
        assert result.successful_stations == 3
        assert result.median_p_s_ratio == pytest.approx(0.5)
        assert result.p_s_ratio_std is not None

    def test_poor_picks_excluded(self):
        picks = [
            PickResult(station="S1", network="II", channel="BHZ",
                      p_s_ratio=0.5, quality="good"),
            PickResult(station="S2", network="II", channel="BHZ",
                      p_s_ratio=None, quality="poor"),
        ]
        result = aggregate_picks(picks)
        assert result.successful_stations == 1
        assert result.median_p_s_ratio == 0.5

    def test_quality_thresholds(self):
        """5+ stations = good quality."""
        picks = [
            PickResult(station=f"S{i}", network="II", channel="BHZ",
                      p_s_ratio=0.4 + i*0.1, quality="good")
            for i in range(5)
        ]
        result = aggregate_picks(picks)
        assert result.quality == "good"

    def test_partial_failure(self):
        """Some stations succeed, some fail - still produces results."""
        picks = [
            PickResult(station="S1", network="II", channel="BHZ",
                      p_s_ratio=0.5, quality="good"),
            PickResult(station="S2", network="II", channel="BHZ",
                      p_s_ratio=None, quality="poor"),
            PickResult(station="S3", network="II", channel="BHZ",
                      p_s_ratio=0.6, quality="good"),
            PickResult(station="S4", network="II", channel="BHZ",
                      p_s_ratio=None, quality="poor"),
        ]
        result = aggregate_picks(picks)
        assert result.successful_stations == 2
        assert result.median_p_s_ratio is not None
        assert result.quality == "poor"  # only 2 successful < 3
