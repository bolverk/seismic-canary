"""Tests for waveform retrieval and P/S measurement."""
import numpy as np
import pytest
from datetime import datetime, timezone

from src.processing.waveforms import (
    is_eligible_for_waveforms,
    process_waveform,
    aggregate_picks,
    measure_amplitudes,
    assess_quality,
    get_default_picker,
    StaLtaPicker,
    SeisBenchPicker,
    PhasePicker,
    _pick_times_from_model_picks,
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


try:
    import obspy
    HAS_OBSPY = True
except ImportError:
    HAS_OBSPY = False


@pytest.mark.skipif(not HAS_OBSPY, reason="ObsPy not installed")
class TestRetrieveWaveforms:
    """Request-budget behaviour of retrieve_waveforms."""

    def _make_stations(self, n):
        from src.ingestion.stations import StationInfo

        return [
            StationInfo(
                network="XX",
                station=f"S{i:02d}",
                latitude=35.0,
                longitude=51.0,
                elevation_m=0.0,
                description="",
            )
            for i in range(n)
        ]

    def _install_client(self, monkeypatch, calls, open_networks):
        from obspy import Stream
        import obspy.clients.fdsn as fdsn

        class FakeClient:
            def __init__(self, name, *args, **kwargs):
                self.name = name

            def get_waveforms(self, network, station, location, channel, starttime, endtime):
                calls.append(f"{network}.{station}")
                if network in open_networks:
                    from obspy import Trace
                    from obspy.core import Stats

                    stats = Stats()
                    stats.network = network
                    stats.station = station
                    stats.channel = "BHZ"
                    stats.sampling_rate = 1.0
                    stats.npts = 10
                    stats.starttime = starttime
                    tr = Trace(data=np.zeros(10), header=stats)
                    return Stream([tr])
                raise Exception("No data")

        monkeypatch.setattr(fdsn, "Client", FakeClient)
        return FakeClient

    def _event_time(self):
        return datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

    def test_stops_once_target_traces_reached(self, monkeypatch):
        from src.processing.waveforms import retrieve_waveforms

        calls = []
        self._install_client(monkeypatch, calls, open_networks={"XX"})
        stations = self._make_stations(5)

        stream = retrieve_waveforms(
            35.0, 51.0, self._event_time(), stations,
            providers=["IRIS"], target_traces=2,
        )

        assert stream is not None
        assert calls == ["XX.S00", "XX.S01"]

    def test_max_attempts_caps_requests(self, monkeypatch):
        from src.processing.waveforms import retrieve_waveforms

        calls = []
        self._install_client(monkeypatch, calls, open_networks=set())
        stations = self._make_stations(10)

        stream = retrieve_waveforms(
            35.0, 51.0, self._event_time(), stations,
            providers=["IRIS"], max_attempts=3,
        )

        assert stream is None
        assert len(calls) == 3

    def test_max_attempts_bounds_across_providers(self, monkeypatch):
        from src.processing.waveforms import retrieve_waveforms

        calls = []
        self._install_client(monkeypatch, calls, open_networks=set())
        stations = self._make_stations(10)

        retrieve_waveforms(
            35.0, 51.0, self._event_time(), stations,
            providers=["IRIS", "ORFEUS", "GFZ"], max_attempts=4,
        )

        assert len(calls) == 4


@pytest.mark.skipif(not HAS_OBSPY, reason="ObsPy not installed")
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


class TestAssessQuality:
    def test_good(self):
        assert assess_quality(0.3, 8.0) == "good"

    def test_marginal_without_snr(self):
        assert assess_quality(0.3, None) == "marginal"

    def test_marginal_low_snr(self):
        assert assess_quality(0.3, 2.0) == "marginal"

    def test_poor_without_ratio(self):
        assert assess_quality(None, 20.0) == "poor"


class TestGetDefaultPicker:
    def test_forced_sta_lta(self):
        assert isinstance(get_default_picker("sta_lta"), StaLtaPicker)

    def test_forced_seisbench(self):
        assert isinstance(get_default_picker("seisbench"), SeisBenchPicker)

    def test_auto_returns_picker(self):
        assert isinstance(get_default_picker("auto"), PhasePicker)


class _P:
    """Minimal stand-in for a SeisBench pick."""

    def __init__(self, phase, t, v):
        self.phase = phase
        self.peak_time = t
        self.peak_value = v


class TestPickTimesFromModelPicks:
    def test_earliest_p_and_first_post_p_s(self):
        picks = [_P("S", 5, 0.2), _P("P", 10, 0.9), _P("P", 12, 0.5), _P("S", 20, 0.7)]
        p, s, pp, sp = _pick_times_from_model_picks(picks)
        assert (p, s, pp, sp) == (10, 20, 0.9, 0.7)

    def test_s_before_p_ignored(self):
        p, s, pp, sp = _pick_times_from_model_picks([_P("S", 5, 0.2), _P("P", 10, 0.9)])
        assert p == 10 and s is None

    def test_empty(self):
        assert _pick_times_from_model_picks([]) == (None, None, None, None)


@pytest.mark.skipif(not HAS_OBSPY, reason="ObsPy not installed")
class TestMeasureAmplitudes:
    def _trace(self, sr=100.0, n=6000, p_sample=1000, s_sample=2000,
               p_amp=5.0, s_amp=10.0):
        from obspy import Trace, UTCDateTime
        from obspy.core import Stats

        data = np.random.normal(0, 0.01, n)
        data[p_sample] = p_amp
        data[s_sample] = s_amp
        stats = Stats()
        stats.network = "II"
        stats.station = "TEST"
        stats.channel = "BHZ"
        stats.sampling_rate = sr
        stats.npts = n
        stats.starttime = UTCDateTime(2024, 1, 15, 12, 30, 0)
        return Trace(data=data.astype(np.float64), header=stats)

    def test_ratio_and_snr(self):
        tr = self._trace()
        pa, sa, ratio, snr = measure_amplitudes(tr, 1000, 2000)
        assert pa == pytest.approx(5.0)
        assert sa == pytest.approx(10.0)
        assert ratio == pytest.approx(np.log10(0.5))
        assert snr is not None

    def test_short_trace_yields_no_ratio(self):
        tr = self._trace(n=50, p_sample=10, s_sample=20)
        pa, sa, ratio, snr = measure_amplitudes(tr, 40, 60)
        assert sa is None
        assert ratio is None


@pytest.mark.skipif(not HAS_OBSPY, reason="ObsPy not installed")
class TestSeisBenchPicker:
    class _Output:
        def __init__(self, picks):
            self.picks = picks

    class _FakeModel:
        def __init__(self, picks):
            self._picks = picks
            self.calls = 0
            self.kwargs = None

        def classify(self, stream, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            return TestSeisBenchPicker._Output(self._picks)

    def _stream(self, components=("HHZ", "HHN", "HHE")):
        from obspy import Stream, Trace, UTCDateTime
        from obspy.core import Stats

        sr, n = 100.0, 6000
        start = UTCDateTime(2024, 1, 15, 12, 30, 0)
        data = np.zeros(n)
        data[1000] = 5.0   # P at 10 s
        data[2000] = 10.0  # S at 20 s
        traces = []
        for chan in components:
            stats = Stats()
            stats.network, stats.station = "XX", "AAA"
            stats.channel = chan
            stats.sampling_rate, stats.npts = sr, n
            stats.starttime = start
            traces.append(Trace(data=data.copy(), header=stats))
        return Stream(traces), start

    def test_builds_pick_with_ratio(self):
        stream, start = self._stream()
        picks = [_P("P", start + 10, 0.9), _P("S", start + 20, 0.8)]
        picker = SeisBenchPicker(model=self._FakeModel(picks))

        out = picker.pick(stream, start.datetime)

        assert len(out) == 1
        pr = out[0]
        assert pr.station == "AAA" and pr.picker == "seisbench"
        assert pr.p_time == pytest.approx(10.0)
        assert pr.s_time == pytest.approx(20.0)
        assert pr.p_probability == 0.9 and pr.s_probability == 0.8
        assert pr.p_s_ratio == pytest.approx(np.log10(0.5))

    def test_default_defers_to_checkpoint_threshold(self):
        stream, start = self._stream()
        picks = [_P("P", start + 10, 0.9), _P("S", start + 20, 0.8)]
        model = self._FakeModel(picks)

        SeisBenchPicker(model=model).pick(stream, start.datetime)

        # No threshold override: SeisBench would silently ignore a bare
        # "threshold" kwarg and the checkpoint's calibrated default applies.
        assert model.kwargs == {}

    def test_configured_threshold_is_per_phase(self):
        stream, start = self._stream()
        picks = [_P("P", start + 10, 0.9), _P("S", start + 20, 0.8)]
        model = self._FakeModel(picks)

        SeisBenchPicker(model=model, threshold=0.05).pick(stream, start.datetime)

        assert model.kwargs == {"P_threshold": 0.05, "S_threshold": 0.05}

    def test_missing_s_yields_poor_pick(self):
        stream, start = self._stream()
        picker = SeisBenchPicker(model=self._FakeModel([_P("P", start + 10, 0.9)]))
        out = picker.pick(stream, start.datetime)
        assert len(out) == 1
        assert out[0].p_s_ratio is None and out[0].quality == "poor"

    def test_single_component_falls_back_to_stalta(self):
        stream, start = self._stream(components=("HHZ",))
        model = self._FakeModel([_P("P", start + 10, 0.9)])
        picker = SeisBenchPicker(model=model)
        out = picker.pick(stream, start.datetime)
        assert model.calls == 0  # model not invoked without 3 components
        assert len(out) == 1 and out[0].picker == "sta_lta"

    def test_model_load_failure_falls_back(self, monkeypatch):
        stream, start = self._stream()
        picker = SeisBenchPicker(model=None)

        def boom():
            raise RuntimeError("no weights")

        monkeypatch.setattr(picker, "_get_model", boom)
        out = picker.pick(stream, start.datetime)
        # Falls back to STA/LTA without raising
        assert isinstance(out, list)
