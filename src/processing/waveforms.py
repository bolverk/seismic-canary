"""Waveform retrieval and P/S amplitude measurement.

Retrieves seismic waveform windows from FDSN services for selected events,
performs instrument correction, filtering, automatic phase picking,
and computes P/S amplitude ratios.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from src.config import Config
from src.ingestion.stations import StationInfo

logger = logging.getLogger(__name__)


@dataclass
class PickResult:
    """Result of phase picking on a single trace."""
    station: str
    network: str
    channel: str
    p_time: Optional[float] = None  # seconds from trace start
    s_time: Optional[float] = None
    p_amplitude: Optional[float] = None
    s_amplitude: Optional[float] = None
    p_s_ratio: Optional[float] = None  # log10(P_amp / S_amp)
    snr: Optional[float] = None
    quality: str = "unavailable"  # good, marginal, poor, unavailable
    p_probability: Optional[float] = None  # picker confidence in P onset
    s_probability: Optional[float] = None  # picker confidence in S onset
    picker: str = "sta_lta"  # which picker produced the onsets


@dataclass
class WaveformResult:
    """Aggregated waveform analysis result for an event."""
    event_id: str
    station_count: int = 0
    successful_stations: int = 0
    median_p_s_ratio: Optional[float] = None
    mean_p_s_ratio: Optional[float] = None
    p_s_ratio_std: Optional[float] = None
    quality: str = "unavailable"
    picks: List[PickResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def is_eligible_for_waveforms(
    magnitude: Optional[float],
    depth_km: Optional[float],
    min_magnitude: float = Config.WAVEFORM_MIN_MAGNITUDE,
    shallow_threshold: float = Config.WAVEFORM_SHALLOW_DEPTH_KM,
) -> bool:
    """Determine if an event is eligible for waveform retrieval.

    Criteria:
    - magnitude >= min_magnitude, OR
    - depth < shallow_threshold km

    Args:
        magnitude: Event magnitude.
        depth_km: Event depth in km.
        min_magnitude: Minimum magnitude threshold.
        shallow_threshold: Shallow depth threshold in km.

    Returns:
        True if waveform retrieval should be attempted.
    """
    if magnitude is not None and magnitude >= min_magnitude:
        return True
    if depth_km is not None and depth_km < shallow_threshold:
        return True
    return False


def retrieve_waveforms(
    event_lat: float,
    event_lon: float,
    event_time: datetime,
    stations: List[StationInfo],
    pre_seconds: float = Config.FDSN_WAVEFORM_PRE_SECONDS,
    post_seconds: float = Config.FDSN_WAVEFORM_POST_SECONDS,
    providers: Optional[List[str]] = None,
    target_traces: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> Optional[object]:
    """Retrieve waveform data from FDSN for given stations.

    Args:
        event_lat: Event latitude.
        event_lon: Event longitude.
        event_time: Event origin time.
        stations: List of stations to query.
        pre_seconds: Seconds before event to retrieve.
        post_seconds: Seconds after event to retrieve.
        providers: FDSN provider names.
        target_traces: Stop querying once this many distinct stations have
            returned data (bounds the number of HTTP requests per event).
        max_attempts: Hard cap on the total number of waveform requests made
            across all providers and stations, so an event with no open data
            cannot trigger an unbounded crawl of the candidate pool.

    Returns:
        ObsPy Stream object with waveforms, or None on failure.
    """
    try:
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime, Stream
    except ImportError:
        logger.error("ObsPy not available.")
        return None

    if providers is None:
        providers = Config.FDSN_PROVIDERS

    starttime = UTCDateTime(event_time) - pre_seconds
    endtime = UTCDateTime(event_time) + post_seconds

    all_traces = Stream()
    stations_with_data = set()
    attempts = 0

    for provider_name in providers:
        if target_traces and len(stations_with_data) >= target_traces:
            break
        if max_attempts is not None and attempts >= max_attempts:
            break
        try:
            client = Client(provider_name)
        except Exception as e:
            logger.warning(f"Cannot create client for {provider_name}: {e}")
            continue

        for station in stations:
            if max_attempts is not None and attempts >= max_attempts:
                break
            attempts += 1
            try:
                st = client.get_waveforms(
                    network=station.network,
                    station=station.station,
                    location="*",
                    channel="BH?,HH?",
                    starttime=starttime,
                    endtime=endtime,
                )
                if st:
                    all_traces += st
                    stations_with_data.add(f"{station.network}.{station.station}")
                    logger.debug(
                        f"Retrieved {len(st)} traces from "
                        f"{station.network}.{station.station} via {provider_name}"
                    )
                    if target_traces and len(stations_with_data) >= target_traces:
                        break
            except Exception as e:
                logger.debug(
                    f"No data from {station.network}.{station.station} "
                    f"via {provider_name}: {e}"
                )
                continue

    return all_traces if len(all_traces) > 0 else None


def measure_amplitudes(
    trace,
    p_sample: int,
    s_sample: int,
    p_window: float = Config.P_WINDOW_SECONDS,
    s_window: float = Config.S_WINDOW_SECONDS,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Measure P/S amplitudes, their log10 ratio, and SNR from two onsets.

    Args:
        trace: ObsPy Trace (already filtered) with the phases present.
        p_sample: P onset sample index.
        s_sample: S onset sample index.
        p_window: P measurement window (seconds).
        s_window: S measurement window (seconds).

    Returns:
        Tuple of (p_amplitude, s_amplitude, p_s_ratio, snr); any may be None.
    """
    data = trace.data
    sr = trace.stats.sampling_rate

    p_amplitude: Optional[float] = None
    s_amplitude: Optional[float] = None
    p_s_ratio: Optional[float] = None
    snr: Optional[float] = None

    p_start = max(0, p_sample)
    p_end = min(p_sample + int(p_window * sr), len(data))
    if p_end > p_start:
        p_amplitude = float(np.max(np.abs(data[p_start:p_end])))

    s_start = max(0, s_sample)
    s_end = min(s_sample + int(s_window * sr), len(data))
    if s_end > s_start and s_start < len(data):
        s_amplitude = float(np.max(np.abs(data[s_start:s_end])))

    if p_amplitude and s_amplitude and s_amplitude > 0:
        p_s_ratio = float(np.log10(p_amplitude / s_amplitude))

    noise_end = max(0, p_sample - int(1.0 * sr))
    noise_start = max(0, noise_end - int(5.0 * sr))
    if noise_end > noise_start and p_end > p_start:
        noise_rms = np.sqrt(np.mean(data[noise_start:noise_end] ** 2))
        if noise_rms > 0:
            signal_rms = np.sqrt(np.mean(data[p_start:p_end] ** 2))
            snr = float(signal_rms / noise_rms)

    return p_amplitude, s_amplitude, p_s_ratio, snr


def assess_quality(p_s_ratio: Optional[float], snr: Optional[float]) -> str:
    """Map a P/S ratio and SNR to a pick-quality label."""
    if p_s_ratio is not None and snr is not None and snr > 5:
        return "good"
    if p_s_ratio is not None:
        return "marginal"
    return "poor"


def process_waveform(
    trace,
    event_time: datetime,
    freqmin: float = Config.BANDPASS_FREQMIN,
    freqmax: float = Config.BANDPASS_FREQMAX,
    p_window: float = Config.P_WINDOW_SECONDS,
    s_window: float = Config.S_WINDOW_SECONDS,
) -> PickResult:
    """Process a single waveform trace to extract P/S measurements.

    Processing steps:
    1. Instrument response removal (if available)
    2. Bandpass filter
    3. P-wave picking (STA/LTA)
    4. S-wave picking
    5. Amplitude measurements
    6. P/S ratio computation

    Args:
        trace: ObsPy Trace object.
        event_time: Event origin time for reference.
        freqmin: Minimum bandpass frequency (Hz).
        freqmax: Maximum bandpass frequency (Hz).
        p_window: P-wave measurement window (seconds).
        s_window: S-wave measurement window (seconds).

    Returns:
        PickResult with measurements.
    """
    from obspy import UTCDateTime
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset

    result = PickResult(
        station=trace.stats.station,
        network=trace.stats.network,
        channel=trace.stats.channel,
    )

    try:
        # Make a copy to avoid modifying original
        tr = trace.copy()

        # Remove instrument response if available
        try:
            tr.remove_response(output="VEL")
        except Exception:
            # Response not attached; work with raw data
            pass

        # Detrend and bandpass filter
        tr.detrend("demean")
        tr.detrend("linear")
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)

        data = tr.data
        sr = tr.stats.sampling_rate

        if len(data) < int(sr * 10):
            result.quality = "poor"
            return result

        # STA/LTA for P-wave detection
        sta_len = int(1.0 * sr)  # 1 second STA
        lta_len = int(10.0 * sr)  # 10 second LTA

        if len(data) < lta_len + sta_len:
            result.quality = "poor"
            return result

        cft = recursive_sta_lta(data, sta_len, lta_len)

        # Trigger thresholds
        triggers = trigger_onset(cft, 3.5, 1.0)

        if len(triggers) == 0:
            result.quality = "poor"
            return result

        # First trigger is likely P-wave
        p_sample = triggers[0][0]
        p_time_sec = p_sample / sr
        result.p_time = p_time_sec

        # S-wave: look for second trigger or estimate from P
        if len(triggers) > 1:
            s_sample = triggers[1][0]
        else:
            # Rough estimate: S arrives ~1.73x P travel time
            event_utc = UTCDateTime(event_time)
            trace_start = tr.stats.starttime
            p_travel = p_time_sec - (event_utc - trace_start)
            if p_travel > 0:
                s_travel = p_travel * 1.73
                s_sample = int((s_travel + (event_utc - trace_start)) * sr)
            else:
                s_sample = p_sample + int(p_window * sr * 2)

        s_time_sec = s_sample / sr
        result.s_time = s_time_sec

        # Amplitude, P/S ratio and SNR from the two onsets
        result.p_amplitude, result.s_amplitude, result.p_s_ratio, result.snr = \
            measure_amplitudes(tr, p_sample, s_sample, p_window, s_window)
        result.quality = assess_quality(result.p_s_ratio, result.snr)

    except Exception as e:
        logger.warning(f"Error processing {trace.id}: {e}")
        result.quality = "poor"

    return result


_SEISBENCH_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def seisbench_available() -> bool:
    """Whether the optional SeisBench + torch dependencies are importable."""
    try:
        import seisbench  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _load_seisbench_model(model_name: str, checkpoint: str):
    """Load and cache a pretrained SeisBench phase-picking model."""
    key = (model_name, checkpoint)
    if key not in _SEISBENCH_MODEL_CACHE:
        import seisbench.models as sbm

        model_cls = getattr(sbm, model_name)
        model = model_cls.from_pretrained(checkpoint)
        model.eval()
        _SEISBENCH_MODEL_CACHE[key] = model
    return _SEISBENCH_MODEL_CACHE[key]


def _group_by_station(stream) -> Dict[str, Any]:
    """Group a Stream into per-station sub-Streams keyed by ``NET.STA``."""
    from obspy import Stream

    groups: Dict[str, Stream] = {}
    for tr in stream:
        key = f"{tr.stats.network}.{tr.stats.station}"
        groups.setdefault(key, Stream()).append(tr)
    return groups


def _pick_times_from_model_picks(model_picks) -> Tuple[Any, Any, Optional[float], Optional[float]]:
    """Extract the first P and the first post-P S onset from model picks.

    Returns (p_time, s_time, p_probability, s_probability); times are absolute
    (ObsPy UTCDateTime) and probabilities are floats or None.
    """
    p_time = s_time = None
    p_prob = s_prob = None
    ordered = sorted(model_picks, key=lambda p: p.peak_time)

    for pk in ordered:
        if str(getattr(pk, "phase", "")).upper() == "P":
            p_time = pk.peak_time
            p_prob = float(pk.peak_value)
            break

    for pk in ordered:
        if str(getattr(pk, "phase", "")).upper() == "S" and (
            p_time is None or pk.peak_time > p_time
        ):
            s_time = pk.peak_time
            s_prob = float(pk.peak_value)
            break

    return p_time, s_time, p_prob, s_prob


class PhasePicker:
    """Strategy for locating P/S onsets on retrieved waveforms."""

    name = "base"

    def pick(self, stream, event_time) -> List[PickResult]:
        raise NotImplementedError


class StaLtaPicker(PhasePicker):
    """Classic per-trace STA/LTA picker (vertical component only)."""

    name = "sta_lta"

    def pick(self, stream, event_time) -> List[PickResult]:
        picks = []
        processed = set()
        for trace in stream:
            key = f"{trace.stats.network}.{trace.stats.station}"
            if key in processed and not trace.stats.channel.endswith("Z"):
                continue
            processed.add(key)
            if trace.stats.channel.endswith("Z"):
                pick = process_waveform(trace, event_time)
                pick.picker = self.name
                picks.append(pick)
        return picks


class SeisBenchPicker(PhasePicker):
    """Pretrained deep-learning phase picker (e.g. EQTransformer via SeisBench).

    Consumes three-component station waveforms and returns P and S onset
    times with confidences. Stations without a full three-component set (or
    when the model cannot be loaded) fall back to the STA/LTA picker so no
    station is silently dropped.

    Detection thresholds default to the checkpoint's calibrated values
    (``Config.SEISBENCH_PICK_THRESHOLD = None``); overriding them with a
    generic probability cut-off discards most valid regional picks.
    """

    name = "seisbench"

    def __init__(
        self,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        threshold: Optional[float] = None,
        model: Any = None,
    ):
        self.model_name = model_name or Config.SEISBENCH_MODEL
        self.checkpoint = checkpoint or Config.SEISBENCH_CHECKPOINT
        self.threshold = (
            Config.SEISBENCH_PICK_THRESHOLD if threshold is None else threshold
        )
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = _load_seisbench_model(self.model_name, self.checkpoint)
        return self._model

    def _classify_kwargs(self) -> dict:
        """Keyword arguments for ``model.classify``.

        SeisBench takes per-phase thresholds as ``P_threshold`` / ``S_threshold``
        (a bare ``threshold`` is silently ignored). With no configured value we
        pass nothing so the checkpoint's calibrated ``default_args`` apply.
        """
        if self.threshold is None:
            return {}
        return {"P_threshold": self.threshold, "S_threshold": self.threshold}

    def pick(self, stream, event_time) -> List[PickResult]:
        try:
            model = self._get_model()
        except Exception as e:  # pragma: no cover - depends on environment
            logger.warning(f"SeisBench model unavailable ({e}); using STA/LTA.")
            return StaLtaPicker().pick(stream, event_time)

        picks: List[PickResult] = []
        for key, sub in _group_by_station(stream).items():
            vertical = [tr for tr in sub if tr.stats.channel.endswith("Z")]
            if len(sub) < 3 or not vertical:
                for tr in vertical:
                    pick = process_waveform(tr, event_time)
                    pick.picker = StaLtaPicker.name
                    picks.append(pick)
                continue

            try:
                output = model.classify(sub, **self._classify_kwargs())
                model_picks = list(output.picks)
            except Exception as e:
                logger.debug(f"SeisBench classify failed for {key}: {e}")
                model_picks = []

            p_time, s_time, p_prob, s_prob = _pick_times_from_model_picks(model_picks)
            picks.append(
                self._build_pick(vertical[0], p_time, s_time, p_prob, s_prob)
            )
        return picks

    def _build_pick(self, z_trace, p_time, s_time, p_prob, s_prob) -> PickResult:
        result = PickResult(
            station=z_trace.stats.station,
            network=z_trace.stats.network,
            channel=z_trace.stats.channel,
            p_probability=p_prob,
            s_probability=s_prob,
            picker=self.name,
        )

        tr = z_trace.copy()
        try:
            tr.detrend("demean")
            tr.detrend("linear")
            tr.filter(
                "bandpass",
                freqmin=Config.BANDPASS_FREQMIN,
                freqmax=Config.BANDPASS_FREQMAX,
                zerophase=True,
            )
        except Exception:
            pass

        sr = tr.stats.sampling_rate
        if p_time is not None:
            result.p_time = float(p_time - tr.stats.starttime)
        if s_time is not None:
            result.s_time = float(s_time - tr.stats.starttime)

        if result.p_time is not None and result.s_time is not None:
            p_sample = int(result.p_time * sr)
            s_sample = int(result.s_time * sr)
            result.p_amplitude, result.s_amplitude, result.p_s_ratio, result.snr = \
                measure_amplitudes(tr, p_sample, s_sample)
            result.quality = assess_quality(result.p_s_ratio, result.snr)
        else:
            result.quality = "poor"

        return result


def get_default_picker(preference: Optional[str] = None) -> PhasePicker:
    """Select a phase picker from config (``auto`` | ``seisbench`` | ``sta_lta``)."""
    pref = (preference or Config.WAVEFORM_PICKER or "auto").lower()
    if pref == "sta_lta":
        return StaLtaPicker()
    if pref == "seisbench":
        return SeisBenchPicker()
    return SeisBenchPicker() if seisbench_available() else StaLtaPicker()


def aggregate_picks(picks: List[PickResult]) -> WaveformResult:
    """Aggregate P/S measurements across multiple stations.

    Args:
        picks: List of PickResult from individual stations.

    Returns:
        WaveformResult with aggregated statistics.
    """
    result = WaveformResult(event_id="", station_count=len(picks))

    good_picks = [p for p in picks if p.p_s_ratio is not None and p.quality in ("good", "marginal")]
    result.successful_stations = len(good_picks)
    result.picks = picks

    if not good_picks:
        result.quality = "unavailable"
        return result

    ratios = [p.p_s_ratio for p in good_picks]
    result.median_p_s_ratio = float(np.median(ratios))
    result.mean_p_s_ratio = float(np.mean(ratios))

    if len(ratios) > 1:
        result.p_s_ratio_std = float(np.std(ratios))

    # Quality assessment
    if result.successful_stations >= 5:
        result.quality = "good"
    elif result.successful_stations >= 3:
        result.quality = "marginal"
    else:
        result.quality = "poor"

    return result


def analyze_event_waveforms(
    event_id: str,
    event_lat: float,
    event_lon: float,
    event_time: datetime,
    stations: List[StationInfo],
    target_traces: Optional[int] = None,
    picker: Optional[PhasePicker] = None,
) -> WaveformResult:
    """Full waveform analysis pipeline for a single event.

    Args:
        event_id: Event identifier.
        event_lat: Event latitude.
        event_lon: Event longitude.
        event_time: Event origin time.
        stations: Stations to use.
        target_traces: Stop downloading once this many distinct stations have
            returned data.
        picker: Phase picker to use; defaults to the configured picker
            (SeisBench when available, otherwise STA/LTA).

    Returns:
        WaveformResult with all measurements.
    """
    result = WaveformResult(event_id=event_id)

    # Retrieve waveforms
    stream = retrieve_waveforms(
        event_lat,
        event_lon,
        event_time,
        stations,
        target_traces=target_traces,
        max_attempts=Config.WAVEFORM_MAX_ATTEMPTS,
    )

    if stream is None:
        result.errors.append("No waveform data retrieved")
        return result

    result.station_count = len(set(tr.stats.station for tr in stream))

    # Pick P/S onsets and measure amplitudes/P/S ratio per station
    picker = picker or get_default_picker()
    picks = picker.pick(stream, event_time)

    # Aggregate
    agg = aggregate_picks(picks)
    result.successful_stations = agg.successful_stations
    result.median_p_s_ratio = agg.median_p_s_ratio
    result.mean_p_s_ratio = agg.mean_p_s_ratio
    result.p_s_ratio_std = agg.p_s_ratio_std
    result.quality = agg.quality
    result.picks = agg.picks

    return result
