"""Waveform retrieval and P/S amplitude measurement.

Retrieves seismic waveform windows from FDSN services for selected events,
performs instrument correction, filtering, automatic phase picking,
and computes P/S amplitude ratios.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

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

    for provider_name in providers:
        try:
            client = Client(provider_name)
        except Exception as e:
            logger.warning(f"Cannot create client for {provider_name}: {e}")
            continue

        for station in stations:
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
                    logger.debug(
                        f"Retrieved {len(st)} traces from "
                        f"{station.network}.{station.station} via {provider_name}"
                    )
            except Exception as e:
                logger.debug(
                    f"No data from {station.network}.{station.station} "
                    f"via {provider_name}: {e}"
                )
                continue

    return all_traces if len(all_traces) > 0 else None


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

        # Measure P amplitude (max absolute in P window)
        p_start = p_sample
        p_end = min(p_sample + int(p_window * sr), len(data))
        if p_end > p_start:
            p_amp = np.max(np.abs(data[p_start:p_end]))
            result.p_amplitude = float(p_amp)

        # Measure S amplitude (max absolute in S window)
        s_start = s_sample
        s_end = min(s_sample + int(s_window * sr), len(data))
        if s_end > s_start and s_start < len(data):
            s_amp = np.max(np.abs(data[s_start:s_end]))
            result.s_amplitude = float(s_amp)

        # Compute P/S ratio
        if result.p_amplitude and result.s_amplitude and result.s_amplitude > 0:
            result.p_s_ratio = float(np.log10(result.p_amplitude / result.s_amplitude))

        # Compute SNR (signal = P window, noise = pre-P window)
        noise_end = max(0, p_sample - int(1.0 * sr))
        noise_start = max(0, noise_end - int(5.0 * sr))
        if noise_end > noise_start:
            noise_rms = np.sqrt(np.mean(data[noise_start:noise_end] ** 2))
            if noise_rms > 0:
                signal_rms = np.sqrt(np.mean(data[p_start:p_end] ** 2))
                result.snr = float(signal_rms / noise_rms)

        # Quality assessment
        if result.p_s_ratio is not None and result.snr is not None and result.snr > 5:
            result.quality = "good"
        elif result.p_s_ratio is not None:
            result.quality = "marginal"
        else:
            result.quality = "poor"

    except Exception as e:
        logger.warning(f"Error processing {trace.id}: {e}")
        result.quality = "poor"

    return result


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
) -> WaveformResult:
    """Full waveform analysis pipeline for a single event.

    Args:
        event_id: Event identifier.
        event_lat: Event latitude.
        event_lon: Event longitude.
        event_time: Event origin time.
        stations: Stations to use.

    Returns:
        WaveformResult with all measurements.
    """
    result = WaveformResult(event_id=event_id)

    # Retrieve waveforms
    stream = retrieve_waveforms(event_lat, event_lon, event_time, stations)

    if stream is None:
        result.errors.append("No waveform data retrieved")
        return result

    result.station_count = len(set(tr.stats.station for tr in stream))

    # Process each trace (use vertical component preferentially)
    processed_stations = set()
    picks = []

    for trace in stream:
        station_key = f"{trace.stats.network}.{trace.stats.station}"
        # Prefer Z component, skip if already processed
        if station_key in processed_stations:
            if not trace.stats.channel.endswith("Z"):
                continue
        processed_stations.add(station_key)

        # Only process Z (vertical) component for P/S
        if trace.stats.channel.endswith("Z"):
            pick = process_waveform(trace, event_time)
            picks.append(pick)

    # Aggregate
    agg = aggregate_picks(picks)
    result.successful_stations = agg.successful_stations
    result.median_p_s_ratio = agg.median_p_s_ratio
    result.mean_p_s_ratio = agg.mean_p_s_ratio
    result.p_s_ratio_std = agg.p_s_ratio_std
    result.quality = agg.quality
    result.picks = agg.picks

    return result
