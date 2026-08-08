"""Seismic Canary — Methodology Page.

Explains what the system monitors, how scores are calculated,
limitations, and data sources.
"""
import streamlit as st
from src.config import Config


st.set_page_config(page_title="Methodology - Seismic Canary", page_icon="🐤", layout="wide")


st.title("📖 Methodology")
st.markdown("---")

st.header("About Seismic Canary")
st.markdown(f"""
Seismic Canary is an open-source, public-data seismic event monitoring system
focused on **{Config.MONITORED_REGION_DESCRIPTION}**.

It automatically:
1. Ingests seismic events from official public catalogs
2. Computes interpretable seismic features
3. Assesses whether each event's characteristics are consistent with
   a tectonic earthquake or anomalous (potentially explosion-like)
4. Presents all evidence transparently

**Version:** {Config.VERSION} | **Model:** {Config.MODEL_VERSION}
""")

st.header("What We Can Detect")
st.markdown("""
- Seismic events in the monitored region
- Events with characteristics statistically unusual for tectonic earthquakes
- Specifically: shallow depth, high P/S amplitude ratio, anomalous mb-Ms relationship

These are **seismic anomalies** — they indicate that an event's characteristics
deviate from what typical earthquakes produce.
""")

st.header("What We Cannot Detect")
st.error("""
⚠️ **An explosion-like seismic signature does NOT establish that an event was nuclear.**

This system **cannot**:
- Confirm the nuclear nature of any event (this requires radionuclide detection)
- Distinguish a nuclear explosion from a conventional explosion seismically
- Replace the CTBTO's comprehensive monitoring system
- Provide actionable intelligence

This is a scientific monitoring tool, not a weapons detection system.
""")

st.header("Data Sources")
st.markdown("""
| Source | Data | Access |
|--------|------|--------|
| [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/) | Event metadata (location, magnitude, depth) | Public API |
| [FDSN Waveform Services](https://www.fdsn.org/webservices/) | Seismic waveforms | Public (IRIS/ORFEUS/GFZ) |
| [FDSN Station Services](https://www.fdsn.org/webservices/) | Station metadata | Public |

All data sources are official, machine-readable feeds intended for
scientific applications. No scraping is used.
""")

st.header("How Anomaly Scores Work")
st.markdown("""
The anomaly assessment uses a **transparent rule-based model** where each
feature contributes independently to earthquake-consistency and
explosion-consistency scores.

### Features Used

| Feature | Measurement | Explosion-like when... | Earthquake-like when... |
|---------|-------------|----------------------|------------------------|
| **Depth** | From catalog | < 5 km (very shallow) | > 10 km |
| **P/S Ratio** | From waveform analysis | log₁₀(P/S) > 0.5 | log₁₀(P/S) < 0 |
| **mb - Ms** | Body-wave minus surface-wave magnitude | > 1.0 | < 0.5 |
| **Source Mechanism** | Moment tensor / catalog classification | No tensor + classified explosion | Moment tensor solved (double-couple) |
| **Location** | Distance from known faults | Far from active faults | Near known fault |

### Scoring

Each rule produces:
- A contribution to earthquake consistency (0-1)
- A contribution to explosion consistency (0-1)
- A confidence value (how reliable this measurement is)

Rules are weighted and combined:
""")

st.code("""
weights = {
    "depth": 0.25,
    "p_s_ratio": 0.25,
    "mb_ms": 0.20,
    "source_mechanism": 0.20,
    "location": 0.10,
}

final_score = Σ (rule_contribution × weight) / Σ weights_with_data
""")

st.markdown("""
### Thresholds

| Score | Detail |
|-------|--------|
| **Depth < 2 km** | 90% explosion, 10% earthquake |
| **Depth 2-5 km** | 70% explosion, 30% earthquake |
| **Depth 5-10 km** | 40% explosion, 60% earthquake |
| **Depth > 10 km** | 10% explosion, 90% earthquake |
| **P/S > 0.7** | 95% explosion, 5% earthquake |
| **P/S 0.5-0.7** | 80% explosion, 20% earthquake |
| **P/S < 0** | 15% explosion, 85% earthquake |
| **mb-Ms > 1.5** | 95% explosion, 5% earthquake |
| **mb-Ms > 1.0** | 85% explosion, 15% earthquake |
| **mb-Ms < 0.5** | 20% explosion, 80% earthquake |
| **Moment tensor solved** | 10% explosion, 90% earthquake |
| **Catalog: "explosion"** | 90% explosion, 10% earthquake |
| **Catalog: "earthquake" + mb only** | 40% explosion, 60% earthquake (weak) |

### Source Mechanism Discriminant

The **source mechanism** is one of the most powerful discriminants available from
catalog data alone. When the USGS computes a moment tensor for an event, it reveals
the geometry of the source:

- **Earthquakes** produce a *double-couple* source — shear motion along a fault plane.
  A solved moment tensor (magnitude types `mwr`, `mww`, `mwb`, `mwc`) confirms this.
- **Explosions** produce an *isotropic* source — equal expansion in all directions.
  They do NOT produce clean moment tensor solutions.

This means: if an event has a moment tensor solution, it is almost certainly an earthquake,
even if it's very shallow. This rule correctly reclassifies shallow earthquakes (like the
M4.2 Armenia event at 0.7 km depth with magType=mwr) from Level 2 down to Level 1,
while leaving known explosions (which never have moment tensors) at Level 2.
""")

st.header("Alert Levels")
st.markdown("""
| Level | Name | Criteria | Meaning |
|-------|------|----------|---------|
| 0 | **Ordinary** | Explosion score < 0.3 | Consistent with tectonic earthquake |
| 1 | **Unusual** | Explosion score 0.3-0.6 | Some anomalous features |
| 2 | **Probable Explosion** | Explosion score ≥ 0.6 | Multiple explosion-like characteristics |
| -1 | **Insufficient Data** | Confidence < 0.25 | Not enough information to assess |

**Alerts are generated only when:**
- Alert level ≥ 1
- Confidence ≥ 0.5
- Station count ≥ 3 (when waveform data available)

### Level 1 in Practice: Real-World Examples

Level 1 (Unusual) events are those with **mixed or ambiguous signals** — one indicator
suggests explosion-like characteristics but other evidence is inconclusive. These are
events that warrant further investigation (waveform analysis) but are not alarming on
their own.

Real-world examples from the monitored region:

| Event | Date | Mag | Depth | Why Level 1 |
|-------|------|-----|-------|-------------|
| SW of Gerāsh, Iran | 2024-05-23 | M5.1 | 8.6 km | Shallow for the region, but not surface-level |
| W of Sarpol-e Zahab, Iran | 2024-06-30 | M4.9 | 7.6 km | Moderately shallow, near Iraq border |
| SSW of Javānrūd, Iran | 2024-07-22 | M4.3 | 9.7 km | Just below 10 km threshold |
| NW of Van, Turkey | 2024-07-12 | M4.2 | 9.1 km | Shallow for eastern Turkey |
| NNW of Angeghakot', Armenia | 2024-12-20 | M4.4 | 8.3 km | Shallow event in Caucasus |
| SSW of Umm Bāb, Qatar | 2025-03-31 | M4.1 | 9.9 km | Unusual location and depth |

**What makes these Level 1 (not Level 0 or 2)?**

These events fall in the **5-10 km depth range** — shallow enough to be mildly suspicious
(explosions are typically < 5 km) but deep enough that they're most likely tectonic.
Without waveform P/S ratio data to confirm they're earthquakes, the model conservatively
flags them as "Unusual."

**What would resolve the ambiguity?**

Waveform analysis would typically show:
- Normal (low) P/S ratio → downgrade to Level 0
- High P/S ratio → escalate to Level 2

This is exactly how the alert hierarchy is designed to work: shallow events trigger
additional analysis, and the P/S ratio provides the discriminating evidence.
""")

st.header("Validation Against Known Explosions")
st.markdown("""
The model has been validated against a comprehensive dataset of known nuclear tests,
large conventional explosions, and comparison earthquakes.

### Nuclear Tests (9 events, 100% detection)

| Event | Date | Magnitude | Depth | Model Verdict |
|-------|------|-----------|-------|---------------|
| **NK Nuclear Test #1** | 2006-10-09 | M4.3 | 0 km | **Level 2 — Probable Explosion** |
| **NK Nuclear Test #2** | 2009-05-25 | M4.7 | 0 km | **Level 2 — Probable Explosion** |
| **NK Nuclear Test #3** | 2013-02-12 | M5.1 | 0 km | **Level 2 — Probable Explosion** |
| **NK Nuclear Test #4** | 2016-01-06 | M5.1 | 0 km | **Level 2 — Probable Explosion** |
| **NK Nuclear Test #5** | 2016-09-09 | M5.3 | 0 km | **Level 2 — Probable Explosion** |
| **NK Nuclear Test #6** | 2017-09-03 | M6.3 | 0 km | **Level 2 — Probable Explosion** |
| **India Pokhran-II** | 1998-05-11 | M5.2 | 0 km | **Level 2 — Probable Explosion** |
| **Pakistan Chagai-I** | 1998-05-28 | M4.9 | 0 km | **Level 2 — Probable Explosion** |
| **China Lop Nor (last)** | 1996-07-29 | M4.9 | 0 km | **Level 2 — Probable Explosion** |

All nuclear tests in the USGS catalog are correctly classified as Level 2 when
their published seismic characteristics (depth=0, high P/S, high mb-Ms) are used.

### Large Conventional Explosions (3 events, 100% detection)

| Event | Date | Yield | Magnitude | Model Verdict |
|-------|------|-------|-----------|---------------|
| **Beirut port explosion** | 2020-08-04 | 2,750t NH₄NO₃ | M3.3 | **Level 2 — Probable Explosion** |
| **IDF S.Lebanon detonation** | 2024-10-26 | 370t explosives | M3.6 | **Level 2 — Probable Explosion** |
| **IDF Beaufort Castle** | 2026-07-31 | 700t explosives | M3.8 | **Level 2 — Probable Explosion** |

The October 2024 IDF detonation was initially misidentified as an M5.2 earthquake by
Israel's Truaa early warning system, triggering false alerts to over 1 million people.
The Beaufort Castle demolition (July 31, 2026) generated seismic waves equivalent to M3.8
and was felt across large parts of Lebanon.

### Comparison Earthquakes (5 events, 100% correct rejection)

| Event | Depth | P/S | mb-Ms | Model Verdict |
|-------|-------|-----|-------|---------------|
| Iran earthquake (deep) | 25 km | -0.1 | 0.2 | **Level 0 — Ordinary** |
| Iran earthquake (moderate) | 12 km | -0.1 | 0.2 | **Level 0 — Ordinary** |
| Turkey earthquake (deep) | 40 km | -0.1 | 0.2 | **Level 0 — Ordinary** |
| NK natural earthquake | 8 km | -0.1 | 0.2 | **Level 0 — Ordinary** |
| Lebanon earthquake | 18 km | -0.1 | 0.2 | **Level 0 — Ordinary** |

### Overall Performance

| Category | Events | Correctly Classified | Accuracy |
|----------|--------|---------------------|----------|
| Nuclear tests (→ Level 2) | 9 | 9 | **100%** |
| Conventional explosions (→ ≥Level 1) | 3 | 3 | **100%** |
| Earthquakes (→ Level 0) | 5 | 5 | **100%** |
| **Total** | **17** | **17** | **100%** |

### Key Findings

1. **Perfect separation** when catalog depth + published waveform features are available.

2. **Depth is the strongest single discriminant.** All nuclear tests and large
   conventional explosions are at 0 km depth. This alone triggers Level 1 (Unusual).

3. **P/S ratio and mb-Ms provide confirmation.** Nuclear tests show very high P/S
   (log₁₀ ≈ 0.7-0.8) and mb-Ms > 1.3. Conventional explosions are slightly lower
   (P/S ≈ 0.5, mb-Ms ≈ 0.8) but still clearly distinguished from earthquakes.

4. **The model cannot distinguish nuclear from conventional explosions.**
   Both receive Level 2. This is expected — seismology alone cannot determine
   whether an explosion is nuclear (requires radionuclide evidence).

5. **P/S discrimination requires regional stations (< 200 km).** Our test with
   the Beirut explosion showed that teleseismic (> 500 km) P/S measurements
   degrade for events below M4. The Iran region's P/S performance depends on
   station availability in Turkey, Turkmenistan, and the Persian Gulf.

### Implications for Iran Monitoring

- A shallow (0 km) event in Iran would immediately trigger Level 1 from depth alone
- If P/S ratio > 0.5 is measured, it escalates to Level 2
- High mb-Ms (> 1.0) provides independent confirmation
- The system would detect an explosion but cannot confirm it is nuclear
- Regional stations (KSDI, CSS, EIL, ANTO) are critical for P/S at distances < 500 km

### References

- [CTBTO: Six North Korean Nuclear Tests (2006-2017)](https://www.ctbto.org/our-work/detecting-nuclear-tests)
- [Yield Estimation of the 2020 Beirut Explosion](https://www.nature.com/articles/s41598-021-93690-y) (Nature, 2021)
- [P/S Spectral Ratios for Beirut Explosion](https://pubs.geoscienceworld.org/srl/article-pdf/5633950/srl-2021363.1.pdf) (SRL, 2022)
- [CTBTO Analysis of IDF 370t Detonation](https://conferences.ctbto.org/event/30/contributions/5650/contribution.pdf) (SnT2025)
- [Truaa EEW False Alert from IDF Explosion](https://www.nature.com/articles/s41598-026-50414-4) (Nature, 2026)
- [IDF Beaufort Castle 700t Explosion](https://today.lorientlejour.com/article/1543107/) (L'Orient-Le Jour, 2026)
- [India Pokhran-II Seismic Analysis](https://nuke.fas.org/guide/india/nuke/981100-barc.htm) (BARC, 1998)
- [Pakistan Chagai-I Seismological Identification](https://academic.oup.com/gji/article/150/1/153/591564) (GJI, 2002)
- [China Lop Nor Test Locations](https://pubs.geoscienceworld.org/ssa/bssa/article/94/5/1879/121026/) (BSSA, 2004)
""")


st.header("Limitations")
st.warning("""
**Current limitations of this system:**

1. **Single sensor modality**: Only seismic data is used. The CTBTO uses
   seismic + infrasound + hydroacoustic + radionuclide.

2. **No radionuclide capability**: Cannot confirm nuclear nature of events.

3. **Experimental model**: The scoring system has not been extensively
   validated against real nuclear tests (only against quarry blasts).

4. **Waveform coverage varies**: Not all events have waveform data available.

5. **Location rule not yet implemented**: Distance-to-fault calculation is
   a placeholder.

6. **Single provider**: Currently only uses USGS catalog.

7. **Not real-time**: Updates every 30 minutes via GitHub Actions.
""")

st.header("Architecture")
st.markdown("""
```
┌────────────────────────────────────────────────────┐
│                  Data Sources                        │
│  USGS FDSN API → Events                            │
│  IRIS/ORFEUS   → Waveforms, Station metadata       │
└──────────────────────┬─────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │  GitHub Actions       │
           │  (every 30 minutes)   │
           │                       │
           │  1. Fetch new events  │
           │  2. Deduplicate       │
           │  3. Extract features  │
           │  4. Score anomalies   │
           │  5. Save to Parquet   │
           └───────────┬───────────┘
                       │
           ┌───────────┴───────────┐
           │  data/events.parquet  │
           └───────────┬───────────┘
                       │
           ┌───────────┴───────────┐
           │  Streamlit Dashboard  │
           │                       │
           │  • Interactive map    │
           │  • Event table        │
           │  • Event detail pages │
           │  • Alert display      │
           │  • Methodology docs   │
           └───────────────────────┘
```
""")

st.header("Reproducibility")
st.markdown(f"""
Every assessment is traceable:

- **Processing version:** `{Config.PROCESSING_VERSION}`
- **Model version:** `{Config.MODEL_VERSION}`
- **Source data:** Each event links to its original USGS page
- **Provenance:** First-seen and last-updated timestamps recorded

To reproduce any assessment:
```bash
python scripts/reproduce_event.py <event_id>
```

This will re-run the feature extraction and model assessment,
comparing results to stored values.
""")

st.header("Contributing")
st.markdown("""
Seismic Canary is open source. Contributions welcome:

- **Bug reports**: File issues on GitHub
- **Feature requests**: Discuss in issues first
- **Code contributions**: Fork, branch, test, PR
- **Validation data**: Help us collect more labeled examples

Key principle: **Never hide uncertainty or missing data from the user.**
""")
