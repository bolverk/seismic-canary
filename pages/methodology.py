"""Seismic Canary — Methodology Page.

Explains what the system monitors, how scores are calculated,
limitations, and data sources.
"""
import streamlit as st
import pandas as pd
from src.config import Config
from src.models.calibration import run_calibration, summarize


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
The model is validated against a small, individually documented calibration set of
known nuclear tests, large conventional explosions, and comparison earthquakes.

**The tables and numbers below are computed live** by running each event through the
actual `RuleBasedModel`, via `src/models/calibration.py` — the same module used by
`scripts/validate_calibration_set.py`. Nothing here is hand-maintained; if the model
or the calibration set changes, this page changes with it.
""")

_calibration_results = run_calibration()
_summary = summarize(_calibration_results)


def _verdict(alert_level: int) -> str:
    return f"Level {alert_level} — {Config.EVENT_LEVEL_LABELS.get(alert_level, 'Unknown')}"


def _category_table(category: str, extra_columns: list) -> pd.DataFrame:
    rows = []
    for r in _calibration_results:
        if r.event.true_category != category:
            continue
        row = {"Event": r.event.name}
        if r.event.date:
            row["Date"] = r.event.date
        if "magnitude" in extra_columns:
            row["Magnitude"] = f"M{r.event.magnitude:.1f}" if r.event.magnitude is not None else "—"
        if "depth" in extra_columns:
            row["Depth"] = f"{r.event.depth_km:.0f} km" if r.event.depth_km is not None else "—"
        if "p_s" in extra_columns:
            row["P/S"] = r.event.p_s_ratio if r.event.p_s_ratio is not None else "—"
        if "mb_ms" in extra_columns:
            row["mb-Ms"] = r.event.mb_ms if r.event.mb_ms is not None else "—"
        row["Model Verdict"] = _verdict(r.alert_level)
        row["Result"] = "✅ as expected" if r.passed else f"⚠️ expected {_verdict(r.event.expected_alert_level)}"
        rows.append(row)
    return pd.DataFrame(rows)


nuke_total, nuke_passed = _summary["nuclear_explosion"]
st.markdown(f"### Nuclear Tests ({nuke_total} events, {nuke_passed}/{nuke_total} correctly classified)")
st.dataframe(_category_table("nuclear_explosion", ["magnitude", "depth"]), use_container_width=True, hide_index=True)
st.caption(
    "All nuclear tests use depth=0 km (well documented by CTBTO) plus the uncontested "
    "ground-truth event_type; no per-event P/S or mb-Ms is recorded in this repo."
)

conv_total, conv_passed = _summary["conventional_explosion"]
st.markdown(f"### Large Conventional Explosions ({conv_total} events, {conv_passed}/{conv_total} correctly classified)")
st.dataframe(_category_table("conventional_explosion", ["magnitude", "depth"]), use_container_width=True, hide_index=True)
st.markdown("""
The October 2024 IDF detonation was initially misidentified as an M5.2 earthquake by
Israel's Truaa early warning system, triggering false alerts to over 1 million people.
""")

# Naive-vs-calibrated demonstration: generated for whichever calibration events
# carry a naive_input counterpart (currently just Ali al-Taher, since it's the
# only one of these with a real, public USGS catalog record to contrast against).
for r in _calibration_results:
    if r.naive_alert_level is None:
        continue
    st.markdown(f"""
**{r.event.name}** ({r.event.date}) shows why ground truth matters: its raw/naive
catalog input ({r.event.source}) scores **{_verdict(r.naive_alert_level)}**
(`explosion_consistency={r.naive_explosion_consistency:.2f}`), while the known ground
truth (near-surface depth, `event_type=explosion`) scores
**{_verdict(r.alert_level)}** (`explosion_consistency={r.explosion_consistency:.2f}`,
confidence {r.confidence:.2f}).
""")

eq_total, eq_passed = _summary["earthquake"]
st.markdown(f"### Comparison Earthquakes ({eq_total} events, {eq_passed}/{eq_total} correctly classified)")
st.dataframe(_category_table("earthquake", ["depth", "p_s", "mb_ms"]), use_container_width=True, hide_index=True)

for r in _calibration_results:
    if r.event.true_category == "earthquake" and not r.passed:
        st.markdown(f"""
**{r.event.name}** does not land on its expected verdict — it scores
**{_verdict(r.alert_level)}** (`explosion_consistency={r.explosion_consistency:.3f}`)
instead of {_verdict(r.event.expected_alert_level)}. {r.summary} It is still correctly
*not* flagged as a probable explosion (Level 2); it lands in the model's 5-10 km
"ambiguous" depth bucket, where even earthquake-typical P/S and mb-Ms aren't enough
to fully offset a neutral location placeholder and a weak source-mechanism signal.
""")

st.markdown("### Overall Performance")
overall_total, overall_passed = _summary["overall"]
_perf_rows = [
    ("Nuclear tests (→ Level 2)", nuke_total, nuke_passed),
    ("Conventional explosions (→ ≥Level 1)", conv_total, conv_passed),
    ("Earthquakes (→ Level 0)", eq_total, eq_passed),
    ("Total", overall_total, overall_passed),
]
st.dataframe(
    pd.DataFrame([
        {"Category": cat, "Events": total, "Correctly Classified": passed,
         "Accuracy": f"{passed / total:.0%}"}
        for cat, total, passed in _perf_rows
    ]),
    use_container_width=True, hide_index=True,
)

st.markdown("""
### Key Findings

1. **Depth is the strongest single discriminant.** All nuclear tests and large
   conventional explosions in this calibration set are at or near 0 km depth.

2. **The model cannot distinguish nuclear from conventional explosions.**
   Both receive Level 2. This is expected — seismology alone cannot determine
   whether an explosion is nuclear (requires radionuclide evidence).

3. **Catalog auto-classification can mislead the model.** Small, shallow,
   anthropogenic events (like the Lebanon tunnel demolitions above) often aren't
   in the USGS catalog at all, or get a default fixed depth and a naive
   `event_type=earthquake` tag when they are. The model is only as good as the
   ground truth it's given.

4. **A moderate "ambiguous" depth (5-10 km) plus a neutral location placeholder
   can push even a textbook earthquake to Level 1**, as shown above. Run
   `python scripts/validate_calibration_set.py` for the full rule-by-rule
   breakdown of every calibration event.

### Implications for Iran Monitoring

- A shallow (0 km) event in Iran would immediately trigger Level 1 from depth alone
- If P/S ratio > 0.5 is measured, it escalates to Level 2
- High mb-Ms (> 1.0) provides independent confirmation
- The system would detect an explosion but cannot confirm it is nuclear
- Regional stations (KSDI, CSS, EIL, ANTO) are critical for P/S at distances < 500 km
""")

st.markdown("### References")
_seen_urls = set()
for r in _calibration_results:
    for title, url in r.event.references:
        if url in _seen_urls:
            continue
        _seen_urls.add(url)
        st.markdown(f"- [{title}]({url})")


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
