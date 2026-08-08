# 🐤 Seismic Canary

**Public-data seismic event monitoring for Iran and surrounding region.**

Seismic Canary automatically monitors publicly available seismic events, computes interpretable explosion-vs-earthquake diagnostic features, and presents transparent evidence records — without making unsupported claims about nuclear tests.

## What This Does

- Ingests seismic events from the USGS FDSN API every 30 minutes
- Monitors Iran, Iraq, Turkey, Armenia, Azerbaijan, Turkmenistan, and the Persian Gulf
- Computes seismic features: P/S amplitude ratio, mb-Ms, spectral characteristics
- Applies a transparent rule-based anomaly model with interpretable scores
- Presents all evidence in a Streamlit dashboard with maps, tables, and detail pages
- Clearly separates: observed data → derived features → model predictions → interpretation

## What This Does NOT Claim

⚠️ **An explosion-like seismic signature does NOT establish that an event was nuclear.**

This system:
- Cannot confirm the nuclear nature of any event (requires radionuclide detection)
- Cannot distinguish nuclear from conventional explosions seismically
- Does not replace the CTBTO's comprehensive monitoring system
- Reports observations and statistical anomalies, not intelligence assessments

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Data Sources                         │
│  USGS FDSN API → Event catalog                  │
│  IRIS/ORFEUS   → Waveforms, Station metadata    │
└────────────────────┬────────────────────────────┘
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

## Quick Start

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd iran_underground_nuclear_experiment_monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run the dashboard
streamlit run app.py

# Fetch initial data
python scripts/update_events.py --days 30
```

### Deploy to Streamlit Community Cloud

1. Push repository to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and `app.py` as entrypoint
5. Deploy

The GitHub Actions workflow will automatically update `data/events.parquet` every 30 minutes.

## Project Structure

```
.
├── app.py                          # Streamlit dashboard entrypoint
├── requirements.txt                # Pinned Python dependencies
├── pages/
│   └── methodology.py             # In-app documentation page
├── src/
│   ├── config.py                  # Centralized configuration
│   ├── ingestion/
│   │   ├── seismic.py            # USGS event ingestion
│   │   ├── stations.py           # FDSN station discovery
│   │   └── providers.py          # Multi-sensor interfaces (future)
│   ├── processing/
│   │   ├── events.py            # Event schema & Parquet storage
│   │   ├── features.py          # Spectral analysis & feature extraction
│   │   ├── waveforms.py         # P/S measurement pipeline
│   │   └── timeline.py          # Event lifecycle tracking
│   └── models/
│       ├── anomaly.py           # Rule-based anomaly model
│       ├── alerts.py            # Alert generation system
│       └── evidence.py          # Multi-sensor evidence model
├── scripts/
│   ├── update_events.py          # Scheduled data pipeline
│   ├── validate_model.py         # Model performance validation
│   └── reproduce_event.py        # Assessment reproducibility tool
├── data/
│   └── events.parquet            # Event database (auto-updated)
├── tests/                         # Comprehensive test suite
└── .github/workflows/
    └── update-data.yml           # Scheduled ingestion (every 30 min)
```

## Data Sources

| Source | Provides | Access |
|--------|----------|--------|
| [USGS FDSN Event API](https://earthquake.usgs.gov/fdsnws/event/1/) | Event metadata | Public |
| [IRIS FDSN Waveform](https://service.iris.edu/fdsnws/) | Seismic waveforms | Public |
| [FDSN Station Service](https://www.fdsn.org/webservices/) | Station metadata | Public |

## Anomaly Model

The rule-based model (version `baseline-001`) assesses each event on:

| Feature | Explosion-like when... | Weight |
|---------|----------------------|--------|
| Depth | < 5 km | 30% |
| P/S Ratio | log₁₀(P/S) > 0.5 | 30% |
| mb - Ms | > 1.0 | 25% |
| Location | Far from active faults | 15% |

**Alert Levels:**
- **Level 0** — Ordinary: Consistent with tectonic earthquake
- **Level 1** — Unusual: Some anomalous features
- **Level 2** — Probable Explosion: Multiple explosion-like characteristics
- **Level 3** — Corroborated: Multi-sensor evidence (future)
- **Level -1** — Insufficient Data

## Reproducibility

Every assessment is traceable to source data and model version:

```bash
# Reproduce any event's assessment
python scripts/reproduce_event.py <event_id>
```

## Future Roadmap

The architecture supports adding:
- Infrasound detection and event association
- Satellite SAR/optical change detection (Sentinel-1/2)
- GNSS displacement measurements
- Radionuclide observations (when publicly available)
- Bayesian evidence fusion across modalities

## Contributing

Contributions welcome. Key principles:
- Never hide uncertainty or missing data from the user
- Every measurement must link to its source
- No unexplained "black box" scoring
- Test everything offline with mocked data

## Author

Created by **Almog Yalinewich**.

## License

MIT License. See [LICENSE](LICENSE).
