"""Seismic Canary - Main Dashboard.

Interactive seismic event monitoring dashboard for the Iran region.
Displays events on a map and in a filterable table.
"""
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone, timedelta

from src.config import Config
from src.processing.events import load_events

# Page configuration
st.set_page_config(
    page_title="Seismic Canary",
    page_icon="🐤",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data() -> pd.DataFrame:
    """Load event data from Parquet storage."""
    return load_events(Config.EVENTS_PARQUET)


def filter_events(
    df: pd.DataFrame,
    time_range: str,
    min_magnitude: float,
    max_magnitude: float,
) -> pd.DataFrame:
    """Apply time and magnitude filters to events."""
    if df.empty:
        return df

    now = datetime.now(timezone.utc)
    if time_range == "Last 24 hours":
        cutoff = now - timedelta(hours=24)
    elif time_range == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif time_range == "Last 30 days":
        cutoff = now - timedelta(days=30)
    else:
        cutoff = None

    filtered = df.copy()
    if cutoff is not None:
        filtered = filtered[filtered["origin_time"] >= cutoff]

    # Magnitude filter
    filtered = filtered[
        (filtered["magnitude"] >= min_magnitude)
        & (filtered["magnitude"] <= max_magnitude)
    ]

    return filtered.sort_values("origin_time", ascending=False)


def get_marker_color(alert_level) -> str:
    """Get marker color based on alert level."""
    if pd.isna(alert_level):
        return Config.EVENT_COLORS[-1]  # gray for no assessment
    return Config.EVENT_COLORS.get(int(alert_level), "gray")


def get_alert_label(alert_level) -> str:
    """Get human-readable alert level label."""
    if pd.isna(alert_level):
        return Config.EVENT_LEVEL_LABELS[-1]
    return Config.EVENT_LEVEL_LABELS.get(int(alert_level), "Unknown")


def create_map(events: pd.DataFrame) -> folium.Map:
    """Create a folium map with event markers."""
    center = Config.REGION_BOUNDS.center
    m = folium.Map(
        location=[center[0], center[1]],
        zoom_start=Config.MAP_ZOOM_START,
        tiles="CartoDB positron",
    )

    # Draw monitored region boundary
    bounds = Config.REGION_BOUNDS
    boundary_coords = [
        [bounds.min_latitude, bounds.min_longitude],
        [bounds.min_latitude, bounds.max_longitude],
        [bounds.max_latitude, bounds.max_longitude],
        [bounds.max_latitude, bounds.min_longitude],
        [bounds.min_latitude, bounds.min_longitude],
    ]
    folium.PolyLine(
        boundary_coords,
        color="#1f77b4",
        weight=2,
        opacity=0.5,
        dash_array="5 10",
        tooltip="Monitored Region",
    ).add_to(m)

    # Add event markers
    if not events.empty:
        for _, row in events.iterrows():
            color = get_marker_color(row.get("alert_level"))
            mag = row["magnitude"] if pd.notna(row["magnitude"]) else "?"
            depth = f"{row['depth_km']:.1f}" if pd.notna(row["depth_km"]) else "?"
            time_str = row["origin_time"].strftime("%Y-%m-%d %H:%M UTC") if pd.notna(row["origin_time"]) else "?"
            label = get_alert_label(row.get("alert_level"))

            popup_html = f"""
            <b>M{mag}</b> - {row.get('place', 'Unknown')}<br>
            <b>Time:</b> {time_str}<br>
            <b>Depth:</b> {depth} km<br>
            <b>Status:</b> {label}<br>
            <b>ID:</b> {row['event_id']}
            """

            # Scale marker size with magnitude
            radius = max(3, float(mag) * 2) if mag != "?" else 4

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"M{mag} - {row.get('place', '')}",
            ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background-color: white; padding: 10px 14px; border-radius: 5px;
                border: 2px solid #ccc; font-size: 13px; line-height: 1.8;">
        <b>Alert Level</b><br>
        <span style="color: blue;">&#9679;</span> Ordinary (Level 0)<br>
        <span style="color: orange;">&#9679;</span> Unusual (Level 1)<br>
        <span style="color: red;">&#9679;</span> Probable Explosion (Level 2)<br>
        <span style="color: gray;">&#9679;</span> Insufficient Data<br>
        <br><b>Sites of Interest</b><br>
        <span style="color: black;">&#9670;</span> Nuclear/military facility
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Sites of interest - known nuclear/military facilities
    sites_of_interest = [
        {"name": "Natanz", "lat": 33.717, "lon": 51.717,
         "desc": "Uranium enrichment facility (underground)"},
        {"name": "Fordow", "lat": 34.885, "lon": 50.996,
         "desc": "Underground uranium enrichment plant, near Qom"},
        {"name": "Parchin", "lat": 35.520, "lon": 51.780,
         "desc": "Military complex, suspected weapons-related experiments"},
        {"name": "Semnan", "lat": 35.235, "lon": 53.921,
         "desc": "Space/missile center; Project Midan test area nearby"},
        {"name": "Lut Desert (Nayband area)", "lat": 33.5, "lon": 57.5,
         "desc": "Remote desert; identified as potential test site in IAEA archive"},
        {"name": "Project Midan (SE of Semnan)", "lat": 34.8, "lon": 54.5,
         "desc": "Underground nuclear test site development (per IAEA archive)"},
    ]

    for site in sites_of_interest:
        folium.Marker(
            location=[site["lat"], site["lon"]],
            popup=folium.Popup(
                f"<b>{site['name']}</b><br>{site['desc']}",
                max_width=250,
            ),
            tooltip=site["name"],
            icon=folium.Icon(
                color="black",
                icon_color="yellow",
                icon="warning-sign",
                prefix="glyphicon",
            ),
        ).add_to(m)

    return m


def render_event_table(events: pd.DataFrame) -> None:
    """Render the event table."""
    if events.empty:
        st.info("No events match the current filters.")
        return

    display_df = events[
        ["origin_time", "place", "magnitude", "depth_km", "alert_level", "event_id"]
    ].copy()

    display_df["Time (UTC)"] = display_df["origin_time"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["Location"] = display_df["place"].fillna("Unknown")
    display_df["Mag"] = display_df["magnitude"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "-"
    )
    display_df["Depth (km)"] = display_df["depth_km"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "-"
    )
    display_df["Status"] = display_df["alert_level"].apply(get_alert_label)

    # Show the table
    st.dataframe(
        display_df[["Time (UTC)", "Location", "Mag", "Depth (km)", "Status", "event_id"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "event_id": st.column_config.TextColumn("Event ID", width="small"),
            "Time (UTC)": st.column_config.TextColumn("Time (UTC)", width="medium"),
            "Location": st.column_config.TextColumn("Location", width="large"),
            "Mag": st.column_config.TextColumn("Mag", width="small"),
            "Depth (km)": st.column_config.TextColumn("Depth (km)", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        },
    )


def render_sidebar() -> dict:
    """Render the sidebar with filters and project info."""
    with st.sidebar:
        st.title("🐤 Seismic Canary")
        st.caption(f"v{Config.VERSION}")

        st.markdown("---")
        st.markdown(
            "**Monitoring:** " + Config.MONITORED_REGION_DESCRIPTION
        )
        st.markdown("---")

        # Filters
        st.subheader("Filters")
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "All events"],
            index=3,
        )

        mag_range = st.slider(
            "Magnitude Range",
            min_value=0.0,
            max_value=9.0,
            value=(2.0, 9.0),
            step=0.5,
        )

        st.markdown("---")

        # Disclaimer
        st.markdown(
            """
            ⚠️ **Disclaimer**

            Seismic Canary monitors publicly available seismic data and
            provides experimental anomaly assessments.

            **An explosion-like seismic signature does not establish
            that an event was nuclear.**

            This system reports observations and statistical anomalies.
            It does not make claims about the nature of events beyond
            what the data supports.
            """
        )

        st.markdown("---")
        st.caption(
            f"Data source: USGS FDSN Event API | "
            f"Model: {Config.MODEL_VERSION}"
        )

    return {
        "time_range": time_range,
        "min_magnitude": mag_range[0],
        "max_magnitude": mag_range[1],
    }


def render_event_detail(events: pd.DataFrame) -> None:
    """Render event detail view when an event is selected."""
    event_id = st.session_state.get("selected_event_id")
    if not event_id:
        return

    event = events[events["event_id"] == event_id]
    if event.empty:
        st.warning(f"Event {event_id} not found.")
        return

    row = event.iloc[0]
    st.markdown("---")
    st.subheader(f"Event Detail: {event_id}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Magnitude", f"{row['magnitude']:.1f}" if pd.notna(row['magnitude']) else "N/A")
    with col2:
        st.metric("Depth", f"{row['depth_km']:.1f} km" if pd.notna(row['depth_km']) else "N/A")
    with col3:
        st.metric("Status", get_alert_label(row.get("alert_level")))

    st.markdown("**Basic Information**")
    info_data = {
        "Origin Time": row["origin_time"].strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(row["origin_time"]) else "N/A",
        "Location": row.get("place", "N/A"),
        "Coordinates": f"{row['latitude']:.3f}°N, {row['longitude']:.3f}°E",
        "Provider": row.get("provider", "N/A"),
        "Event Type": row.get("event_type", "N/A"),
        "Magnitude Type": row.get("magnitude_type", "N/A"),
    }
    for k, v in info_data.items():
        st.text(f"  {k}: {v}")

    # Source link
    source_url = row.get("source_url")
    if source_url and pd.notna(source_url):
        st.markdown(f"[View on source provider]({source_url})")

    # Seismic features (if available)
    st.markdown("**Seismic Features**")
    features = {
        "P/S Ratio (log10)": row.get("p_s_ratio"),
        "mb - Ms": row.get("mb_ms"),
        "Corner Frequency": row.get("corner_frequency"),
        "Spectral Slope": row.get("spectral_slope"),
        "Dominant Frequency": row.get("dominant_frequency"),
        "SNR": row.get("snr"),
        "Station Count": row.get("station_count"),
        "Waveform Quality": row.get("waveform_quality"),
    }

    has_features = False
    for name, value in features.items():
        if pd.notna(value):
            st.text(f"  {name}: {value}")
            has_features = True

    if not has_features:
        st.caption("  No waveform features available yet.")

    # Provenance
    st.markdown("**Provenance**")
    st.text(f"  First seen: {row.get('first_seen', 'N/A')}")
    st.text(f"  Last updated: {row.get('last_updated', 'N/A')}")
    st.text(f"  Processing version: {row.get('processing_version', 'N/A')}")
    st.text(f"  Model version: {row.get('model_version', 'N/A')}")

    if st.button("← Back to overview"):
        st.session_state.pop("selected_event_id", None)
        st.rerun()


def main():
    """Main dashboard entrypoint."""
    # Initialize session state
    if "selected_event_id" not in st.session_state:
        st.session_state["selected_event_id"] = None

    # Render sidebar and get filter settings
    filters = render_sidebar()

    # Load data
    events = load_data()

    # Apply filters
    filtered = filter_events(
        events,
        time_range=filters["time_range"],
        min_magnitude=filters["min_magnitude"],
        max_magnitude=filters["max_magnitude"],
    )

    # Check if we're in detail view
    if st.session_state.get("selected_event_id"):
        render_event_detail(events)
        return

    # Main content: header
    st.title("🐤 Seismic Canary")
    st.markdown(
        f"Monitoring for underground nuclear tests and anomalous explosions in {Config.MONITORED_REGION_DESCRIPTION}"
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(filtered))
    with col2:
        unusual = len(filtered[filtered["alert_level"] >= 1]) if "alert_level" in filtered.columns else 0
        st.metric("Unusual Events", unusual)
    with col3:
        if not filtered.empty and "magnitude" in filtered.columns:
            max_mag = filtered["magnitude"].max()
            st.metric("Max Magnitude", f"{max_mag:.1f}" if pd.notna(max_mag) else "-")
        else:
            st.metric("Max Magnitude", "-")
    with col4:
        if not filtered.empty and "origin_time" in filtered.columns:
            latest = filtered["origin_time"].max()
            if pd.notna(latest):
                st.metric("Latest Event", latest.strftime("%m/%d %H:%M UTC"))
            else:
                st.metric("Latest Event", "-")
        else:
            st.metric("Latest Event", "-")

    # Map
    st.subheader("Event Map")
    event_map = create_map(filtered)
    st_folium(event_map, use_container_width=True, height=500)

    # Event table
    st.subheader(f"Events ({len(filtered)})")

    # Event selection
    if not filtered.empty:
        selected_id = st.selectbox(
            "Select event for details:",
            options=[""] + filtered["event_id"].tolist(),
            format_func=lambda x: "Select an event..." if x == "" else f"{x} - {filtered[filtered['event_id']==x].iloc[0].get('place', 'Unknown') if not filtered[filtered['event_id']==x].empty else x}",
        )
        if selected_id:
            st.session_state["selected_event_id"] = selected_id
            st.rerun()

    render_event_table(filtered)


if __name__ == "__main__":
    main()
