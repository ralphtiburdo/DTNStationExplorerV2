from functools import lru_cache
from time import sleep
import requests
import pandas as pd
import reverse_geocode
import streamlit as st
import pydeck as pdk
import io
import logging
import re
import calendar
import numpy as np
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

load_dotenv()

CLIENT_ID_INTERNAL = os.getenv("CLIENT_ID_INTERNAL")
CLIENT_SECRET_INTERNAL = os.getenv("CLIENT_SECRET_INTERNAL")
CLIENT_ID_CORE = os.getenv("CLIENT_ID_CORE")
CLIENT_SECRET_CORE = os.getenv("CLIENT_SECRET_CORE")
CLIENT_ID_PLUS = os.getenv("CLIENT_ID_PLUS")
CLIENT_SECRET_PLUS = os.getenv("CLIENT_SECRET_PLUS")


# --- Enhanced Helper functions ---
@st.cache_data(ttl=3600)  # Cache tokens for 1 hour
def get_token(client_id, client_secret):
    url = 'https://api.auth.dtn.com/v1/tokens/authorize'
    payload = {"grant_type": "client_credentials", "client_id": client_id,
               "client_secret": client_secret, "audience": "https://weather.api.dtn.com/observations"}
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return (data.get('data') or data).get('access_token')
    except requests.exceptions.RequestException as e:
        st.error(f"Authentication failed: {str(e)}")
        return None


@lru_cache(maxsize=None)
def reverse_geocode_cached(lat, lon):
    try:
        return reverse_geocode.search([(lat, lon)])[0]['country']
    except:
        return "Unknown"


@st.cache_data(ttl=600, show_spinner="Fetching station data…")  # Cache for 10 minutes
def get_stations_by_access(access_choice: str):
    creds = {
        "Internal": (CLIENT_ID_INTERNAL, CLIENT_SECRET_INTERNAL),
        "Core": (CLIENT_ID_CORE, CLIENT_SECRET_CORE),
        "Plus": (CLIENT_ID_PLUS, CLIENT_SECRET_PLUS),
    }
    client_id, client_secret = creds.get(access_choice, creds["Internal"])
    token = get_token(client_id, client_secret)

    if not token:
        return pd.DataFrame(), None

    url = 'https://obs.api.dtn.com/v1/observations/stations'
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params={
            'by': 'boundingBox', 'minLat': '-90', 'maxLat': '90', 'minLon': '-180', 'maxLon': '180',
            'obsTypes': 'RWIS,AG,METAR,SYNOP,BUOY,Citizen,SHIP,Hydro,Others,HFM,GHCND,Customer,ISD'
        }, timeout=30)
        resp.raise_for_status()
        df = pd.json_normalize(resp.json())

        # Batch reverse geocoding for performance
        unique_coords = df[['latitude', 'longitude']].drop_duplicates()
        unique_coords['Country'] = unique_coords.apply(
            lambda r: reverse_geocode_cached(r.latitude, r.longitude), axis=1
        )
        df = df.merge(unique_coords, on=['latitude', 'longitude'], how='left')

        tag_map = {'tags.name': 'name', 'tags.mgID': 'mgID', 'tags.wmo': 'wmo', 'tags.icao': 'icao',
                   'tags.madisId': 'madisId', 'tags.eaukID': 'eaukID', 'tags.iata': 'iata', 'tags.faa': 'faa',
                   'tags.dwdID': 'dwdID', 'tags.davisId': 'davisId', 'tags.dtnLegacyID': 'dtnLegacyID',
                   'tags.ghcndID': 'ghcndID'}

        # Create new columns with default empty string
        for new_col in tag_map.values():
            df[new_col] = ""

        # Fill new columns from tags where available
        for old, new in tag_map.items():
            if old in df.columns:
                df[new] = df[old].fillna("")

        # Drop original tag columns
        df.drop(columns=[c for c in df.columns if c.startswith('tags.')], errors='ignore', inplace=True)

        # Handle list-type columns safely
        df['stationCode'] = df.get('stationCode', pd.Series([""] * len(df))).fillna("")
        df['obsTypes'] = df.get('obsTypes', pd.Series([[]] * len(df))).apply(
            lambda x: list(map(str, x)) if isinstance(x, list) else [str(x)] if pd.notna(x) else [])
        df['parameters'] = df.get('parameters', pd.Series([[]] * len(df))).apply(
            lambda x: list(map(str, x)) if isinstance(x, list) else [str(x)] if pd.notna(x) else [])

        df['search_blob'] = df.astype(str).apply(lambda row: ' '.join(row.values).lower(), axis=1)
        df.reset_index(drop=True, inplace=True)
        return df, token
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch stations: {str(e)}")
        return pd.DataFrame(), None


@st.cache_data
def get_summary(df):
    expl = df[['Country', 'obsTypes']].explode('obsTypes')
    pivot = expl.pivot_table(index='Country', columns='obsTypes', aggfunc='size', fill_value=0)
    pivot['Total'] = pivot.sum(axis=1)
    pivot = pivot[pivot.index.notna()].sort_index()
    return pivot.reset_index()


def extract_parameter_metadata(archive_counts):
    recs = []
    for param, months in (archive_counts or {}).items():
        if not isinstance(months, dict): continue
        keys = sorted(months.keys(), key=lambda x: tuple(map(int, x.split('-'))))
        first, last = (keys[0], keys[-1]) if keys else ("-", "-")
        recs.append({
            "Parameter": param,
            "First Obs": pd.to_datetime(first, format="%Y-%m").strftime("%b %Y") if first != "-" else "-",
            "Latest Obs": pd.to_datetime(last, format="%Y-%m").strftime("%b %Y") if last != "-" else "-"
        })
    return pd.DataFrame(recs).sort_values("Parameter")


def generate_heatmap_plotly(archive_counts, param):
    """Generate an interactive heatmap using Plotly"""
    counts = archive_counts.get(param, {})
    if not counts:
        return None

    # Extract data for heatmap
    years = set()
    months_data = []

    for month_key, count in counts.items():
        try:
            year, month = map(int, month_key.split('-'))
            years.add(year)
            months_data.append({
                "year": year,
                "month": month,
                "count": count,
                "month_name": calendar.month_abbr[month],
                "date": f"{calendar.month_abbr[month]} {year}"
            })
        except:
            continue

    if not months_data:
        return None

    # Create a DataFrame
    df_heat = pd.DataFrame(months_data)

    # Pivot for heatmap
    heatmap_data = df_heat.pivot_table(values='count', index='month', columns='year',
                                       aggfunc='sum', fill_value=0)

    # Create the heatmap
    fig = px.imshow(
        heatmap_data.values,
        labels=dict(x="Year", y="Month", color="Observations"),
        x=sorted(years),
        y=[calendar.month_abbr[i] for i in range(1, 13)],
        aspect="auto",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        title=f"Observation Frequency for {param}",
        xaxis_nticks=len(years),
        height=400
    )

    return fig


def drop_blank_columns(df):
    return df.loc[:, df.apply(lambda col: col.replace("", pd.NA).dropna().astype(str).str.strip().ne("").any())]


@st.cache_data(ttl=300, show_spinner=False)  # Cache metadata for 5 minutes
def fetch_station_metadata(code, token, retries=5):
    if not token:
        return {}

    url = "https://obs.api.dtn.com/v2/observations/stations"
    params = {"by": "stationCodes", "stationCodes": code, "isArchive": "true", "archiveCounts": "true"}

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=20)
            resp.raise_for_status()
            feats = resp.json().get("features", [])
            return feats[0] if feats else {}
        except Exception as e:
            logging.warning(f"Retry {attempt + 1} failed: {e}")
            sleep(2 ** attempt)

    return {}


def create_station_map(fdf, selected_station=None):
    """Create an interactive map with stations"""
    if fdf.empty:
        return None

    # Prepare data for the map
    map_data = fdf.copy()

    # Add color coding based on observation types
    obs_type_colors = {
        'METAR': [255, 99, 71, 180],  # Tomato
        'SYNOP': [65, 105, 225, 180],  # Royal Blue
        'BUOY': [50, 205, 50, 180],  # Lime Green
        'AG': [255, 165, 0, 180],  # Orange
        'RWIS': [148, 0, 211, 180],  # Dark Violet
        'Citizen': [220, 20, 60, 180],  # Crimson
        'SHIP': [0, 191, 255, 180],  # Deep Sky Blue
        'Hydro': [0, 100, 0, 180],  # Dark Green
        'Others': [128, 128, 128, 180],  # Gray
    }

    # Assign colors based on primary observation type
    def get_color(obs_types):
        for obs_type in obs_types:
            if obs_type in obs_type_colors:
                return obs_type_colors[obs_type]
        return [200, 200, 200, 180]  # Default gray

    map_data['color'] = map_data['obsTypes'].apply(get_color)

    # Highlight selected station if any
    if selected_station:
        map_data['is_selected'] = map_data['stationCode'] == selected_station
        map_data.loc[map_data['is_selected'], 'color'] = [255, 0, 0, 255]  # Red for selected

    # Create the layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position='[longitude, latitude]',
        get_radius=1000,
        get_fill_color='color',
        pickable=True,
        auto_highlight=True,
        radius_min_pixels=5,
        radius_max_pixels=20,
        radius_scale=1
    )

    # Set initial view state
    if selected_station and not map_data[map_data['is_selected']].empty:
        # Focus on selected station
        sel_station = map_data[map_data['is_selected']].iloc[0]
        view_state = pdk.ViewState(
            latitude=sel_station.latitude,
            longitude=sel_station.longitude,
            zoom=8,
            pitch=0
        )
    else:
        # Focus on all stations
        view_state = pdk.ViewState(
            latitude=map_data.latitude.mean(),
            longitude=map_data.longitude.mean(),
            zoom=2,
            pitch=0
        )

    # Create tooltip
    tooltip = {
        "html": "<b>Station:</b> {stationCode}<br/><b>Name:</b> {name}<br/><b>Types:</b> {obsTypes}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    # Create deck
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='light' if not st.session_state.get('dark_mode', False) else 'dark'
    )

    return deck


def create_obs_type_chart(df):
    """Create a chart showing distribution of observation types"""
    expl = df.explode('obsTypes')
    type_counts = expl['obsTypes'].value_counts().reset_index()
    type_counts.columns = ['Observation Type', 'Count']

    fig = px.bar(
        type_counts,
        x='Observation Type',
        y='Count',
        title="Distribution of Observation Types",
        color='Observation Type'
    )

    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45
    )

    return fig


def create_country_chart(df):
    """Create a chart showing stations by country"""
    country_counts = df['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']

    # Show top 15 countries
    top_countries = country_counts.head(15)

    fig = px.bar(
        top_countries,
        x='Country',
        y='Count',
        title="Stations by Country (Top 15)",
        color='Country'
    )

    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45
    )

    return fig


# --- Enhanced UI and main logic ---
def show_dashboard(df, token):
    from pydeck.data_utils.viewport_helpers import compute_view

    # Inject custom CSS for styling
    st.markdown("""
    <style>
        /* Main table styling */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }

        .dataframe th {
            background: linear-gradient(135deg, #0072b5 0%, #00a99d 100%);
            color: white;
            font-weight: 600;
            padding: 12px 15px;
            text-align: left;
            position: sticky;
            top: 0;
        }

        .dataframe td {
            padding: 10px 15px;
            border-bottom: 1px solid #e0e0e0;
        }

        .dataframe tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .dataframe tr:hover {
            background-color: #e9f7fe;
            transition: background-color 0.2s;
        }

        /* Station card styling */
        .station-card {
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #00a99d;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .station-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }

        .station-card h3 {
            color: #005a87;
            border-bottom: 1px solid #e0f0f7;
            padding-bottom: 10px;
            margin-top: 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .status-active {
            background: linear-gradient(135deg, #00a99d 0%, #007a6e 100%);
            color: white;
            padding: 3px 12px;
            border-radius: 12px;
            display: inline-block;
            font-size: 0.85em;
            font-weight: bold;
        }

        .status-inactive {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            color: white;
            padding: 3px 12px;
            border-radius: 12px;
            display: inline-block;
            font-size: 0.85em;
            font-weight: bold;
        }

        .station-property {
            padding: 8px 0;
            border-bottom: 1px solid #f0f9fb;
            display: flex;
        }

        .station-property:last-child {
            border-bottom: none;
        }

        .property-label {
            font-weight: 600;
            min-width: 140px;
            color: #0072b5;
        }

        .property-value {
            flex: 1;
            color: #005a87;
        }

        .summary-card {
            background: linear-gradient(135deg, #0072b5 0%, #00a99d 100%);
            color: white;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            transition: transform 0.3s;
        }

        .summary-card:hover {
            transform: scale(1.02);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 5px 0;
            text-align: center;
        }

        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
            text-align: center;
        }

        .section-header {
            background: linear-gradient(90deg, #0072b5, #00a99d);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 20px 0 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .export-btn {
            background: linear-gradient(135deg, #0072b5 0%, #00a99d 100%);
            color: white !important;
            border: none;
            border-radius: 5px;
            padding: 8px 15px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            text-align: center;
            display: block;
            margin: 10px 0;
        }

        .export-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .collapsible-panel {
            background-color: #f8fdff;
            border: 1px solid #cceff5;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        }

        .panel-header {
            color: #0072b5;
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }

        .panel-header i {
            margin-right: 10px;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #f8fdff;
            border-radius: 6px;
            border: 1px solid #cceff5;
        }

        .stTextInput input {
            background-color: #f8fdff;
            border-radius: 6px;
            border: 1px solid #cceff5;
        }

        /* Parameter card styling */
        .parameter-card {
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #00a99d;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .parameter-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }

        /* Placeholder card styling */
        .placeholder-card {
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #00a99d;
            transition: transform 0.3s, box-shadow 0.3s;
            text-align: center;
            font-size: 1.1rem;
        }

        .placeholder-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }

        .parameter-header {
            font-weight: bold;
            color: #0072b5;
            margin-bottom: 8px;
            font-size: 1rem;
        }

        .parameter-row {
            display: flex;
            padding: 4px 0;
        }

        .parameter-label {
            font-weight: 600;
            min-width: 100px;
            color: #0072b5;
        }

        .parameter-value {
            flex: 1;
            color: #005a87;
        }

        /* Heatmap styling */
        .heatmap-container {
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 4px;
            border: 1px solid #e0f0f7;
        }

        .heatmap-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, 12px);
            gap: 2px;
            max-width: 100%;
            overflow-x: auto;
        }

        .heatmap-cell {
            width: 12px;
            height: 12px;
            border-radius: 2px;
            display: block;
            transition: all 0.2s;
        }

        .heatmap-cell:hover {
            transform: scale(1.5);
            z-index: 100;
            box-shadow: 0 0 2px rgba(0,0,0,0.5);
        }

        .heatmap-gradient {
            flex-grow: 1;
            height: 8px;
            margin: 0 5px;
            background: linear-gradient(90deg, #f0f0f0, #2e8b57);
            border-radius: 3px;
        }

        .heatmap-note {
            font-size: 0.7rem;
            color: #666;
            margin-top: 5px;
            text-align: center;
        }

        /* Custom expander styling */
        .custom-expander {
            margin-top: 10px;
            border-radius: 4px;
            overflow: hidden;
        }

        .custom-expander-header {
            background-color: #e9f7fe;
            padding: 8px 12px;
            border-radius: 4px;
            font-weight: bold;
            color: #0072b5;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.2s;
        }

        .custom-expander-header:hover {
            background-color: #d0eefd;
        }

        .custom-expander-content {
            padding: 10px;
            background: white;
            border: 1px solid #cceff5;
            border-radius: 0 0 4px 4px;
            border-top: none;
            margin-top: 1px;
        }

        .filter-section {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .filter-header {
            font-weight: bold;
            color: #0072b5;
            margin-bottom: 10px;
        }

        /* Toggle switch for dark mode */
        .toggle-container {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }

        .toggle-label {
            margin-left: 10px;
            font-weight: 500;
        }

    </style>
    """, unsafe_allow_html=True)

    # Add dark mode styles
    if st.session_state.get('dark_mode', False):
        st.markdown("""
        <style>
            /* Dark mode overrides */
            body {
                background-color: #0e1117;
                color: #f0f2f6;
            }

            .summary-card, .station-card, .parameter-card, .collapsible-panel, 
            .custom-expander-content, .filter-section, .section-header {
                background-color: #1a1d24 !important;
                color: #f0f2f6 !important;
                border-color: #2d3746 !important;
            }

            .station-card h3, .parameter-header, 
            .property-label, .property-value,
            .parameter-label, .parameter-value {
                color: #f0f2f6 !important;
            }

            .dataframe th {
                background: linear-gradient(135deg, #004a7f 0%, #00877a 100%) !important;
            }

            .dataframe tr:nth-child(even) {
                background-color: #1e222d !important;
            }

            .dataframe tr:hover {
                background-color: #2a3040 !important;
            }

            .stSelectbox div[data-baseweb="select"] > div,
            .stTextInput input {
                background-color: #1a1d24 !important;
                border-color: #2d3746 !important;
                color: #f0f2f6 !important;
            }

            .placeholder-card {
                background-color: #1a1d24 !important;
                border-color: #2d3746 !important;
                color: #9aa5b1 !important;
            }

            .filter-section {
                background-color: #1a1d24 !important;
            }

            .custom-expander-header {
                background-color: #2a3040 !important;
                color: #f0f2f6 !important;
            }

            .custom-expander-header:hover {
                background-color: #343b4d !important;
            }
        </style>
        """, unsafe_allow_html=True)

    # Main content area - Filters and Summary
    st.markdown("<div class='filter-section'>", unsafe_allow_html=True)

    # Search functionality
    search = st.text_input(
        "Search Station",
        help="Search by Station Code, ICAO, WMO, or mgID",
        key="main_search"
    )

    # Filters
    st.markdown("<div class='filter-header'>Filters</div>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        sel_countries = st.multiselect(
            "Filter by Country:",
            options=sorted(df['Country'].unique()),
            help="Filter stations by their country location"
        )
    with col4:
        sel_obs = st.multiselect(
            "Filter by Observation Types:",
            options=sorted({o for row in df['obsTypes'] for o in row}),
            help="Show only stations reporting selected observation types"
        )
    with col5:
        sel_params = st.multiselect(
            "Filter by Parameters:",
            options=sorted({p for row in df['parameters'] for p in row}),
            help="Show only stations reporting selected parameters"
        )

    # Add clear filters button
    if st.button("Clear All Filters", use_container_width=True):
        st.session_state.main_search = ""
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # Close filter-section

    # Process filters
    fdf = df.copy()
    filters_applied = False
    show = False

    if search.strip():
        terms = [t.strip().lower() for t in re.split(r"[,\s]+", search) if t.strip()]
        mask = fdf[['stationCode', 'icao', 'wmo', 'mgID', 'madisId', 'iata', 'faa', 'name', 'davisId', 'dtnLegacyID',
                    'ghcndID']].astype(str).apply(
            lambda col: col.str.lower().isin(terms)).any(axis=1)
        if mask.any():
            fdf = fdf[mask]
            show = True
            filters_applied = True
        else:
            st.warning("No stations matched your search.")
            filters_applied = True
    if sel_countries:
        mask = fdf['Country'].isin(sel_countries)
        if mask.any():
            fdf = fdf[mask]
            show = True
            filters_applied = True
        else:
            st.warning("No stations in the selected countries")
            filters_applied = True
    if sel_obs:
        mask = fdf['obsTypes'].apply(lambda lst: all(o in lst for o in sel_obs))
        if mask.any():
            fdf = fdf[mask]
            show = True
            filters_applied = True
        else:
            st.warning("No stations have the selected observation types")
            filters_applied = True
    if sel_params:
        # Requires stations to have ALL selected parameters
        mask = fdf['parameters'].apply(lambda lst: all(p in lst for p in sel_params))
        if mask.any():
            fdf = fdf[mask]
            show = True
            filters_applied = True
        else:
            st.warning("No stations have all the selected parameters")
            filters_applied = True

    # Load summary data
    summary = get_summary(df)
    summary_tbl = summary.drop(columns=['obsTypes'], errors='ignore').rename(columns={'Total': 'Station Count'})

    # Only show summary cards when NO filters are applied
    if not filters_applied:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="summary-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Stations</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="summary-card">
                <div class="metric-value">{len(summary)}</div>
                <div class="metric-label">Countries</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Count unique observation types
            all_obs_types = set()
            for obs_list in df['obsTypes']:
                all_obs_types.update(obs_list)
            st.markdown(f"""
            <div class="summary-card">
                <div class="metric-value">{len(all_obs_types)}</div>
                <div class="metric-label">Observation Types</div>
            </div>
            """, unsafe_allow_html=True)

        # Add charts when no filters are applied
        tab1, tab2 = st.tabs(["Observation Types", "Countries"])

        with tab1:
            fig = create_obs_type_chart(df)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = create_country_chart(df)
            st.plotly_chart(fig, use_container_width=True)

    # Only show export if filters are applied
    if filters_applied:
        # Export functionality
        with st.expander("Export Results", expanded=False):
            # Define columns to exclude from export
            EXCLUDE_COLS = ['tags.mgID', 'tags.name', 'search_blob', 'isArchived', 'tags.wmo', 'tags.icao', 'tags.iata',
                            'tags.madisId', 'tags.ghcndID', 'tags.eaukID', 'tags.davisId', 'tags.dtnLegacyID',
                            'tags.dwdID',
                            'tags.faa', 'lastObsTimestamp']

            # Get available columns excluding hidden ones
            available_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

            cols = st.multiselect("Columns to export",
                                  options=available_cols,
                                  default=available_cols,
                                  key="export_cols")

            if cols:
                fmt = st.selectbox("Format:", ["CSV", "TXT", "Excel (single)", "Excel (1 sheet/country)"],
                                   key="fmt")
                df_to_export = fdf[cols]
                if fmt == "CSV":
                    st.download_button("Download", df_to_export.to_csv(index=False),
                                       file_name="stations.csv", use_container_width=True)
                elif fmt == "TXT":
                    st.download_button("Download", df_to_export.to_csv(index=False, sep="\t"),
                                       file_name="stations.txt", use_container_width=True)
                elif fmt == "Excel (single)":
                    try:
                        import xlsxwriter
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                            df_to_export.to_excel(w, sheet_name="Stations", index=False)
                        st.download_button("Download", buf.getvalue(),
                                           file_name="stations.xlsx", use_container_width=True)
                    except ImportError:
                        st.warning(
                            "Excel export requires the xlsxwriter module. Please install it with `pip install xlsxwriter`")
                else:
                    try:
                        import xlsxwriter
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                            for cty, grp in df_to_export.groupby("Country"):
                                grp.to_excel(w, sheet_name=cty[:31] or "Unknown", index=False)
                        st.download_button("Download", buf.getvalue(),
                                           file_name="stations_by_country.xlsx", use_container_width=True)
                    except ImportError:
                        st.warning(
                            "Excel export requires the xlsxwriter module. Please install it with `pip install xlsxwriter`")

    fdf = fdf.dropna(subset=['stationCode', 'name', 'latitude', 'longitude'])
    fdf = fdf[fdf['stationCode'].str.strip() != ""]
    required = ['Country', 'stationCode', 'name', 'latitude', 'longitude', 'elevation', 'obsTypes', 'parameters']
    optional = [c for c in
                ['mgID', 'wmo', 'icao', 'madisId', 'eaukID', 'iata', 'faa', 'dwdID', 'davisId', 'dtnLegacyID',
                 'ghcndID'] if c in fdf.columns]
    raw = fdf[required + optional]
    raw.columns = [c.title().replace('Stationcode', 'Station Code').replace('Obstypes', 'Obs Types') for c in
                   raw.columns]
    results = drop_blank_columns(raw)

    # Move station details to sidebar
    with st.sidebar:

        st.markdown("<div class='panel-header'>🔍 Station Details</div>", unsafe_allow_html=True)

        md = {}  # Initialize empty station metadata
        sel = None

        if not fdf.empty:
            sel = st.selectbox("Select a station:",
                               options=results['Station Code'],
                               key="station_selector",
                               index=None,
                               placeholder="Select station...")
            status_placeholder = st.empty()

            if sel:
                with st.spinner("Fetching station metadata..."):
                    md = fetch_station_metadata(sel, token)

        # If no station selected, show placeholder
        if not md:
            st.markdown("""
            <div class="placeholder-card">
                <div style="font-size: 5rem; margin-bottom: 20px;">🌦️</div>
                <div>Please select a station to view details</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Process station metadata
            p = md.get("properties", {})
            g = md.get("geometry", {})
            tags = p.get("tags", {}) or {}
            last_obs_str = p.get("lastObsTimestamp")

            # Status styling
            status, status_class = "Unknown", ""
            human_time = "N/A"
            if last_obs_str:
                try:
                    last_obs_dt = datetime.fromisoformat(last_obs_str.replace("Z", "+00:00"))
                    delta = datetime.now(timezone.utc) - last_obs_dt
                    if delta.days > 0:
                        human_time = f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
                    elif delta.seconds >= 3600:
                        human_time = f"{delta.seconds // 3600} hour(s) ago"
                    elif delta.seconds >= 60:
                        human_time = f"{delta.seconds // 60} minute(s) ago"
                    else:
                        human_time = "just now"
                    status = "Active" if delta.total_seconds() <= 3 * 86400 else "Inactive"
                    status_class = "status-active" if status == "Active" else "status-inactive"
                except:
                    human_time = last_obs_str

            first_obs_fmt = "-"
            first_obs_str = p.get("firstObsTimestamp")
            if first_obs_str:
                try:
                    dt = datetime.fromisoformat(first_obs_str.replace("Z", "+00:00"))
                    first_obs_fmt = dt.strftime("%B %d, %Y %I:%M %p").lstrip("0")
                except:
                    first_obs_fmt = first_obs_str

            # Build station card
            card_html = f"""
            <div class="station-card">
                <h3>
                    <span>{md.get('stationCode', '-')}</span>
                    <span class="{status_class}">{status}</span>
                </h3>
                <div class="station-property">
                    <div class="property-label">Name:</div>
                    <div class="property-value">{tags.get('name', '-')}</div>
                </div>
                <div class="station-property">
                    <div class="property-label">Coordinates:</div>
                    <div class="property-value">{g.get('coordinates', ['-', '-'])[0]}, {g.get('coordinates', ['-', '-'])[1]}</div>
                </div>
                <div class="station-property">
                    <div class="property-label">Elevation:</div>
                    <div class="property-value">{p.get('elevation', '-')} m</div>
                </div>
                <div class="station-property">
                    <div class="property-label">Obs Types:</div>
                    <div class="property-value">{', '.join(p.get('obsTypes', [])) or '-'}</div>
                </div>
                <div class="station-property">
                    <div class="property-label">First Obs:</div>
                    <div class="property-value">{first_obs_fmt}</div>
                </div>
                <div class="station-property">
                    <div class="property-label">Latest Obs:</div>
                    <div class="property-value">{human_time}</div>
                </div>
            """

            # Add identifiers
            if any(tags.values()):
                card_html += """<div class="station-property">
                    <div class="property-label">Identifiers:</div>
                    <div class="property-value">"""
                for k, v in tags.items():
                    if v and k != "name":
                        card_html += f"<div><strong>{k.replace('_', ' ').title()}:</strong> {v}</div>"
                card_html += "</div></div>"

            card_html += "</div>"

            st.markdown(card_html, unsafe_allow_html=True)

            # Parameter metadata table
            ac = p.get("archiveCounts", {})
            if ac:
                st.markdown("<div class='section-header'>Parameter Metadata</div>", unsafe_allow_html=True)
                param_df = extract_parameter_metadata(ac)

                # Create card for each parameter
                for _, row in param_df.iterrows():
                    param_name = row['Parameter']

                    param_card = f"""
                    <div class="parameter-card">
                        <div class="parameter-header">{param_name}</div>
                        <div class="parameter-row">
                            <div class="parameter-label">First Obs:</div>
                            <div class="parameter-value">{row['First Obs']}</div>
                        </div>
                        <div class="parameter-row">
                            <div class="parameter-label">Latest Obs:</div>
                            <div class="parameter-value">{row['Latest Obs']}</div>
                        </div>
                    </div>
                    """
                    st.markdown(param_card, unsafe_allow_html=True)

                    # Use expander for heatmap with custom label
                    with st.expander(f"{param_name} Availability", expanded=False):
                        # Generate interactive heatmap using Plotly
                        heatmap_fig = generate_heatmap_plotly(ac, param_name)
                        if heatmap_fig:
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                        else:
                            st.info("No data available for heatmap visualization")
            else:
                st.info("No parameter archive metadata available.")

    # Main content area - Map and Table
    if show and not fdf.empty:
        st.markdown("<div class='section-header'>Stations Map</div>", unsafe_allow_html=True)

        # Create enhanced map
        deck = create_station_map(fdf, sel)
        if deck:
            st.pydeck_chart(deck)

        st.markdown(f"<div class='section-header'>{len(results)} Active Station(s)</div>", unsafe_allow_html=True)

        # Main table
        height = min(35 * len(results) + 50, 1000)
        st.dataframe(results.reset_index(drop=True), hide_index=True, use_container_width=True, height=height)

    else:
        # Only show country table when NO filters are applied
        if not filters_applied:
            st.markdown("<div class='section-header'>Active Stations per Country</div>", unsafe_allow_html=True)
            height = min(35 * len(summary_tbl) + 40, 900)
            st.dataframe(summary_tbl, use_container_width=True, height=height, hide_index=True)


def top_nav_bar():
    """Create a fixed top navigation bar with access level dropdown in popover"""
    st.markdown(
        """
        <style>
        /* Top navigation bar styling */
        .hamburger-menu {
            font-size: 1.8rem;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 4px;
            transition: all 0.3s;
        }
        .hamburger-menu:hover {
            background-color: #f0f9ff;
            transform: scale(1.1);
        }
        .access-level {
            font-weight: 500;
            color: #0072b5;
            background-color: #e6f7ff;
            padding: 4px 12px;
            border-radius: 4px;
            border: 1px solid #91d5ff;
        }
        .logout-btn {
            background: none;
            border: 1px solid #ddd;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
        }
        .logout-btn:hover {
            background-color: #f8f8f8;
            border-color: #0072b5;
            color: #0072b5;
        }
        /* Add padding to the main content so it doesn't hide behind the top bar */
        .stApp {
            margin-top: 4.5rem;
        }
        /* Position popover to the right */
        .stPopover {
            right: 100px !important;
            left: 0px !important;
        }
        .popover-content {
            min-width: 250px;
            padding: 15px;
        }
        .popover-header {
            font-size: 1.2rem;
            font-weight: bold;
            color: #0072b5;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the top bar
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        # Hamburger menu popover
        with st.popover("☰", use_container_width=False):
            # Popover content without inner hamburger icon
            st.markdown('<div class="popover-content">', unsafe_allow_html=True)
            st.markdown('<div class="popover-header">Settings</div>', unsafe_allow_html=True)

            # Access level selection
            st.markdown("Access Level")
            access_options = ["Internal", "Core", "Plus"]
            current_access = st.session_state.get("access_level", "Internal")

            # Create dropdown for access level
            new_access = st.selectbox(
                "Choose credentials level:",
                access_options,
                index=access_options.index(current_access),
                key="popover_access_dropdown"
            )

            # Update session state if changed
            if new_access != current_access:
                st.session_state.access_level = new_access
                st.rerun()

            # Dark mode toggle
            dark_mode = st.toggle("Dark Mode",
                                  value=st.session_state.get("dark_mode", False),
                                  key="dark_mode_toggle")
            if dark_mode != st.session_state.get("dark_mode", False):
                st.session_state.dark_mode = dark_mode
                st.rerun()

            # Refresh data button
            if st.button("Refresh Data", use_container_width=True, key="refresh_btn"):
                st.cache_data.clear()
                st.rerun()

            # Logout button
            if st.button("Logout", use_container_width=True, key="logout_btn"):
                st.info("Logout functionality would go here")

            st.markdown('</div>', unsafe_allow_html=True)  # Close popover-content

    with col2:
        st.markdown(f"<h1 style='text-align: center; margin: 0; color: #0072b5;'>DTN Station Explorer</h1>",
                    unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end; align-items: center; height: 100%;'>
            <span class="access-level">{st.session_state.access_level} Access</span>
        </div>
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="DTN Station Explorer",
        layout="wide",
        page_icon="🌦️",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if "access_level" not in st.session_state:
        st.session_state.access_level = "Internal"
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    # Add the top navigation bar
    top_nav_bar()

    # Load data and show dashboard
    try:
        with st.spinner("Loading station data..."):
            df, token = get_stations_by_access(st.session_state.access_level)

        if df.empty:
            st.error("No station data available. Please check your credentials and try again.")
            if st.button("Retry"):
                st.cache_data.clear()
                st.rerun()
        else:
            show_dashboard(df, token)
    except Exception as e:
        st.error(f"Could not load `{st.session_state.access_level}` data: {e}")
        if st.button("Retry"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()