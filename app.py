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
import hashlib
from functools import wraps

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'login_attempted' not in st.session_state:
    st.session_state.login_attempted = False


# Login function with modern floating label design
def login():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Hide Streamlit's default elements that might cause unwanted rectangles */
        .stApp > header {
            display: none !important;
        }

        .stApp > div:first-child {
            padding-top: 0 !important;
        }

        /* Hide any empty containers or divs that might show up */
        .stMarkdown:empty,
        .stContainer:empty,
        div[data-testid="stMarkdownContainer"]:empty {
            display: none !important;
        }

        /* Hide the main container padding */
        .main .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        /* Main container styling */
        .login-container {
            max-width: 420px;
            margin: 80px auto;
            padding: 40px;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.1),
                0 2px 8px rgba(0, 0, 0, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
            overflow: hidden;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Animated background gradient */
        .login-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, 
                rgba(0, 114, 181, 0.1) 0%, 
                rgba(0, 169, 157, 0.1) 25%,
                rgba(138, 43, 226, 0.1) 50%,
                rgba(255, 20, 147, 0.1) 75%,
                rgba(0, 114, 181, 0.1) 100%);
            animation: gradientShift 8s ease-in-out infinite;
            z-index: -1;
        }

        @keyframes gradientShift {
            0%, 100% { transform: rotate(0deg) scale(1); }
            25% { transform: rotate(90deg) scale(1.1); }
            50% { transform: rotate(180deg) scale(1); }
            75% { transform: rotate(270deg) scale(1.1); }
        }

        /* Title styling */
        .login-title {
            text-align: center;
            background: linear-gradient(135deg, #0072b5 0%, #00a99d 50%, #8a2be2 100%);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 40px;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.5px;
            position: relative;
        }

        .login-title::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, #0072b5, #00a99d);
            border-radius: 2px;
            animation: titleUnderline 2s ease-out;
        }

        @keyframes titleUnderline {
            0% { width: 0; opacity: 0; }
            100% { width: 60px; opacity: 1; }
        }

        /* Streamlit input container styling */
        .stTextInput {
            margin-bottom: 24px;
        }

        /* Style the input field */
        .stTextInput > div > div > input {
            padding: 16px !important;
            border: 2px solid rgba(0, 114, 181, 0.2) !important;
            border-radius: 12px !important;
            background: rgba(255, 255, 255, 0.8) !important;
            font-size: 16px !important;
            font-weight: 400 !important;
            color: #2d3748 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            backdrop-filter: blur(10px) !important;
            box-shadow: none !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #0072b5 !important;
            background: rgba(255, 255, 255, 0.95) !important;
            box-shadow: 
                0 0 0 4px rgba(0, 114, 181, 0.1),
                0 4px 12px rgba(0, 114, 181, 0.15) !important;
            transform: translateY(-2px) !important;
            outline: none !important;
        }

        /* Style placeholders */
        .stTextInput > div > div > input::placeholder {
            color: rgba(0, 114, 181, 0.6) !important;
            font-weight: 500 !important;
            transition: opacity 0.3s ease !important;
        }

        .stTextInput > div > div > input:focus::placeholder {
            opacity: 0.3 !important;
        }

        /* Hide the default Streamlit labels */
        .stTextInput > label {
            display: none !important;
        }

        /* Login button styling */
        .stButton > button {
            width: 100% !important;
            background: linear-gradient(135deg, #0072b5 0%, #00a99d 50%, #8a2be2 100%) !important;
            background-size: 200% 200% !important;
            color: white !important;
            border: none !important;
            padding: 16px 32px !important;
            border-radius: 12px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 4px 16px rgba(0, 114, 181, 0.3) !important;
            text-transform: uppercase !important;
        }

        .stButton > button:hover {
            background-position: 100% 0 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(0, 114, 181, 0.4) !important;
        }

        .stButton > button:active {
            transform: translateY(0) !important;
            box-shadow: 0 4px 12px rgba(0, 114, 181, 0.3) !important;
        }

        /* Button ripple effect */
        .stButton > button::before {
            content: '' !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            width: 0 !important;
            height: 0 !important;
            border-radius: 50% !important;
            background: rgba(255, 255, 255, 0.3) !important;
            transform: translate(-50%, -50%) !important;
            transition: width 0.6s, height 0.6s !important;
        }

        .stButton > button:active::before {
            width: 300px !important;
            height: 300px !important;
        }

        /* Error message styling */
        .stAlert {
            border-radius: 8px !important;
            border-left: 4px solid #e53e3e !important;
        }

        /* Floating particles animation */
        .login-container::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: 
                radial-gradient(circle at 20% 30%, rgba(0, 114, 181, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(0, 169, 157, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 60% 20%, rgba(138, 43, 226, 0.1) 0%, transparent 50%);
            animation: particleFloat 12s ease-in-out infinite;
            pointer-events: none;
            z-index: -1;
        }

        @keyframes particleFloat {
            0%, 100% { 
                background-position: 0% 0%, 100% 100%, 50% 0%;
                opacity: 0.5;
            }
            33% { 
                background-position: 30% 30%, 70% 70%, 80% 30%;
                opacity: 0.8;
            }
            66% { 
                background-position: 70% 10%, 30% 80%, 20% 60%;
                opacity: 0.6;
            }
        }

        /* Responsive design */
        @media (max-width: 480px) {
            .login-container {
                margin: 40px 20px;
                padding: 30px 24px;
            }

            .login-title {
                font-size: 24px;
            }
        }

        /* Smooth page entry animation */
        .login-container {
            animation: slideInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }

        @keyframes slideInUp {
            0% {
                opacity: 0;
                transform: translateY(30px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>""", unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)

    with st.form("login_form"):
        # Username field with placeholder
        username = st.text_input("Username", placeholder="Username", key="username_input", label_visibility="collapsed")

        # Password field with placeholder
        password = st.text_input("Password", type="password", placeholder="Password", key="password_input",
                                 label_visibility="collapsed")

        submitted = st.form_submit_button("Sign In")

        if submitted:
            if username == os.getenv("USERNAME") and password == os.getenv("PASSWORD"):
                st.session_state.authenticated = True
                st.session_state.login_attempted = False
                st.rerun()
            else:
                st.session_state.login_attempted = True
                st.error("Invalid username or password")

    st.markdown('</div>', unsafe_allow_html=True)

# Show login screen if not authenticated
if not st.session_state.authenticated:
    login()
    st.stop()

load_dotenv()  # <-- this makes os.getenv pick up your .env

CLIENT_ID_INTERNAL = os.getenv("CLIENT_ID_INTERNAL")
CLIENT_SECRET_INTERNAL = os.getenv("CLIENT_SECRET_INTERNAL")
CLIENT_ID_CORE = os.getenv("CLIENT_ID_CORE")
CLIENT_SECRET_CORE = os.getenv("CLIENT_SECRET_CORE")
CLIENT_ID_PLUS = os.getenv("CLIENT_ID_PLUS")
CLIENT_SECRET_PLUS = os.getenv("CLIENT_SECRET_PLUS")


# --- Helper functions ---
def get_token(client_id, client_secret):
    url = 'https://api.auth.dtn.com/v1/tokens/authorize'
    payload = {"grant_type": "client_credentials", "client_id": client_id,
               "client_secret": client_secret, "audience": "https://weather.api.dtn.com/observations"}
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return (data.get('data') or data).get('access_token')


@lru_cache(maxsize=None)
def reverse_geocode_cached(lat, lon):
    return reverse_geocode.search([(lat, lon)])[0]['country']


# Custom hash function for DataFrames with unhashable types
def hash_dataframe(df):
    """Create a hash for a DataFrame that may contain unhashable types"""
    # Convert any unhashable types to hashable representations
    df_copy = df.copy()

    # Convert lists and tuples to strings for hashing
    for col in df_copy.columns:
        if df_copy[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            df_copy[col] = df_copy[col].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)

    # Create a hash of the DataFrame's content
    return hashlib.md5(pd.util.hash_pandas_object(df_copy).values.tobytes()).hexdigest()


# Custom caching implementation for DataFrames with unhashable types
def cached_station_data(access_choice):
    """Custom caching implementation for station data"""
    cache_key = f"station_data_{access_choice}"

    # Check if we have a cached version
    if cache_key in st.session_state:
        cached_data = st.session_state[cache_key]
        cached_hash = st.session_state.get(f"{cache_key}_hash")

        # Verify the hash to ensure data integrity
        current_hash = hash_dataframe(cached_data[0])
        if cached_hash == current_hash:
            return cached_data

    # If not cached or hash mismatch, fetch new data
    creds = {
        "Internal": (CLIENT_ID_INTERNAL, CLIENT_SECRET_INTERNAL),
        "Core": (CLIENT_ID_CORE, CLIENT_SECRET_CORE),
        "Plus": (CLIENT_ID_PLUS, CLIENT_SECRET_PLUS),
    }
    client_id, client_secret = creds.get(access_choice, creds["Internal"])
    token = get_token(client_id, client_secret)
    url = 'https://obs.api.dtn.com/v1/observations/stations'
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params={
        'by': 'boundingBox', 'minLat': '-90', 'maxLat': '90', 'minLon': '-180', 'maxLon': '180',
        'obsTypes': 'RWIS,AG,METAR,SYNOP,BUOY,Citizen,SHIP,Hydro,Others,HFM,GHCND,Customer,ISD'
    })
    resp.raise_for_status()
    df = pd.json_normalize(resp.json())
    df['Country'] = df.apply(lambda r: reverse_geocode_cached(r.latitude, r.longitude), axis=1)
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

    # Handle list-type columns safely - convert to tuples for hashability
    df['stationCode'] = df.get('stationCode', pd.Series([""] * len(df))).fillna("")

    # Convert lists to tuples for hashability
    def safe_convert_to_tuple(x):
        if isinstance(x, list):
            return tuple(x)
        elif pd.notna(x):
            return (str(x),)
        else:
            return tuple()

    df['obsTypes'] = df.get('obsTypes', pd.Series([[]] * len(df))).apply(safe_convert_to_tuple)
    df['parameters'] = df.get('parameters', pd.Series([[]] * len(df))).apply(safe_convert_to_tuple)

    df['search_blob'] = df.astype(str).apply(lambda row: ' '.join(row.values).lower(), axis=1)
    df.reset_index(drop=True, inplace=True)

    # Cache the result
    st.session_state[cache_key] = (df, token)
    st.session_state[f"{cache_key}_hash"] = hash_dataframe(df)

    return df, token


# Use the custom caching function instead of st.cache_data
def get_stations_by_access(access_choice: str):
    with st.spinner("Fetching station data…"):
        return cached_station_data(access_choice)




# Disable Arrow serialization by default to prevent mixed object type crashes
os.environ["STREAMLIT_DATAFRAME_USE_ARROW"] = "0"


def safe_cache_data(*args, **kwargs):
    """
    A drop-in replacement for st.cache_data that handles DataFrames with unhashable types.

    Features:
    - Overrides the default hash function for Pandas DataFrames
    - By default, hashes only DataFrame shape + column names (not contents)
    - Allows passing extra hash_funcs like st.cache_data
    - Automatically disables Arrow serialization
    """
    # Extract hash_funcs from kwargs or use empty dict
    hash_funcs = kwargs.pop('hash_funcs', {})

    # Define our custom DataFrame hash function
    def _hash_dataframe(df):
        """Hash a DataFrame based on its shape and column names only"""
        try:
            # Try to use the default hash if possible
            return pd.util.hash_pandas_object(df).values.tobytes()
        except (TypeError, ValueError):
            # Fall back to shape and column names for unhashable DataFrames
            return (df.shape, tuple(df.columns))

    # Add our custom hash function for DataFrames if not already provided
    if pd.DataFrame not in hash_funcs:
        hash_funcs[pd.DataFrame] = _hash_dataframe

    # Add support for other common unhashable types
    if list not in hash_funcs:
        hash_funcs[list] = lambda x: str(x)
    if dict not in hash_funcs:
        hash_funcs[dict] = lambda x: str(sorted(x.items())) if x else "empty_dict"
    if set not in hash_funcs:
        hash_funcs[set] = lambda x: str(sorted(x)) if x else "empty_set"
    if tuple not in hash_funcs:
        hash_funcs[tuple] = lambda x: str(x)

    # Pass the updated hash_funcs to st.cache_data
    return st.cache_data(*args, **kwargs, hash_funcs=hash_funcs)


# Example usage in your code:
@safe_cache_data
def get_summary(df):
    # Create a copy with lists instead of tuples for exploding
    df_with_lists = df.copy()
    df_with_lists['obsTypes'] = df_with_lists['obsTypes'].apply(list)
    expl = df_with_lists[['Country', 'obsTypes']].explode('obsTypes')
    pivot = expl.pivot_table(index='Country', columns='obsTypes', aggfunc='size', fill_value=0)
    pivot['Total'] = pivot.sum(axis=1)
    pivot = pivot[pivot.index.notna()].sort_index()
    return pivot.reset_index()


# You can also pass custom hash functions if needed
@safe_cache_data(hash_funcs={pd.Series: lambda x: (len(x), tuple(x.dtype))})
def my_function_with_series(data):
    # Your function logic here
    return processed_data

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


def generate_heatmap(archive_counts, param):
    """Generate a heatmap visualization for parameter availability using gradient colors"""
    # Get counts for the parameter
    counts = archive_counts.get(param, {})
    if not counts:
        return "<div>No data available</div>"

    # Extract all years and months
    all_months = {}
    min_year = float('inf')
    max_year = 0
    monthly_counts = []

    for month_key, count in counts.items():
        try:
            year, month = map(int, month_key.split('-'))
            min_year = min(min_year, year)
            max_year = max(max_year, year)
            all_months[(year, month)] = count
            monthly_counts.append(count)
        except:
            continue

    if min_year == float('inf'):
        return "<div>No valid data</div>"

    # Find max count for normalization
    max_count = max(monthly_counts) if monthly_counts else 1

    # Generate heatmap HTML in chronological order
    html = ['<div class="heatmap-grid">']

    # Create columns for each year
    for year in range(min_year, max_year + 1):
        for month in range(1, 13):
            count = all_months.get((year, month), 0)
            # Calculate percentage relative to max count
            percentage = (count / max_count) * 100 if max_count > 0 else 0

            # Determine gradient color based on percentage
            # Red (low) to Green (high) gradient
            if count == 0:
                color = "#f0f0f0"  # Light gray for no data
            else:
                # Calculate hue: 0° (red) to 120° (green)
                hue = 120 * (percentage / 100)
                color = f"hsl({hue}, 100%, 45%)"

            month_name = calendar.month_abbr[month]
            tooltip = f"{month_name} {year}: {count} observations ({percentage:.1f}%)"
            html.append(
                f'<div class="heatmap-cell" title="{tooltip}" style="background-color: {color}">'
                f'</div>'
            )
    html.append('</div>')

    return "".join(html)


def drop_blank_columns(df):
    return df.loc[:, df.apply(lambda col: col.replace("", pd.NA).dropna().astype(str).str.strip().ne("").any())]


def fetch_station_metadata(code, token, retries=10, status_placeholder=None):
    url = "https://obs.api.dtn.com/v2/observations/stations"
    params = {"by": "stationCodes", "stationCodes": code, "isArchive": "false", "archiveCounts": "true"}
    for attempt in range(retries):
        try:
            if attempt > 0 and status_placeholder:
                status_placeholder.warning(f"Retrying... Attempt {attempt + 1} of {retries}")
            resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=20)
            resp.raise_for_status()
            if status_placeholder: status_placeholder.empty()
            feats = resp.json().get("features", [])
            return feats[0] if feats else {}
        except Exception as e:
            logging.warning(f"Retry {attempt + 1} failed: {e}")
            sleep(2 ** attempt)
            if attempt == retries - 1 and status_placeholder:
                status_placeholder.error(f"Failed after {retries} retries.")
                return {}
def get_parameter_counts(archive_counts):
    """Create a DataFrame with parameter observation counts"""
    recs = []
    for param, months in (archive_counts or {}).items():
        total = sum(months.values())
        recs.append({
            "Parameter": param,
            "Total Observations": total
        })
    return pd.DataFrame(recs).sort_values("Parameter")

# --- UI and main logic ---
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
            options=sorted(df['Country'].unique()),  # Added sorted()
            help="Filter stations by their country location"
        )
    with col4:
        # Convert tuples to lists for the multiselect options
        obs_types = sorted({o for row in df['obsTypes'] for o in row})
        sel_obs = st.multiselect(
            "Filter by Observation Types:",
            options=obs_types,
            help="Show only stations reporting selected observation types"
        )
    with col5:
        # Convert tuples to lists for the multiselect options
        params = sorted({p for row in df['parameters'] for p in row})
        sel_params = st.multiselect(
            "Filter by Parameters:",
            options=params,
            help="Show only stations reporting selected parameters"
        )

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
        # Handle tuples in filter
        mask = fdf['obsTypes'].apply(lambda tup: all(o in tup for o in sel_obs))
        if mask.any():
            fdf = fdf[mask]
            show = True
            filters_applied = True
        else:
            st.warning("No stations have the selected observation types")
            filters_applied = True
    if sel_params:
        # Handle tuples in filter
        mask = fdf['parameters'].apply(lambda tup: all(p in tup for p in sel_params))
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
        col1, col2 = st.columns(2)
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

                # Convert tuples to strings for export
                for col in df_to_export.columns:
                    if df_to_export[col].dtype == object and df_to_export[col].apply(
                            lambda x: isinstance(x, tuple)).any():
                        df_to_export[col] = df_to_export[col].apply(
                            lambda x: ', '.join(x) if isinstance(x, tuple) else x)

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
    required = ['stationCode', 'name', 'latitude', 'longitude', 'elevation', 'obsTypes', 'parameters']
    optional = [c for c in
                ['Country', 'mgID', 'wmo', 'icao', 'madisId', 'eaukID', 'iata', 'faa', 'dwdID', 'davisId', 'dtnLegacyID',
                 'ghcndID'] if c in fdf.columns]
    raw = fdf[required + optional]
    raw.columns = [c.title().replace('Stationcode', 'Station Code').replace('Obstypes', 'Obs Types') for c in
                   raw.columns]

    # Convert tuples to strings for display - FIXED SettingWithCopyWarning
    for col in raw.columns:
        if raw[col].dtype == object and raw[col].apply(lambda x: isinstance(x, tuple)).any():
            raw.loc[:, col] = raw[col].apply(lambda x: ', '.join(x) if isinstance(x, tuple) else x)

    results = drop_blank_columns(raw)

    # Move station details to sidebar
    with st.sidebar:

        st.markdown("<div class='panel-header'>Station Details</div>", unsafe_allow_html=True)

        md = {}  # Initialize empty station metadata
        sel = None

        if not fdf.empty:
            # Convert tuples to strings for display in the selectbox
            display_options = results['Station Code']
            sel = st.selectbox("Select a station:",
                               options=display_options,
                               key="station_selector",
                               index=None,
                               placeholder="Select station...")
            status_placeholder = st.empty()

            if sel:
                md = fetch_station_metadata(sel, token, status_placeholder=status_placeholder)

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
                        # Wrap in custom styling
                        st.markdown(f"""
                        <div class="custom-expander">
                            <div class="custom-expander-content">
                                {generate_heatmap(ac, param_name)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Add download button for parameter counts
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                param_counts_df = get_parameter_counts(ac)
                csv = param_counts_df.to_csv(index=False)
                st.download_button(
                    label="Download Parameter Counts",
                    data=csv,
                    file_name=f"{sel}_parameter_counts.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No parameter archive metadata available.")

    # Main content area - Map and Table
    if show and not fdf.empty:
        st.markdown("<div class='section-header'>Stations Map</div>", unsafe_allow_html=True)
        layer = pdk.Layer("ScatterplotLayer", data=fdf, get_position='[longitude, latitude]',
                          get_radius=20, get_fill_color=[1, 164, 159, 180], pickable=True, auto_highlight=True,
                          radius_min_pixels=2)

        vs = compute_view(fdf[['longitude', 'latitude']].dropna().values.tolist())
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=vs,
                                 tooltip={"html": "<b>Station Code:</b> {stationCode}<br/><b>Name:</b> {name}",
                                          "style": {"backgroundColor": "#0072b5", "color": "white"}}))

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
    with st.container():
        col1, col2, col3 = st.columns([1, 4, 1])

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

                # Logout button
                if st.button("Logout", use_container_width=True, key="logout_btn"):
                    st.session_state.authenticated = False
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)  # Close popover-content


def main():
    st.set_page_config(
        page_title="DTN Station Explorer",
        layout="wide",
        page_icon="🌦️",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    if "access_level" not in st.session_state:
        st.session_state.access_level = "Internal"
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    # Add the top navigation bar
    top_nav_bar()

    # Header
    st.markdown(
        f"<div style='margin-top: 20px; margin-bottom: 20px;'>"
        f"<h1 style='color: #0072b5; font-size:1.8rem; border-bottom: 2px solid #00a99d; padding-bottom: 10px;'>"
        f"{st.session_state.access_level} Access</h1>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Load data and show dashboard
    try:
        df, token = get_stations_by_access(st.session_state.access_level)
        show_dashboard(df, token)
    except Exception as e:
        st.error(f"Could not load `{st.session_state.access_level}` data: {e}")


if __name__ == "__main__":
    main()