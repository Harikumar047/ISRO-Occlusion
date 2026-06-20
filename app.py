import streamlit as st
import networkx as nx
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from data_loader import download_road_network, download_poi_data
from occlusion import simulate_occlusion
from gnn_model import (
    train_link_predictor,
    generate_candidate_edges,
    predict_reconstruction_links,
    heuristic_reconstruction,
    mst_disjoint_set_healing
)
from criticality import compute_criticality, run_stress_test, compute_connectivity_ratio, run_node_ablation_test

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Occlusion-Robust Road Network Reconstruction GNN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@600;700&display=swap');
    
    :root {
        --bg-deep: #0B1120;
        --bg-panel: #141B2D;
        --bg-panel-hover: #1B2438;
        --border-subtle: #243049;
        --signal-cyan: #22D3EE;
        --signal-amber: #F59E0B;
        --signal-green: #34D399;
        --signal-red: #F87171;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }
    
    /* Main App Container Styling */
    .stApp {
        background-color: var(--bg-deep) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
    }
    
    /* JetBrains Mono for telemetry/numerical data */
    .telemetry-numeric, .metric-value, .status-name, td:has(span), td, code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Metric Cards Grid Layout */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .metric-card {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 120px;
    }
    
    .metric-card:hover {
        border-color: var(--signal-cyan) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(34, 211, 238, 0.15) !important;
    }
    
    .metric-title {
        color: var(--text-muted) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 8px !important;
    }
    
    .metric-value {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        line-height: 1.1 !important;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .metric-info {
        color: var(--text-secondary) !important;
        font-size: 0.75rem !important;
        margin-top: 6px !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .metric-delta-pos {
        color: var(--signal-green) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        margin-top: 6px !important;
    }
    
    .metric-delta-neg {
        color: var(--signal-red) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        margin-top: 6px !important;
    }
    
    /* Active Reconstruction Method Slim Status Bar */
    .method-status-bar {
        background-color: var(--bg-panel);
        padding: 12px 20px;
        border-radius: 6px;
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    .status-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        color: var(--text-muted);
    }
    .status-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.02em;
    }
    
    /* Custom Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    
    .sidebar-divider {
        border-top: 1px solid var(--border-subtle);
        margin: 20px 0;
    }
    
    .sidebar-section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 12px;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Sidebar glow button override */
    [data-testid="stSidebar"] div.stButton > button {
        background-color: var(--signal-cyan) !important;
        color: var(--bg-deep) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 0 10px rgba(34, 211, 238, 0.2) !important;
        transition: all 0.3s ease !important;
        padding: 10px 20px !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stSidebar"] div.stButton > button:hover {
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.6) !important;
        transform: translateY(-1px) !important;
        background-color: var(--signal-cyan) !important;
        color: var(--bg-deep) !important;
    }
    
    /* Streamlit Tabs Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.05em !important;
        color: var(--text-muted) !important;
        padding: 10px 16px !important;
        background-color: transparent !important;
        border: none !important;
        transition: color 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--signal-cyan) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--signal-cyan) !important;
    }
    .stTabs [data-baseweb="tab-highlight-pointer"] {
        background-color: var(--signal-cyan) !important;
        height: 2px !important;
    }
    
    /* Map Layout Components */
    .map-legend-header {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-bottom: none;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .legend-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text-secondary);
        letter-spacing: 0.05em;
    }
    .legend-items {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
    }
    .legend-item {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: var(--text-secondary);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    
    iframe[title="streamlit_folium.st_folium"] {
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-bottom-left-radius: 10px !important;
        border-bottom-right-radius: 10px !important;
        background-color: var(--bg-panel) !important;
    }
    
    /* Telemetry Info Panels */
    .telemetry-panel {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .panel-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: var(--signal-amber);
        margin-bottom: 8px;
    }
    .panel-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 12px;
    }
    .panel-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0;
    }
    .status-awaiting {
        border-left: 4px solid var(--signal-amber) !important;
    }
    
    .section-spacer {
        margin-top: 24px;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    /* Styling Streamlit Standard Tables and DataFrames */
    .stDataFrame {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background-color: var(--bg-panel) !important;
    }
    
    /* Custom buttons in main area */
    div.stButton > button {
        background-color: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border-radius: 6px !important;
    }
    div.stButton > button:hover {
        border-color: var(--signal-cyan) !important;
        color: var(--signal-cyan) !important;
        background-color: var(--bg-panel-hover) !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get edge coordinate list for Folium
def get_edge_coords(G, u, v, data):
    if "geometry" in data:
        # coords is list of (lon, lat)
        coords = list(data["geometry"].coords)
        return [[lat, lon] for lon, lat in coords]
    else:
        u_lat, u_lon = G.nodes[u]["y"], G.nodes[u]["x"]
        v_lat, v_lon = G.nodes[v]["y"], G.nodes[v]["x"]
        return [[u_lat, u_lon], [v_lat, v_lon]]

# Helper function to generate styled SVG confidence indicator
def make_confidence_indicator_html(prob, size=32, text_style=""):
    # prob is float between 0 and 1
    percent = int(round(prob * 100))
    
    # Color mapping: green (>=80%), amber (60-80%), red (<60%)
    if prob >= 0.8:
        color = "var(--signal-green)"
    elif prob >= 0.6:
        color = "var(--signal-amber)"
    else:
        color = "var(--signal-red)"
        
    r = 11
    circ = 2 * 3.14159265 * r
    stroke_dashoffset = circ - (prob * circ)
    
    svg = f"""
    <div style="display: inline-flex; align-items: center; gap: 8px; vertical-align: middle; {text_style}">
        <svg width="{size}" height="{size}" viewBox="0 0 32 32" style="transform: rotate(-90deg); overflow: visible;">
            <!-- Background circle -->
            <circle cx="16" cy="16" r="{r}" stroke="var(--border-subtle)" stroke-width="2.5" fill="none" />
            <!-- Active progress arc -->
            <circle cx="16" cy="16" r="{r}" stroke="{color}" stroke-width="2.5" fill="none"
                    stroke-dasharray="{circ:.2f}" stroke-dashoffset="{stroke_dashoffset:.2f}" stroke-linecap="round" />
        </svg>
        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.95rem;">{percent}%</span>
    </div>
    """
    return svg

# Helper function to generate styled SVG criticality (CCI) indicator
def make_cci_indicator_html(cci_val, size=32):
    # cci_val is float (typically 0 to 1)
    # Reverse color coding for criticality: high CCI is danger (red), low is safe (green)
    if cci_val >= 0.8:
        color = "var(--signal-red)"
    elif cci_val >= 0.5:
        color = "var(--signal-amber)"
    else:
        color = "var(--signal-green)"
        
    r = 11
    circ = 2 * 3.14159265 * r
    # Cap cci_val at 1.0 for visual display
    display_val = min(max(cci_val, 0.0), 1.0)
    stroke_dashoffset = circ - (display_val * circ)
    
    svg = f"""
    <div style="display: inline-flex; align-items: center; gap: 8px; vertical-align: middle;">
        <svg width="{size}" height="{size}" viewBox="0 0 32 32" style="transform: rotate(-90deg); overflow: visible;">
            <!-- Background circle -->
            <circle cx="16" cy="16" r="{r}" stroke="var(--border-subtle)" stroke-width="2.5" fill="none" />
            <!-- Active progress arc -->
            <circle cx="16" cy="16" r="{r}" stroke="{color}" stroke-width="2.5" fill="none"
                    stroke-dasharray="{circ:.2f}" stroke-dashoffset="{stroke_dashoffset:.2f}" stroke-linecap="round" />
        </svg>
        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.95rem;">{cci_val:.4f}</span>
    </div>
    """
    return svg

# Title Panel
st.markdown("""
<div class="app-header" style="margin-bottom: 24px;">
    <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.15em; color: var(--text-muted); text-transform: uppercase;">
        URBAN RESILIENCE INTELLIGENCE SYSTEM
    </div>
    <h1 style="margin: 4px 0 8px 0; font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; color: var(--text-primary);">
        🛣️ Occlusion-Robust Road Network Reconstruction
    </h1>
    <div style="font-family: 'Inter', sans-serif; font-size: 0.95rem; color: var(--text-secondary); margin-bottom: 16px;">
        GNN Link Prediction & Confidence-Quantified Criticality Analysis
    </div>
    <div style="height: 2px; background-color: var(--signal-cyan); width: 100%;"></div>
</div>
""", unsafe_allow_html=True)

# Session State Initialization
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
    st.session_state.place_name = ""
    st.session_state.G_original = None
    st.session_state.G_observed = None
    st.session_state.removed_edges = []
    st.session_state.occlusion_centers = []
    st.session_state.gnn_history = []
    st.session_state.reconstructed = []
    st.session_state.model_trained = False
    st.session_state.poi_data = None
    st.session_state.last_stress_test_loss = None
# Sidebar Controls
st.sidebar.markdown('<div class="sidebar-section-header">⚙️ CONFIGURATION PANELS</div>', unsafe_allow_html=True)

# 1. Dataset Selection
st.sidebar.markdown('<div class="sidebar-section-header">🛰️ 1. DATA SETUP</div>', unsafe_allow_html=True)
preset_cities = ["Panaji, Goa, India", "Dehradun, Uttarakhand, India", "Pondicherry, India", "Custom Address"]
city_selection = st.sidebar.selectbox("Select Target Region", preset_cities, index=0)

if city_selection == "Custom Address":
    target_address = st.sidebar.text_input("Enter Address/City Query", value="Madikeri, Karnataka, India")
else:
    target_address = city_selection

# 2. Occlusion Simulation Settings
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-header">☁️ 2. SATELLITE OCCLUSION</div>', unsafe_allow_html=True)
num_clouds = st.sidebar.slider("Number of Cloud Centers", min_value=1, max_value=5, value=2, step=1)
cloud_radius = st.sidebar.slider("Cloud Radius (m)", min_value=100, max_value=800, value=300, step=50)
deletion_ratio = st.sidebar.slider("Edge Occlusion Ratio", min_value=0.1, max_value=1.0, value=0.7, step=0.05)

# 3. GNN Model & Training Settings
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-header">🧠 3. GNN LINK PREDICTION</div>', unsafe_allow_html=True)
gnn_type = st.sidebar.selectbox("GNN Architecture", ["GraphSAGE", "GAT"])
training_epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=150, value=80, step=10)
learning_rate = 0.01  # standard
candidate_dist = st.sidebar.slider("Max Search Distance for Roads (m)", min_value=100, max_value=500, value=250, step=25)

# 4. Reconstruction Method
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-header">⚙️ 4. RECONSTRUCTION METHOD</div>', unsafe_allow_html=True)
recon_method = st.sidebar.radio("Method", ["GNN Link Prediction", "MST + Disjoint Set (Baseline)", "Heuristic Stitching"])

# 5. Criticality Weights
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-header">⚖️ 5. CRITICALITY WEIGHTS</div>', unsafe_allow_html=True)
st.sidebar.markdown("Adjust CCI formula weights:")
w1 = st.sidebar.slider("Betweenness (w1)", 0.0, 1.0, 0.5, 0.1)
w2 = st.sidebar.slider("Degree (w2)", 0.0, 1.0, 0.3, 0.1)
w3 = st.sidebar.slider("Hospital Access (w3)", 0.0, 1.0, 0.1, 0.1)
w4 = st.sidebar.slider("POI Density (w4)", 0.0, 1.0, 0.1, 0.1)
cci_weights = {"w1": w1, "w2": w2, "w3": w3, "w4": w4}

# Button to trigger full pipeline run
run_btn = st.sidebar.button("🚀 Run Reconstruction Pipeline", use_container_width=True)
# Auto-initialize with default if not run yet
if not st.session_state.pipeline_run and not run_btn:
    st.markdown("""
    <div class="telemetry-panel status-awaiting">
        <div class="panel-eyebrow">SYSTEM STATUS</div>
        <div class="panel-title">AWAITING PIPELINE EXECUTION</div>
        <p class="panel-text">Click the 'Run Reconstruction Pipeline' button in the sidebar to execute on Panaji (default). Auto-initialization is acquiring satellite road networks, simulating cloud occlusion, and training the GNN link predictor for a live demonstration.</p>
    </div>
    """, unsafe_allow_html=True)
    # We will trigger the initialization automatically with cached graph to show immediate visual demo
    with st.spinner("INITIALIZING DEMONSTRATION SEQUENCE: PANAJI, GOA, INDIA..."):
        try:
            place = "Panaji, Goa, India"
            G_orig = download_road_network(place)
            poi_data = download_poi_data(place)
            G_obs, rem_edges, centers = simulate_occlusion(G_orig, num_centers=2, radius_meters=300, occlusion_ratio=0.7, seed=42)
            
            # Fast train for initialization (30 epochs)
            encoder, predictor, history = train_link_predictor(
                G_obs, rem_edges, centers, model_type="SAGE", epochs=30, seed=42
            )
            
            candidates = generate_candidate_edges(G_obs, centers, max_dist=200)
            reconstructed = predict_reconstruction_links(encoder, predictor, G_obs, candidates)
            
            # Store to session state
            st.session_state.G_original = G_orig
            st.session_state.G_observed = G_obs
            st.session_state.removed_edges = rem_edges
            st.session_state.occlusion_centers = centers
            st.session_state.gnn_history = history
            st.session_state.reconstructed = reconstructed
            st.session_state.place_name = place
            st.session_state.model_trained = True
            st.session_state.pipeline_run = True
            st.session_state.poi_data = poi_data
            st.session_state.last_stress_test_loss = None
            st.session_state.active_method = "GNN Link Prediction"
            st.session_state.stress_results = None
            st.session_state.ablation_results = None
        except Exception as e:
            st.error(f"Error during auto-initialization: {e}")

# If user clicks the Run button
if run_btn:
    st.session_state.pipeline_run = False  # Reset
    with st.spinner(f"ACQUIRING SATELLITE ROAD NETWORK & POI GEOMETRIES FOR '{target_address.upper()}'..."):
        try:
            G_orig = download_road_network(target_address, force_reload=True)
            st.session_state.G_original = G_orig
            poi_data = download_poi_data(target_address, force_reload=True)
            st.session_state.poi_data = poi_data
            st.session_state.last_stress_test_loss = None
        except Exception as e:
            st.error(f"Failed to fetch road network: {e}. Please check address spelling or try another location.")
            st.stop()
            
    with st.spinner("SIMULATING CLOUD OCCLUSION DEGRADATION MODEL..."):
        G_obs, rem_edges, centers = simulate_occlusion(
            st.session_state.G_original,
            num_centers=num_clouds,
            radius_meters=cloud_radius,
            occlusion_ratio=deletion_ratio,
            seed=42
        )
        st.session_state.G_observed = G_obs
        st.session_state.removed_edges = rem_edges
        st.session_state.occlusion_centers = centers

    if recon_method == "GNN Link Prediction":
        with st.spinner(f"TRAINING DEEP GNN ENCODER ({gnn_type.upper()}) & DECODER CLASSIFIER..."):
            # Training placeholder for print output
            log_placeholder = st.empty()
            
            # We will train and capture print logs
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            encoder, predictor, history = train_link_predictor(
                G_obs,
                rem_edges,
                centers,
                model_type="GAT" if gnn_type == "GAT" else "SAGE",
                epochs=training_epochs,
                lr=learning_rate,
                seed=42
            )
            
            sys.stdout = old_stdout
            st.session_state.gnn_history = history
            st.session_state.model_trained = True
            
            # Print logs to UI
            with st.expander("📚 GNN Training CLI Log"):
                st.text_area("Training Progress Output", mystdout.getvalue(), height=200)
    else:
        st.session_state.model_trained = False

    with st.spinner("COMPUTING TOPOLOGICAL EDGE STITCHING & LINK PROBABILITIES..."):
        candidates = generate_candidate_edges(G_obs, centers, max_dist=candidate_dist)
        if recon_method == "GNN Link Prediction":
            if not st.session_state.model_trained:
                st.warning("Model not trained, using heuristic.")
                reconstructed = heuristic_reconstruction(G_obs, candidates)
            else:
                reconstructed = predict_reconstruction_links(encoder, predictor, G_obs, candidates)
        elif recon_method == "MST + Disjoint Set (Baseline)":
            reconstructed = mst_disjoint_set_healing(G_obs, candidates)
        else:
            reconstructed = heuristic_reconstruction(G_obs, candidates)
            
        st.session_state.reconstructed = reconstructed
        st.session_state.place_name = target_address
        st.session_state.active_method = recon_method
        st.session_state.pipeline_run = True
        st.session_state.stress_results = None
        st.session_state.ablation_results = None
        st.success(f"Pipeline executed successfully for '{target_address}'!")

# Check that we have a valid state to display the tabs
if st.session_state.pipeline_run:
    active_method = st.session_state.get("active_method", "GNN Link Prediction")
    if active_method == "GNN Link Prediction":
        status_color = "var(--signal-cyan)"
        status_label = "GNN LINK PREDICTION"
    elif active_method == "MST + Disjoint Set (Baseline)":
        status_color = "var(--signal-amber)"
        status_label = "MST + DISJOINT SET (BASELINE)"
    else:
        status_color = "var(--text-muted)"
        status_label = "HEURISTIC STITCHING"
        
    st.markdown(f"""
    <div class="method-status-bar" style="border-left: 4px solid {status_color};">
        <div class="status-eyebrow">ACTIVE RECONSTRUCTION METHOD</div>
        <div class="status-name">{status_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # Setup Tabs
    tab_overview, tab_map, tab_criticality = st.tabs([
        "📊 GNN Model & Training Summary", 
        "🗺️ Interactive Reconstruction Map", 
        "⚠️ Criticality Analysis & Stress Testing"
    ])
    
    # ------------------ TAB 1: GNN MODEL & TRAINING SUMMARY ------------------
    with tab_overview:
        st.subheader("GNN Link Prediction Architecture & Training Performance")
        active_method = st.session_state.get("active_method", "GNN Link Prediction")
        if active_method != "GNN Link Prediction":
            st.warning(f"**{active_method}** is currently active — the training metrics below are from the last GNN run, not the current reconstruction.")
            
        if not st.session_state.get("gnn_history"):
            st.info("GNN has not been trained yet. Run the pipeline with GNN Link Prediction to see metrics.")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Model Design")
                st.markdown(f"""
                - **Encoder Architecture**: 2-layer `{gnn_type}` Conv Layer
                - **Decoder Module**: 2-layer MLP Classifier with ReLU & Sigmoid
                - **Node Features**:
                  1. **Latitude** (Min-Max Normalized)
                  2. **Longitude** (Min-Max Normalized)
                  3. **Degree** (Structural local connectivity)
                  4. **PageRank** (Global structural importance score)
                - **Positive Training Samples**: {st.session_state.G_observed.number_of_edges()} edges present in observed graph
                - **Negative Training Samples**: Bidirectional node pairs without connection (sampled 1:1 ratio)
                """)
                
                st.markdown("#### Pipeline Stage: Road Skeletonization")
                st.info("📐 **Skeletonization (ISRO Phase II):** In the full pipeline, a segmentation model output (binary road mask) is first converted to 1-pixel-wide centerlines via `skimage.morphology.skeletonize` before graph construction. This prototype uses OSM-derived graphs directly as a proxy for pre-skeletonized road centerlines.")
                
                # Fetch final validation stats
                final_epoch = st.session_state.gnn_history[-1]
                st.markdown("#### Final Metrics (On Held-out Occluded Edges)")
                st.metric("Validation ROC-AUC", f"{final_epoch['val_auc']:.4f}")
                st.metric("Validation Average Precision (AP)", f"{final_epoch['val_ap']:.4f}")
                
            with col2:
                st.markdown("#### Training History Metrics")
                # Build loss & AUC dataframe for plotting
                history_df = pd.DataFrame(st.session_state.gnn_history)
                
                # Plot Loss Curve and AUC Curve
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                fig.patch.set_facecolor('#1e293b')
                
                # Loss plot
                ax1.plot(history_df['epoch'], history_df['loss'], color='#f59e0b', linewidth=2)
                ax1.set_title("Training BCE Loss", color='#f8fafc', fontsize=10)
                ax1.set_xlabel("Epoch", color='#94a3b8')
                ax1.set_ylabel("Loss", color='#94a3b8')
                ax1.tick_params(colors='#94a3b8')
                ax1.set_facecolor('#0f172a')
                ax1.grid(color='#334155', linestyle='--', alpha=0.5)
                
                # AUC plot
                ax2.plot(history_df['epoch'], history_df['val_auc'], color='#10b981', linewidth=2, label='AUC')
                ax2.plot(history_df['epoch'], history_df['val_ap'], color='#3b82f6', linewidth=2, linestyle='--', label='AP')
                ax2.set_title("Validation Metrics (Missing Roads)", color='#f8fafc', fontsize=10)
                ax2.set_xlabel("Epoch", color='#94a3b8')
                ax2.set_ylabel("Score", color='#94a3b8')
                ax2.tick_params(colors='#94a3b8')
                ax2.set_facecolor('#0f172a')
                ax2.grid(color='#334155', linestyle='--', alpha=0.5)
                ax2.legend(facecolor='#1e293b', labelcolor='#f8fafc')
                
                st.pyplot(fig)


    # ------------------ TAB 2: INTERACTIVE RECONSTRUCTION MAP ------------------
    with tab_map:
        st.subheader("Confidence-Quantified Road Network Reconstruction Map")
        
        # Interactive slider for Confidence Threshold filter
        conf_threshold = st.slider(
            "GNN Confidence Cutoff Threshold for Reconstructed Roads", 
            min_value=0.1, max_value=1.0, value=0.6, step=0.05
        )
        st.caption("Note: Changing this threshold also affects which reconstructed roads are included in the Criticality Analysis tab, since only roads above this confidence level are added to the analyzed network.")
        
        # Filter reconstructed edges based on threshold
        reconstructed_filtered = [e for e in st.session_state.reconstructed if e["prob"] >= conf_threshold]
        
        # Metrics Row
        total_missing = len(st.session_state.removed_edges)
        recovered_missing = 0
        
        # Check how many of the removed edges are predicted above threshold
        removed_lookup = set()
        for u, v, _ in st.session_state.removed_edges:
            removed_lookup.add((min(u, v), max(u, v)))
            
        high_conf = 0
        low_conf = 0
        
        for edge in reconstructed_filtered:
            u, v = edge["u"], edge["v"]
            if edge["prob"] >= 0.8:
                high_conf += 1
            else:
                low_conf += 1
                
            if (min(u, v), max(u, v)) in removed_lookup:
                recovered_missing += 1
                
        recovery_rate = (recovered_missing / total_missing * 100) if total_missing > 0 else 100.0
        avg_confidence = np.mean([e["prob"] for e in reconstructed_filtered]) if len(reconstructed_filtered) > 0 else 0.0
        
        # Get connectivity loss from session state
        loss_text = f"{st.session_state.last_stress_test_loss:.1f}%" if st.session_state.last_stress_test_loss is not None else "Run stress test to see impact"
        
        # Render Premium Metrics Cards in a single grid layout
        avg_conf_svg = make_confidence_indicator_html(avg_confidence) if avg_confidence > 0 else "0%"
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div>
                    <div class="metric-title">Recovery Rate</div>
                    <div class="metric-value">{recovery_rate:.1f}%</div>
                </div>
                <div class="metric-info">{recovered_missing} of {total_missing} True Missing</div>
            </div>
            <div class="metric-card">
                <div>
                    <div class="metric-title">Avg Confidence</div>
                    <div class="metric-value">{avg_conf_svg}</div>
                </div>
                <div class="metric-info">Mean prediction score</div>
            </div>
            <div class="metric-card">
                <div>
                    <div class="metric-title">Confidence Spread</div>
                    <div class="metric-value" style="font-size: 1.5rem;">
                        <span style="color: var(--signal-green);">{high_conf}</span>
                        <span style="font-size: 1rem; color: var(--text-muted);">/</span>
                        <span style="color: var(--signal-red);">{low_conf}</span>
                    </div>
                </div>
                <div class="metric-info"><span style="color: var(--signal-green);">High (≥80%)</span> / <span style="color: var(--signal-red);">Low (<60%)</span></div>
            </div>
            <div class="metric-card">
                <div>
                    <div class="metric-title">Connectivity Loss</div>
                    <div class="metric-value" style="font-size: 1.6rem;">{loss_text}</div>
                </div>
                <div class="metric-info">Stressed drop in LCC</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Map Header Legend strip
        st.markdown("""
        <div class="map-legend-header">
            <span class="legend-title">🛰️ TELEMETRY LAYER VISUALIZATION</span>
            <div class="legend-items">
                <span class="legend-item"><span class="dot" style="background-color: #334155;"></span>Observed (Intact)</span>
                <span class="legend-item"><span class="dot" style="background-color: #38bdf8; opacity: 0.6;"></span>Occlusion Clouds</span>
                <span class="legend-item"><span class="dot" style="border: 1px dashed #ef4444; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>True Missing</span>
                <span class="legend-item"><span class="dot" style="background-color: var(--signal-green);"></span>High Conf GNN (≥80%)</span>
                <span class="legend-item"><span class="dot" style="background-color: var(--signal-amber);"></span>Medium Conf GNN (60-80%)</span>
                <span class="legend-item"><span class="dot" style="background-color: var(--signal-red);"></span>Low Conf GNN (<60%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Build Folium Map
        # Compute map center
        nodes_df = pd.DataFrame([
            {"lat": data["y"], "lon": data["x"]} 
            for node, data in st.session_state.G_original.nodes(data=True)
        ])
        center_lat = nodes_df["lat"].mean()
        center_lon = nodes_df["lon"].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="cartodbpositron")
        
        # Layer 1: Observed (Intact) Roads
        observed_layer = folium.FeatureGroup(name="Observed Network (Intact)", show=True)
        for u, v, data in st.session_state.G_observed.edges(data=True):
            coords = get_edge_coords(st.session_state.G_observed, u, v, data)
            folium.PolyLine(
                locations=coords,
                color="#334155",  # slate grey
                weight=3,
                opacity=0.8,
                tooltip=f"Observed Edge: {u} - {v}"
            ).add_to(observed_layer)
        observed_layer.add_to(m)
        
        # Layer 2: Occlusion Circles (Clouds)
        clouds_layer = folium.FeatureGroup(name="Simulated Occlusion Clouds", show=True)
        for center in st.session_state.occlusion_centers:
            folium.Circle(
                location=[center["y"], center["x"]],
                radius=center["radius"],
                color="#38bdf8",  # light blue
                fill=True,
                fill_color="#38bdf8",
                fill_opacity=0.15,
                weight=1,
                tooltip=f"Satellite Occlusion Center"
            ).add_to(clouds_layer)
        clouds_layer.add_to(m)

        # Layer 3: True Missing (Occluded) Edges (Ground Truth)
        missing_layer = folium.FeatureGroup(name="True Missing Edges (Occluded)", show=False)
        for u, v, data in st.session_state.removed_edges:
            coords = get_edge_coords(st.session_state.G_original, u, v, data)
            folium.PolyLine(
                locations=coords,
                color="#ef4444",  # Red
                weight=2,
                opacity=0.6,
                dash_array="5, 5",
                tooltip=f"Ground Truth Missing: {u} - {v} | Length: {data.get('length', 0):.1f}m"
            ).add_to(missing_layer)
        missing_layer.add_to(m)
        
        # Layer 4: Reconstructed Edges (Color coded by GNN confidence)
        reconstruction_layer = folium.FeatureGroup(name="GNN Reconstructed Roads", show=True)
        for edge in reconstructed_filtered:
            u, v = edge["u"], edge["v"]
            prob = edge["prob"]
            dist = edge["dist"]
            
            # Color code
            if prob >= 0.8:
                color = "#10b981"  # Emerald Green (High confidence)
                label = "High Confidence"
            elif prob >= 0.6:
                color = "#f59e0b"  # Amber Yellow (Medium confidence)
                label = "Medium Confidence"
            else:
                color = "#ef4444"  # Rose Red (Low confidence - Needs verification)
                label = "Low Confidence (Needs Verification)"
                
            # Draw line
            u_lat, u_lon = st.session_state.G_original.nodes[u]["y"], st.session_state.G_original.nodes[u]["x"]
            v_lat, v_lon = st.session_state.G_original.nodes[v]["y"], st.session_state.G_original.nodes[v]["x"]
            
            tooltip_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem;">
                <b>GNN Link Proposed:</b> {u} ↔ {v}<br>
                <b>Confidence:</b> {make_confidence_indicator_html(prob, size=24)}<br>
                <b>Distance:</b> {dist:.1f}m<br>
                <b>Status:</b> <span style="color: {color}; font-weight: 600;">{label}</span>
            </div>
            """
            
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color=color,
                weight=4,
                opacity=0.9,
                tooltip=tooltip_html
            ).add_to(reconstruction_layer)
        reconstruction_layer.add_to(m)
        
        # Add layer control to map
        folium.LayerControl().add_to(m)
        
        # Display folium map in streamlit
        st_folium(m, width=1200, height=550, returned_objects=[])
        
        # -- METHOD COMPARISON TABLE --
        st.markdown("---")
        with st.expander("📊 Three-Way Method Comparison (GNN vs MST Baseline vs Heuristic)", expanded=False):
            st.markdown("**Compare all three reconstruction methods on the same occluded graph:**")
            st.caption("GNN is trained — for MST and Heuristic, we run live on the current observed graph. Connectivity Ratio = % increase in LCC size after healing.")
            
            G_obs_cmp = st.session_state.G_observed
            rem_edges_cmp = st.session_state.removed_edges
            centers_cmp = st.session_state.occlusion_centers
            removed_lookup_cmp = set((min(u, v), max(u, v)) for u, v, _ in rem_edges_cmp)
            total_missing_cmp = len(rem_edges_cmp)
            
            candidates_cmp = generate_candidate_edges(G_obs_cmp, centers_cmp, max_dist=candidate_dist)
            
            comparison_rows = []
            methods_to_run = {
                "MST + Disjoint Set (Baseline)": lambda: mst_disjoint_set_healing(G_obs_cmp, candidates_cmp),
                "Heuristic Stitching": lambda: heuristic_reconstruction(G_obs_cmp, candidates_cmp),
            }
            if st.session_state.get("model_trained") and st.session_state.get("gnn_history"):
                methods_to_run["GNN Link Prediction"] = None  # signal to use cached results
            
            for method_name, run_fn in methods_to_run.items():
                if run_fn is None:
                    # Use current session results for GNN
                    edges_cmp = st.session_state.reconstructed
                else:
                    edges_cmp = run_fn()
                
                edges_above = [e for e in edges_cmp if e["prob"] >= conf_threshold]
                recovered = sum(1 for e in edges_above if (min(e["u"], e["v"]), max(e["u"], e["v"])) in removed_lookup_cmp)
                rec_rate = (recovered / total_missing_cmp * 100) if total_missing_cmp > 0 else 100.0
                avg_conf = np.mean([e["prob"] for e in edges_above]) if edges_above else 0.0
                
                # Build healed graph and compute connectivity ratio
                G_healed_cmp = G_obs_cmp.copy()
                for e in edges_above:
                    G_healed_cmp.add_edge(e["u"], e["v"], length=e["dist"])
                conn_ratio = compute_connectivity_ratio(G_obs_cmp, G_healed_cmp)
                
                comparison_rows.append({
                    "Method": method_name,
                    "Roads Recovered": recovered,
                    "Recovery Rate %": f"{rec_rate:.1f}%",
                    "Avg Confidence": f"{avg_conf:.2%}",
                    "Connectivity Ratio (Recovery Impact Score)": f"{conn_ratio:+.2f}%"
                })
            
            if comparison_rows:
                # Custom HTML table for method comparison
                table_html = """
                <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-family: 'Inter', sans-serif; background-color: var(--bg-panel); border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden;">
                    <thead>
                        <tr style="background-color: var(--bg-panel-hover); border-bottom: 2px solid var(--border-subtle); text-transform: uppercase; font-size: 0.8rem; color: var(--text-muted); letter-spacing: 0.05em;">
                            <th style="padding: 16px 12px; text-align: left; font-weight: 600;">Method</th>
                            <th style="padding: 16px 12px; text-align: center; font-weight: 600;">Roads Recovered</th>
                            <th style="padding: 16px 12px; text-align: center; font-weight: 600;">Recovery Rate %</th>
                            <th style="padding: 16px 12px; text-align: center; font-weight: 600;">Avg Confidence</th>
                            <th style="padding: 16px 12px; text-align: center; font-weight: 600;">Connectivity Ratio</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for row in comparison_rows:
                    table_html += f"""
                        <tr style="border-bottom: 1px solid var(--border-subtle); transition: background-color 0.2s;">
                            <td style="padding: 14px 12px; font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: var(--text-primary);">{row['Method']}</td>
                            <td style="padding: 14px 12px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700;">{row['Roads Recovered']}</td>
                            <td style="padding: 14px 12px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--signal-cyan);">{row['Recovery Rate %']}</td>
                            <td style="padding: 14px 12px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700;">{row['Avg Confidence']}</td>
                            <td style="padding: 14px 12px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--signal-green);">{row['Connectivity Ratio (Recovery Impact Score)']}</td>
                        </tr>
                    """
                table_html += "</tbody></table></div>"
                st.markdown(table_html, unsafe_allow_html=True)

    # ------------------ TAB 3: CRITICALITY ANALYSIS & STRESS TESTING ------------------
    with tab_criticality:
        st.subheader("Gatekeeper Node & Road Analysis — Stress-Testing & Resilience")
        
        # We need a reconstructed graph model to analyze.
        # This is G_observed + reconstructed_filtered edges.
        G_reconstructed = st.session_state.G_observed.copy()
        for edge in reconstructed_filtered:
            u, v = edge["u"], edge["v"]
            dist = edge["dist"]
            G_reconstructed.add_edge(u, v, length=dist, source="reconstructed", confidence=edge["prob"])
            
        # 1. Compute Criticality metrics on Reconstructed Graph
        node_cci, edge_cci, articulation_pts, crit_data = compute_criticality(
            G_reconstructed, 
            weights=cci_weights, 
            poi_data=st.session_state.poi_data
        )
        
        # Connectivity Ratio for reconstructed vs observed
        connectivity_ratio = compute_connectivity_ratio(st.session_state.G_observed, G_reconstructed)
        
        col_ctrl, col_stats = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown("#### 1. Stress-Test Parameters")
            st.markdown("""
            This test ranks **Gatekeeper Roads** and **Gatekeeper Nodes** by their **Composite Criticality Index (CCI)**:
            $$\\text{CCI}_{\\text{road}} = w_1 \\times \\text{Betweenness} + w_2 \\times \\text{Degree Sum} + w_3 \\times \\text{Hospital Access} + w_4 \\times \\text{POI Density}$$
            Removing them simulates disaster scenarios (flooding, bridge collapse, road blockage).
            """)
            
            stress_mode = st.radio(
                "🔧 Stress Test Mode",
                ["Edge Removal (Gatekeeper Roads)", "Node Ablation (Gatekeeper Node Failure)"],
                horizontal=True
            )
            
            stress_k = st.slider("Number of Gatekeeper Roads/Nodes to Destroy (K)", min_value=1, max_value=20, value=5, step=1)
            trigger_stress = st.button("💥 Simulate Targeted Destruction", use_container_width=True)
            
            # Show articulation points statistic
            st.info(f"🔍 **Gatekeeper Node Count:** We detected **{len(articulation_pts)} Gatekeeper Nodes** (articulation points) in the reconstructed graph. Disabling any of these single intersection nodes will isolate portions of the city.")
            st.metric("Connectivity Ratio (Recovery Impact Score)", f"{connectivity_ratio:+.2f}%", help="Percentage increase in Largest Connected Component size after healing (vs. pre-healing observed graph).")
            
        with col_stats:
            st.markdown("#### 2. Resilience Metrics Before vs. After Destruction")
            
            if trigger_stress:
                if stress_mode == "Edge Removal (Gatekeeper Roads)":
                    results = run_stress_test(
                        G_reconstructed, 
                        top_k=stress_k,
                        weights=cci_weights,
                        poi_data=st.session_state.poi_data,
                        edge_cci=edge_cci
                    )
                    st.session_state.last_stress_test_loss = results["changes_pct"]["lcc"]
                    st.session_state.stress_results = results
                    st.session_state.ablation_results = None
                else:
                    results = run_node_ablation_test(
                        G_reconstructed,
                        top_k=stress_k,
                        weights=cci_weights,
                        poi_data=st.session_state.poi_data,
                        node_cci=node_cci
                    )
                    st.session_state.last_stress_test_loss = results["changes_pct"]["lcc"]
                    st.session_state.ablation_results = results
                    st.session_state.stress_results = None
                st.rerun()
                
            # Display results depending on which mode was last run
            active_results = st.session_state.get("stress_results") or st.session_state.get("ablation_results")
            is_ablation = st.session_state.get("ablation_results") is not None
            
            if active_results:
                results = active_results
                
                # Render stress metrics in a single grid layout
                delta_lcc_str = f"{delta_lcc:+.2f}% drop" if delta_lcc < 0 else f"{delta_lcc:+.2f}% change"
                delta_eff_str = f"{delta_eff:+.2f}% drop" if delta_eff < 0 else f"{delta_eff:+.2f}% change"
                delta_aspl_str = f"{delta_aspl:+.2f}% change"
                
                color_lcc = "var(--signal-red)" if delta_lcc < 0 else "var(--signal-green)"
                color_eff = "var(--signal-red)" if delta_eff < 0 else "var(--signal-green)"
                color_aspl = "var(--signal-red)" if delta_aspl > 0 else "var(--signal-green)"
                
                resilience_index = results.get("resilience_index", (baseline_aspl / stressed_aspl if stressed_aspl > 0 else 0.0))
                ri_color = "var(--signal-green)" if resilience_index >= 0.8 else "var(--signal-amber)" if resilience_index >= 0.5 else "var(--signal-red)"
                
                st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div>
                            <div class="metric-title">Largest Connected Component</div>
                            <div class="metric-value">{stressed_lcc:.2%}</div>
                        </div>
                        <div>
                            <div class="metric-info">Baseline: {baseline_lcc:.2%}</div>
                            <div class="metric-info" style="color: {color_lcc}; font-weight: 500;">{delta_lcc_str}</div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>
                            <div class="metric-title">Global Efficiency</div>
                            <div class="metric-value">{stressed_eff:.4f}</div>
                        </div>
                        <div>
                            <div class="metric-info">Baseline: {baseline_eff:.4f}</div>
                            <div class="metric-info" style="color: {color_eff}; font-weight: 500;">{delta_eff_str}</div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>
                            <div class="metric-title">Avg Shortest Path (LCC)</div>
                            <div class="metric-value">{stressed_aspl:.1f}m</div>
                        </div>
                        <div>
                            <div class="metric-info">Baseline: {baseline_aspl:.1f}m</div>
                            <div class="metric-info" style="color: {color_aspl}; font-weight: 500;">{delta_aspl_str}</div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>
                            <div class="metric-title">Resilience Index</div>
                            <div class="metric-value" style="color: {ri_color};">{resilience_index:.4f}</div>
                        </div>
                        <div>
                            <div class="metric-info">baseline / stressed ASPL</div>
                            <div class="metric-info" style="font-size: 0.7rem; color: var(--text-muted);">RI &lt; 0.5 = Critical Failure</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                    
                # Detail the removed elements
                if is_ablation:
                    st.markdown("**Top-K Gatekeeper Nodes Destroyed in Simulation:**")
                    removed_list = []
                    for idx, node in enumerate(results["removed_nodes"]):
                        cci_val = node_cci.get(node, 0.0)
                        degree = G_reconstructed.degree(node)
                        removed_list.append({
                            "Rank": idx + 1,
                            "Gatekeeper Node": str(node),
                            "Node CCI": f"{cci_val:.4f}",
                            "Degree": degree
                        })
                    table_html = """
                    <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-family: 'Inter', sans-serif; background-color: var(--bg-panel); border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden;">
                        <thead>
                            <tr style="background-color: var(--bg-panel-hover); border-bottom: 2px solid var(--border-subtle); text-transform: uppercase; font-size: 0.8rem; color: var(--text-muted); letter-spacing: 0.05em;">
                                <th style="padding: 14px 12px; text-align: left; width: 80px; font-weight: 600;">Rank</th>
                                <th style="padding: 14px 12px; text-align: left; font-weight: 600;">Gatekeeper Node</th>
                                <th style="padding: 14px 12px; text-align: center; font-weight: 600;">Node CCI</th>
                                <th style="padding: 14px 12px; text-align: center; font-weight: 600;">Degree</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    for row in removed_list:
                        table_html += f"""
                            <tr style="border-bottom: 1px solid var(--border-subtle);">
                                <td style="padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: var(--signal-amber); font-weight: 700;">#{row['Rank']}</td>
                                <td style="padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--text-primary);">{row['Gatekeeper Node']}</td>
                                <td style="padding: 12px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--signal-red);">{row['Node CCI']}</td>
                                <td style="padding: 12px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--text-primary);">{row['Degree']}</td>
                            </tr>
                        """
                    table_html += "</tbody></table></div>"
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.markdown("**Top-K Gatekeeper Roads Destroyed in Simulation:**")
                    removed_edges_list = []
                    for idx, (u, v) in enumerate(results["removed_edges"]):
                        cci_val = edge_cci.get((min(u, v), max(u, v)), 0.0)
                        removed_edges_list.append({
                            "Rank": idx + 1,
                            "Gatekeeper Road Segment": f"Node {u} ↔ Node {v}",
                            "Edge CCI": f"{cci_val:.4f}"
                        })
                    table_html = """
                    <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-family: 'Inter', sans-serif; background-color: var(--bg-panel); border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden;">
                        <thead>
                            <tr style="background-color: var(--bg-panel-hover); border-bottom: 2px solid var(--border-subtle); text-transform: uppercase; font-size: 0.8rem; color: var(--text-muted); letter-spacing: 0.05em;">
                                <th style="padding: 14px 12px; text-align: left; width: 80px; font-weight: 600;">Rank</th>
                                <th style="padding: 14px 12px; text-align: left; font-weight: 600;">Gatekeeper Road Segment</th>
                                <th style="padding: 14px 12px; text-align: center; font-weight: 600;">Edge CCI</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    for row in removed_edges_list:
                        table_html += f"""
                            <tr style="border-bottom: 1px solid var(--border-subtle);">
                                <td style="padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: var(--signal-amber); font-weight: 700;">#{row['Rank']}</td>
                                <td style="padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--text-primary);">{row['Gatekeeper Road Segment']}</td>
                                <td style="padding: 12px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--signal-red);">{row['Edge CCI']}</td>
                            </tr>
                        """
                    table_html += "</tbody></table></div>"
                    st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("Click 'Simulate Targeted Destruction' to run the stress test and see before/after stats.")
                
        # 3. Criticality Heatmap Map Rendering
        st.markdown("""
        <div class="map-legend-header">
            <span class="legend-title">⚠️ CRITICALITY HEATMAP LAYER VISUALIZATION</span>
            <div class="legend-items">
                <span class="legend-item"><span class="dot" style="background-color: #334155;"></span>CCI &lt; 0.20 (Low)</span>
                <span class="legend-item"><span class="dot" style="background-color: #f59e0b;"></span>CCI 0.20-0.50 (Mid)</span>
                <span class="legend-item"><span class="dot" style="background-color: #f97316;"></span>CCI 0.50-0.80 (High)</span>
                <span class="legend-item"><span class="dot" style="background-color: #e11d48;"></span>CCI ≥ 0.80 (Critical)</span>
                <span class="legend-item"><span class="dot" style="background-color: #ec4899; box-shadow: 0 0 6px #ec4899;"></span>Gatekeeper Node</span>
                <span class="legend-item"><span class="dot" style="background-color: #f59e0b; box-shadow: 0 0 6px #f59e0b;"></span>High-CCI Node</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Build Map
        m_heat = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="cartodbpositron")
        
        # Draw edges colored by CCI (Gatekeeper Roads)
        edge_layer = folium.FeatureGroup(name="Gatekeeper Road Heatmap (CCI)", show=True)
        for u, v, data_edge in G_reconstructed.edges(data=True):
            edge_key = (min(u, v), max(u, v))
            cci = edge_cci.get(edge_key, 0.0)
            
            if cci < 0.2:
                color = "#334155"
                weight = 2
            elif cci < 0.5:
                color = "#f59e0b"
                weight = 3
            elif cci < 0.8:
                color = "#f97316"
                weight = 4.5
            else:
                color = "#e11d48"  # Gatekeeper Road
                weight = 6
                
            coords = get_edge_coords(G_reconstructed, u, v, data_edge)
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=weight,
                opacity=0.85,
                tooltip=f"Gatekeeper Road: Node {u} – Node {v}<br>Edge CCI: {cci:.4f}"
            ).add_to(edge_layer)
        edge_layer.add_to(m_heat)
        
        # Highlight Gatekeeper Nodes (articulation points) — clickable
        artic_layer = folium.FeatureGroup(name="Gatekeeper Nodes (Articulation Points)", show=True)
        # Sort by node_cci descending to rank them
        sorted_node_cci = sorted(node_cci.items(), key=lambda x: x[1], reverse=True)
        node_rank_map = {node: rank+1 for rank, (node, _) in enumerate(sorted_node_cci)}
        
        for node in articulation_pts:
            node_data = G_reconstructed.nodes[node]
            n_cci = node_cci.get(node, 0.0)
            rank = node_rank_map.get(node, "N/A")
            btw = crit_data.get("betweenness", {}).get(node, 0.0)
            folium.CircleMarker(
                location=[node_data["y"], node_data["x"]],
                radius=7,
                color="#ec4899",
                fill=True,
                fill_color="#f472b6",
                fill_opacity=0.85,
                weight=2,
                tooltip=f"🔴 Gatekeeper Node: {node}<br>CCI Rank: #{rank}<br>Node CCI: {n_cci:.4f}<br>Betweenness: {btw:.4f}<br>Degree: {G_reconstructed.degree(node)}<br>⚠️ Disabling this node disconnects the city!"
            ).add_to(artic_layer)
        artic_layer.add_to(m_heat)
        
        # High-CCI non-articulation nodes layer
        high_cci_layer = folium.FeatureGroup(name="High-CCI Nodes (Non-Articulation)", show=False)
        top_n_nodes = [node for node, _ in sorted_node_cci[:20] if node not in articulation_pts]
        for node in top_n_nodes:
            node_data = G_reconstructed.nodes[node]
            n_cci = node_cci.get(node, 0.0)
            rank = node_rank_map.get(node, "N/A")
            folium.CircleMarker(
                location=[node_data["y"], node_data["x"]],
                radius=5,
                color="#f59e0b",
                fill=True,
                fill_color="#fbbf24",
                fill_opacity=0.7,
                weight=1.5,
                tooltip=f"⭐ High-CCI Node: {node}<br>CCI Rank: #{rank}<br>Node CCI: {n_cci:.4f}<br>Degree: {G_reconstructed.degree(node)}"
            ).add_to(high_cci_layer)
        high_cci_layer.add_to(m_heat)
        
        folium.LayerControl().add_to(m_heat)
        map_click_data = st_folium(m_heat, width=1200, height=550, key="heatmap_map", returned_objects=["last_object_clicked"])
        
        # 4. Interactive Single-Node Disable Simulation
        st.markdown("#### 4. Single Gatekeeper Node: Click-to-Disable Simulation")
        
        clicked = map_click_data.get("last_object_clicked") if map_click_data else None
        
        if clicked and clicked.get("lat") is not None:
            click_lat = clicked["lat"]
            click_lng = clicked["lng"]
            
            # Find the nearest Gatekeeper Node (articulation point) to click
            best_node = None
            best_dist = float("inf")
            for node in articulation_pts:
                n_data = G_reconstructed.nodes[node]
                d = ((n_data["y"] - click_lat)**2 + (n_data["x"] - click_lng)**2)**0.5
                if d < best_dist:
                    best_dist = d
                    best_node = node
            
            if best_node is not None and best_dist < 0.005:  # ~500m in degrees
                n_cci_val = node_cci.get(best_node, 0.0)
                n_btw = crit_data.get("betweenness", {}).get(best_node, 0.0)
                n_rank = node_rank_map.get(best_node, "N/A")
                
                # Show Selected Node card with CCI progress indicator
                node_cci_indicator = make_cci_indicator_html(n_cci_val)
                st.markdown(f"""
                <div class="metric-card">
                    <div>
                        <div class="metric-title">Selected Gatekeeper Node</div>
                        <div class="metric-value">{best_node}</div>
                    </div>
                    <div style="margin-top: 10px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 2px;">CCI Score</span>
                            {node_cci_indicator}
                        </div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: var(--text-secondary);">
                            CCI Rank: <b class="telemetry-numeric">#{n_rank}</b> | 
                            Betweenness: <b class="telemetry-numeric">{n_btw:.4f}</b> | 
                            Degree: <b class="telemetry-numeric">{G_reconstructed.degree(best_node)}</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"⚡ Simulate Disabling This Gatekeeper Node ({best_node})", use_container_width=True):
                    # Run single-node ablation
                    single_node_results = run_node_ablation_test(
                        G_reconstructed, top_k=1, 
                        weights=cci_weights, 
                        poi_data=st.session_state.poi_data,
                        node_cci={best_node: n_cci_val}
                    )
                    
                    sn_baseline_lcc = single_node_results["baseline"]["lcc_fraction"]
                    sn_stressed_lcc = single_node_results["stressed"]["lcc_fraction"]
                    sn_baseline_aspl = single_node_results["baseline"]["aspl"]
                    sn_stressed_aspl = single_node_results["stressed"]["aspl"]
                    sn_ri = single_node_results["resilience_index"]
                    sn_aspl_change = single_node_results["changes_pct"]["aspl"]
                    sn_lcc_change = single_node_results["changes_pct"]["lcc"]
                    
                    sn_ri_color = "var(--signal-green)" if sn_ri >= 0.8 else "var(--signal-amber)" if sn_ri >= 0.5 else "var(--signal-red)"
                    
                    st.markdown(f"""
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div>
                                <div class="metric-title">LCC After Disabling</div>
                                <div class="metric-value">{sn_stressed_lcc:.2%}</div>
                            </div>
                            <div class="metric-info">Baseline LCC: {sn_baseline_lcc:.2%} ({sn_lcc_change:+.2f}% drop)</div>
                        </div>
                        <div class="metric-card">
                            <div>
                                <div class="metric-title">Avg Travel Time Impact</div>
                                <div class="metric-value">{sn_aspl_change:+.1f}%</div>
                            </div>
                            <div class="metric-info">ASPL: {sn_baseline_aspl:.0f}m → {sn_stressed_aspl:.0f}m</div>
                        </div>
                        <div class="metric-card">
                            <div>
                                <div class="metric-title">Resilience Index</div>
                                <div class="metric-value" style="color: {sn_ri_color};">{sn_ri:.4f}</div>
                            </div>
                            <div class="metric-info">baseline / stressed ASPL ratio</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hospital reachability impact
                    if st.session_state.poi_data and st.session_state.poi_data.get("hospitals"):
                        n_hospitals = len(st.session_state.poi_data["hospitals"])
                        st.warning(f"🏥 **Hospital Reachability Impact:** {n_hospitals} hospital(s) are tracked in this area. Disabling this Gatekeeper Node may reroute emergency vehicles by an estimated **{abs(sn_aspl_change):.1f}% longer path**.")
            else:
                st.info("Click on a **magenta Gatekeeper Node** on the map above to see its CCI details and simulate disabling it.")
        else:
            st.info("Click on a **magenta Gatekeeper Node** on the map above to simulate disabling it and see the connectivity/resilience impact.")


