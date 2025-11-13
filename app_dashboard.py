"""
Streamlit dashboard for Phase 3 customer segmentation.

Usage:
  streamlit run app_dashboard.py
  
  Or use the automated scripts:
  ./start_dashboard.sh  (to run in background)
  ./stop_dashboard.sh   (to stop)
"""

import warnings
warnings.filterwarnings("ignore")

import sys

# Try to import streamlit - if it fails, show helpful error
try:
    import streamlit as st
except ImportError:
    print("=" * 60)
    print("ERROR: Streamlit is not installed or not in the current Python environment!")
    print("=" * 60)
    print("\nThis is a Streamlit app and must be run with 'streamlit run', not 'python'.")
    print("\nTo fix this, use one of these methods:")
    print("\n1. Using the automated script (recommended):")
    print("   ./setup.sh          # Install dependencies first")
    print("   ./start_dashboard.sh  # Then start the dashboard")
    print("\n2. Using Streamlit directly:")
    print("   source .venv/bin/activate")
    print("   streamlit run app_dashboard.py")
    print("\n3. Using Python module:")
    print("   source .venv/bin/activate")
    print("   python -m streamlit run app_dashboard.py")
    print("\n" + "=" * 60)
    sys.exit(1)

import numpy as np
import pandas as pd
import plotly.express as px

from segmentation_phase3 import (
    prepare_rfm_data,
    scale_rfm,
    cluster_kmeans,
    cluster_dbscan,
    cluster_kmedoids,
    profile_clusters,
    safe_silhouette,
)


st.set_page_config(page_title="AI Customer Segmentation", layout="wide")
st.title("AI-Based Customer Segmentation Dashboard")
st.caption("Explore RFM-based clusters with K-Means, DBSCAN, and K-Medoids")


@st.cache_data(show_spinner=False)
def load_and_scale(csv_path: str):
    rfm = prepare_rfm_data(csv_path)
    X = scale_rfm(rfm)
    out = rfm.join(X)
    return rfm, X, out


with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("CSV path", value="online_retail.csv")
    algo = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "K-Medoids"])

    if algo in {"K-Means", "K-Medoids"}:
        k = st.slider("Number of clusters (k)", 2, 10, 3)
    if algo == "DBSCAN":
        eps = st.slider("eps", 0.1, 5.0, 0.7, 0.1)
        min_samples = st.slider("min_samples", 3, 50, 10)


rfm, X, merged = load_and_scale(data_path)

# Run chosen algorithm
if algo == "K-Means":
    labels = cluster_kmeans(X, k)
elif algo == "DBSCAN":
    labels = cluster_dbscan(X, eps=eps, min_samples=min_samples)
else:  # K-Medoids
    try:
        labels = cluster_kmedoids(X, k)
    except Exception as e:
        st.error(str(e))
        st.stop()

merged["Cluster"] = labels
sil = safe_silhouette(X, labels)


col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("3D Cluster Scatter (scaled RFM)")
    fig3d = px.scatter_3d(
        merged.reset_index(),
        x="Recency_scaled",
        y="Frequency_scaled",
        z="Monetary_log_scaled",
        color="Cluster",
        hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
        opacity=0.8,
    )
    st.plotly_chart(fig3d, use_container_width=True)

with col2:
    st.subheader("2D Pairwise Scatter")
    fig2d = px.scatter(
        merged.reset_index(),
        x="Recency_scaled",
        y="Monetary_log_scaled",
        color="Cluster",
        hover_data=["CustomerID", "Frequency"],
        opacity=0.8,
    )
    st.plotly_chart(fig2d, use_container_width=True)


st.markdown("---")
st.subheader("Cluster Summary")
summary = profile_clusters(rfm, labels)
st.dataframe(summary)

st.metric("Silhouette Score", "NA" if np.isnan(sil) else f"{sil:.3f}")


with st.expander("Raw segmented data"):
    st.dataframe(merged.reset_index())


