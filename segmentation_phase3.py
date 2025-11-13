"""
Phase 3 utilities: alternative clustering algorithms (DBSCAN, K-Medoids),
comparative evaluation, and CSV exports reusing the existing RFM pipeline.

This module is intentionally self-contained so it can be used from both
command-line scripts and the Streamlit dashboard.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

try:
    # sklearn-extra for KMedoids
    from sklearn_extra.cluster import KMedoids
    _HAS_KMEDOIDS = True
except Exception:  # pragma: no cover - optional dep
    KMedoids = None  # type: ignore
    _HAS_KMEDOIDS = False


# ----------------------------- Data Preparation ----------------------------- #

def prepare_rfm_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]) 
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    snapshot_date = df["InvoiceDate"].max()
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum",
    }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})
    return rfm


def scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])  # robust to heavy tail
    features = ["Recency", "Frequency", "Monetary_log"]
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])
    X_df = pd.DataFrame(X, index=rfm.index, columns=[f"{c}_scaled" for c in features])
    return X_df


# ---------------------------- Clustering Routines --------------------------- #

def cluster_kmeans(X: pd.DataFrame, k: int, random_state: int = 42) -> pd.Series:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return pd.Series(labels, index=X.index, name="Cluster")


def cluster_dbscan(X: pd.DataFrame, eps: float = 0.7, min_samples: int = 10) -> pd.Series:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return pd.Series(labels, index=X.index, name="Cluster")


def cluster_kmedoids(X: pd.DataFrame, k: int, random_state: int = 42) -> pd.Series:
    if not _HAS_KMEDOIDS:
        raise ImportError("KMedoids requires sklearn-extra. Install via: pip install scikit-learn-extra")
    model = KMedoids(n_clusters=k, random_state=random_state, metric="euclidean")
    labels = model.fit_predict(X)
    return pd.Series(labels, index=X.index, name="Cluster")


# ----------------------------- Eval and Export ------------------------------ #

def safe_silhouette(X: pd.DataFrame, labels: pd.Series) -> float:
    # For DBSCAN, if only 1 cluster or all noise, silhouette is undefined
    unique = np.unique(labels)
    if len(unique) <= 1 or (len(unique) == 2 and -1 in unique and (labels != -1).sum() == 0):
        return float('nan')
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float('nan')


def profile_clusters(rfm: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    out = rfm.copy()
    out["Cluster"] = labels.values
    profile = out.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
    }).round(2)
    profile["Size"] = out["Cluster"].value_counts().sort_index()
    profile["Percentage"] = (profile["Size"] / len(out) * 100).round(1)
    return profile


def export_segmented_csv(rfm: pd.DataFrame, labels: pd.Series, path: str) -> None:
    df = rfm.copy()
    df["Cluster"] = labels.values
    df.to_csv(path, index_label="CustomerID")


def run_all_algorithms(
    csv_path: str,
    kmeans_k: int = 3,
    kmedoids_k: int = 3,
    dbscan_eps: float = 0.7,
    dbscan_min_samples: int = 10,
) -> dict:
    rfm = prepare_rfm_data(csv_path)
    X = scale_rfm(rfm)

    # KMeans
    km_labels = cluster_kmeans(X, kmeans_k)
    km_sil = safe_silhouette(X, km_labels)
    export_segmented_csv(rfm, km_labels, "segmented_customers_kmeans.csv")

    # DBSCAN
    db_labels = cluster_dbscan(X, eps=dbscan_eps, min_samples=dbscan_min_samples)
    db_sil = safe_silhouette(X, db_labels)
    export_segmented_csv(rfm, db_labels, "segmented_customers_dbscan.csv")

    # KMedoids
    if _HAS_KMEDOIDS:
        kmdo_labels = cluster_kmedoids(X, kmedoids_k)
        kmdo_sil = safe_silhouette(X, kmdo_labels)
        export_segmented_csv(rfm, kmdo_labels, "segmented_customers_kmedoids.csv")
    else:
        kmdo_labels = pd.Series(index=X.index, dtype=int)
        kmdo_sil = float('nan')

    return {
        "rfm": rfm,
        "X": X,
        "kmeans": {"labels": km_labels, "silhouette": km_sil, "profile": profile_clusters(rfm, km_labels)},
        "dbscan": {"labels": db_labels, "silhouette": db_sil, "profile": profile_clusters(rfm, db_labels)},
        "kmedoids": {"labels": kmdo_labels, "silhouette": kmdo_sil, "profile": profile_clusters(rfm, kmdo_labels) if len(kmdo_labels) else pd.DataFrame()},
        "has_kmedoids": _HAS_KMEDOIDS,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: run alternative clustering and export CSVs")
    parser.add_argument("--csv", default="online_retail.csv")
    parser.add_argument("--kmeans_k", type=int, default=3)
    parser.add_argument("--kmedoids_k", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.7)
    parser.add_argument("--min_samples", type=int, default=10)
    args = parser.parse_args()

    results = run_all_algorithms(
        csv_path=args.csv,
        kmeans_k=args.kmeans_k,
        kmedoids_k=args.kmedoids_k,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
    )

    # Quick report
    print("Silhouette Scores (higher is better; NaN may indicate noise-only clusters):")
    for name in ["kmeans", "dbscan", "kmedoids"]:
        val = results[name]["silhouette"] if name in results else float('nan')
        print(f"- {name}: {val}")
















