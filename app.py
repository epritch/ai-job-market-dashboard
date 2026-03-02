import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px

st.set_page_config(page_title="AI Job Market Mining Dashboard", layout="wide")

st.title("AI Impact on Job Market (2024–2030) — Data Mining Dashboard")


# Helpers
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace in string columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    return df

def infer_column_groups(df: pd.DataFrame):
    # Simple heuristics for selecting columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols

# Sidebar: load + settings
st.sidebar.header("Data & Settings")

default_path = "ai_job_market_insights.csv"
csv_path = st.sidebar.text_input("CSV file path", value=default_path)

try:
    df_raw = load_data(csv_path)
except Exception as e:
    st.error("Could not load CSV. Check the file path/name.")
    st.caption(f"Error: {e}")
    st.stop()

df = basic_clean(df_raw.copy())

st.sidebar.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

numeric_cols, cat_cols = infer_column_groups(df)

# Let user select columns for mining
st.sidebar.subheader("Feature Selection")

selected_numeric = st.sidebar.multiselect(
    "Numeric features",
    options=numeric_cols,
    default=numeric_cols[: min(4, len(numeric_cols))]
)

selected_cat = st.sidebar.multiselect(
    "Categorical features",
    options=cat_cols,
    default=cat_cols[: min(3, len(cat_cols))]
)

if len(selected_numeric) == 0 and len(selected_cat) == 0:
    st.warning("Select at least one feature (numeric or categorical) to run mining.")
    st.stop()

# Mining settings
st.sidebar.subheader("Clustering")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)

st.sidebar.subheader("Dimensionality Reduction")
pca_components = st.sidebar.slider("PCA components", min_value=2, max_value=5, value=2)

# Build preprocessing + model pipeline
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selected_numeric),
        ("cat", categorical_transformer, selected_cat),
    ],
    remainder="drop"
)

# PCA + KMeans
model = Pipeline(steps=[
    ("prep", preprocessor),
    ("pca", PCA(n_components=pca_components, random_state=42)),
    ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42))
])

# Run model
with st.spinner("Running preprocessing + PCA + clustering..."):
    model.fit(df)
    cluster_labels = model.named_steps["kmeans"].labels_
    pca_coords = model.named_steps["pca"].transform(model.named_steps["prep"].transform(df))

df_result = df.copy()
df_result["cluster"] = cluster_labels

# For plotting: use first 2 PCA components
plot_df = pd.DataFrame(pca_coords[:, :2], columns=["PC1", "PC2"])
plot_df["cluster"] = cluster_labels

# Layout: charts + insights
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Cluster Map (PCA 2D)")
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="Jobs grouped by selected features",
        hover_data={"cluster": True}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Cluster Summary")
    st.write("Basic cluster sizes:")
    st.dataframe(
        df_result["cluster"].value_counts().sort_index().rename("count").to_frame(),
        use_container_width=True
    )

    st.write("Preview rows with cluster labels:")
    st.dataframe(df_result.head(15), use_container_width=True)

st.markdown("---")
st.subheader("How to Interpret the Clusters")

st.markdown(
    """
**What the clusters represent:**  
Each cluster groups job roles that share similar characteristics based on the selected
features (such as salary, automation risk, and required skills). Jobs within the same
cluster are more similar to each other than to jobs in other clusters.

**How PCA is used:**  
Principal Component Analysis (PCA) is applied to reduce the complexity of the data while
preserving the most important patterns. The two axes shown in the cluster map (PC1 and
PC2) represent combinations of the original features that explain the most variation.

**Important note:**  
Clusters do not represent definitive outcomes or predictions. Instead, they highlight
patterns and relationships in the data that help compare how different job roles may be
affected by AI and automation.
"""
)

# Compare clusters on a numeric feature
st.subheader("Compare Clusters")
if len(selected_numeric) > 0:
    compare_feature = st.selectbox("Choose a numeric feature to compare", options=selected_numeric)
    box_fig = px.box(df_result, x="cluster", y=compare_feature, points="all", title=f"{compare_feature} by cluster")
    st.plotly_chart(box_fig, use_container_width=True)
else:
    st.info("Select at least one numeric feature to enable cluster comparison plots.")
