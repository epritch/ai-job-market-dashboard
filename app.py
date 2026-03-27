import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.express as px

st.set_page_config(page_title="AI Job Market Mining Dashboard", layout="wide")

st.title("AI Impact on Job Market (2024–2030) — Data Mining Dashboard")


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]

    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # -----------------------------
    # Convert selected categorical columns to numeric
    # -----------------------------
    if "AI_Adoption_Level" in df.columns:
        df["AI_Adoption_Level"] = df["AI_Adoption_Level"].map({
            "Low": 1,
            "Medium": 2,
            "High": 3
        })

    if "Automation_Risk" in df.columns:
        df["Automation_Risk"] = df["Automation_Risk"].map({
            "Low": 1,
            "Medium": 2,
            "High": 3
        })

    if "Job_Growth_Projection" in df.columns:
        df["Job_Growth_Projection"] = df["Job_Growth_Projection"].map({
            "Decline": -1,
            "Stable": 0,
            "Growth": 1
        })

    if "Remote_Friendly" in df.columns:
        df["Remote_Friendly"] = df["Remote_Friendly"].map({
            "Yes": 1,
            "No": 0
        })

    return df


def infer_column_groups(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols


# -----------------------------
# Sidebar: load + settings
# -----------------------------
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

# -----------------------------
# Feature selection
# -----------------------------
st.sidebar.subheader("Feature Selection")

default_numeric = [c for c in ["Salary_USD", "Automation_Risk", "AI_Adoption_Level"] if c in numeric_cols]
if not default_numeric:
    default_numeric = numeric_cols[: min(4, len(numeric_cols))]

default_cat = [c for c in ["Job_Title", "Industry", "Company_Size"] if c in cat_cols]
if not default_cat:
    default_cat = cat_cols[: min(3, len(cat_cols))]

selected_numeric = st.sidebar.multiselect(
    "Numeric features",
    options=numeric_cols,
    default=default_numeric
)

selected_cat = st.sidebar.multiselect(
    "Categorical features",
    options=cat_cols,
    default=default_cat
)

if len(selected_numeric) == 0 and len(selected_cat) == 0:
    st.warning("Select at least one feature (numeric or categorical) to run mining.")
    st.stop()

# -----------------------------
# Mining settings
# -----------------------------
st.sidebar.subheader("Clustering")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)

st.sidebar.subheader("Dimensionality Reduction")
max_pca = 5
pca_components = st.sidebar.slider("PCA components", min_value=2, max_value=max_pca, value=2)

# -----------------------------
# Build preprocessing + model pipeline
# -----------------------------
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

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("pca", PCA(n_components=pca_components, random_state=42)),
    ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42))
])

# -----------------------------
# Run model
# -----------------------------
with st.spinner("Running preprocessing + PCA + clustering..."):
    model.fit(df)

    prep_data = model.named_steps["prep"].transform(df)
    cluster_labels = model.named_steps["kmeans"].labels_
    pca_coords = model.named_steps["pca"].transform(prep_data)

    explained_var = model.named_steps["pca"].explained_variance_ratio_
    total_var = explained_var.sum()

    sil_score = None
    if len(df) > k:
        try:
            sil_score = silhouette_score(prep_data, cluster_labels)
        except Exception:
            sil_score = None

df_result = df.copy()
df_result["cluster"] = cluster_labels

plot_df = pd.DataFrame(pca_coords[:, :2], columns=["PC1", "PC2"])
plot_df["cluster"] = cluster_labels

# -----------------------------
# Key metrics
# -----------------------------
st.subheader("Model Summary")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Clusters (k)", k)

with m2:
    st.metric("PCA Variance Explained", f"{total_var:.2%}")

with m3:
    if sil_score is not None:
        st.metric("Silhouette Score", f"{sil_score:.3f}")
    else:
        st.metric("Silhouette Score", "N/A")

st.caption(
    "Higher PCA variance explained means the reduced dimensions preserve more of the original structure. "
    "A higher silhouette score generally indicates more separated and cohesive clusters."
)

# -----------------------------
# Silhouette comparison across k
# -----------------------------
st.subheader("Silhouette Score Comparison Across k Values")

k_values = list(range(2, 9))
scores = []

for k_test in k_values:
    try:
        kmeans_test = KMeans(n_clusters=k_test, n_init=10, random_state=42)
        labels_test = kmeans_test.fit_predict(prep_data)
        score = silhouette_score(prep_data, labels_test)
    except Exception:
        score = None
    scores.append(score)

sil_df = pd.DataFrame({
    "k": k_values,
    "silhouette_score": scores
})

sil_col1, sil_col2 = st.columns([1, 1.2])

with sil_col1:
    st.dataframe(sil_df, use_container_width=True)

with sil_col2:
    sil_fig = px.line(
        sil_df,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Silhouette Score vs Number of Clusters"
    )
    st.plotly_chart(sil_fig, use_container_width=True)

# -----------------------------
# Layout: charts + insights
# -----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Cluster Map (PCA 2D)")
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color=plot_df["cluster"].astype(str),
        title="Jobs grouped by selected features",
        labels={"color": "Cluster"},
        hover_data={"cluster": True}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Cluster Counts")
    cluster_counts = (
        df_result["cluster"]
        .value_counts()
        .sort_index()
        .rename("count")
        .to_frame()
    )
    st.dataframe(cluster_counts, use_container_width=True)

    st.subheader("Preview with Cluster Labels")
    st.dataframe(df_result.head(15), use_container_width=True)

# -----------------------------
# Cluster summary cards
# -----------------------------
st.subheader("Cluster Overview Cards")

if len(selected_numeric) > 0:
    for cluster_id in sorted(df_result["cluster"].unique()):
        cluster_data = df_result[df_result["cluster"] == cluster_id]

        st.markdown(f"### Cluster {cluster_id}")
        c1, c2, c3 = st.columns(3)

        c1.metric("Count", len(cluster_data))

        avg_feature_1 = cluster_data[selected_numeric[0]].mean()
        c2.metric(f"Avg {selected_numeric[0]}", f"{avg_feature_1:.2f}")

        if len(selected_numeric) > 1:
            avg_feature_2 = cluster_data[selected_numeric[1]].mean()
            c3.metric(f"Avg {selected_numeric[1]}", f"{avg_feature_2:.2f}")
        else:
            c3.metric("Features Used", len(selected_numeric))
else:
    st.info("Select at least one numeric feature to generate cluster overview cards.")

# -----------------------------
# Cluster summary statistics
# -----------------------------
st.subheader("Cluster Summary Statistics")

summary_numeric = selected_numeric.copy()

if len(summary_numeric) > 0:
    cluster_summary = df_result.groupby("cluster")[summary_numeric].mean().round(2)
    cluster_summary["count"] = df_result["cluster"].value_counts().sort_index()
    st.dataframe(cluster_summary, use_container_width=True)
else:
    st.info("Select at least one numeric feature to generate cluster summary statistics.")

# -----------------------------
# PCA loadings / feature influence
# -----------------------------
st.subheader("PCA Feature Influence")

try:
    pca_model = model.named_steps["pca"]
    feature_names = model.named_steps["prep"].get_feature_names_out()

    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f"PC{i+1}" for i in range(pca_components)],
        index=feature_names
    )

    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(10)
    top_pc1_df = top_pc1.reset_index()
    top_pc1_df.columns = ["feature", "importance"]

    load_fig = px.bar(
        top_pc1_df,
        x="feature",
        y="importance",
        title="Top Features Influencing PC1"
    )
    st.plotly_chart(load_fig, use_container_width=True)

    with st.expander("View PCA Loadings Table"):
        st.dataframe(loadings.round(3), use_container_width=True)

except Exception as e:
    st.info("PCA loadings could not be displayed for the current feature configuration.")
    st.caption(f"Details: {e}")

# -----------------------------
# Interpretation text
# -----------------------------
st.markdown("---")
st.subheader("How to Interpret the Clusters")

st.markdown(
    """
**What the clusters represent:**  
Each cluster groups job roles that share similar characteristics based on the selected
features, such as salary, automation risk, industry, and required skills. Jobs within the same
cluster are more similar to each other than to jobs in other clusters.

**How PCA is used:**  
Principal Component Analysis (PCA) reduces the complexity of the dataset while preserving
important variation. The two plotted axes (PC1 and PC2) represent combinations of the
original features that capture major patterns in the data.

**Why this matters:**  
This allows complex job-market relationships to be visualized in a simpler two-dimensional
space, making it easier to see how different types of roles group together.

**Important note:**  
Clusters do not represent definitive predictions. They highlight patterns and relationships in
the data that can help users compare how different job roles may be affected by AI and automation.
"""
)

# -----------------------------
# Compare clusters on one feature
# -----------------------------
st.subheader("Compare Clusters")

if len(selected_numeric) > 0:
    compare_feature = st.selectbox(
        "Choose a numeric feature to compare",
        options=selected_numeric
    )

    box_fig = px.box(
        df_result,
        x=df_result["cluster"].astype(str),
        y=compare_feature,
        points="all",
        title=f"{compare_feature} by cluster",
        labels={"x": "Cluster", "y": compare_feature}
    )
    st.plotly_chart(box_fig, use_container_width=True)
else:
    st.info("Select at least one numeric feature to enable cluster comparison plots.")

# -----------------------------
# Four-cluster narrative
# -----------------------------
st.subheader("Four-Cluster Narrative")

st.markdown(
    """
- **Cluster 0:** May represent higher-salary and more technical roles that still show meaningful automation exposure.  
- **Cluster 1:** May include lower-risk roles spread across industries, suggesting greater resilience to AI-driven change.  
- **Cluster 2:** May capture mid-range roles with mixed automation risk, reflecting transitional job categories.  
- **Cluster 3:** May reflect specialized or skill-intensive roles with lower automation risk, suggesting protection through complexity, creativity, or task diversity.  

**Overall takeaway:**  
The cluster structure suggests that AI’s impact on the job market is not driven by salary alone. Instead, job vulnerability appears to depend on a combination of skills, task structure, and industry context.
"""
)

# -----------------------------
# Early analytical insights
# -----------------------------
st.subheader("Early Cluster Insights")

st.markdown(
    """
- Some clusters may group higher-paying technical roles with elevated automation risk,
  suggesting that salary alone does not determine resilience to AI.
- Lower-risk clusters may reflect roles with more varied, human-centered, or less automatable responsibilities.
- Changes in selected features can alter the cluster structure, showing that salary, industry,
  and automation risk interact in meaningful ways.
- These observations are exploratory patterns, not definitive conclusions, and should be interpreted carefully.
"""
)
