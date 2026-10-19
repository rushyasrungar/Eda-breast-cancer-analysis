"""
Exploratory Data Analysis - Breast Cancer Wisconsin (Diagnostic) Dataset
Run from the repository root:  python src/eda.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG, REP = ROOT / "data", ROOT / "figures", ROOT / "reports"
for p in (DATA, FIG, REP):
    p.mkdir(exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------ 1. Load
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# In sklearn: 0 = malignant, 1 = benign. Make it readable.
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
df = df.drop(columns="target")
df.columns = [c.replace(" ", "_") for c in df.columns]
df.to_csv(DATA / "breast_cancer.csv", index=False)
features = [c for c in df.columns if c != "diagnosis"]
print(f"Shape: {df.shape}")

# ------------------------------------------------------- 2. Data quality
quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "unique": df.nunique(),
})
print("Missing values total:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
quality.to_csv(REP / "data_quality.csv")

# ------------------------------------------------- 3. Statistical summary
summary = df[features].describe().T
summary["skew"] = df[features].skew()
summary["kurtosis"] = df[features].kurt()
summary.round(3).to_csv(REP / "summary_statistics.csv")
counts = df["diagnosis"].value_counts()
print(counts, (counts / len(df) * 100).round(1), sep="\n")

# Fig 1: class balance
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x="diagnosis", order=["Benign", "Malignant"], ax=ax)
for c in ax.containers:
    ax.bar_label(c)
ax.set_title("Class Distribution")
save("01_class_distribution.png")

# Fig 2: histograms of the 10 "mean" features by diagnosis
mean_cols = [c for c in features if c.startswith("mean_")]
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
for ax, col in zip(axes.ravel(), mean_cols):
    sns.histplot(data=df, x=col, hue="diagnosis", kde=True, bins=25, ax=ax, stat="density", common_norm=False)
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("")
save("02_distributions_mean_features.png")

# Fig 3: boxplots for outlier detection
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
for ax, col in zip(axes.ravel(), mean_cols):
    sns.boxplot(data=df, x="diagnosis", y=col, order=["Benign", "Malignant"], ax=ax)
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("")
save("03_boxplots_mean_features.png")

# ------------------------------------------------------------ 4. Outliers
q1, q3 = df[features].quantile(0.25), df[features].quantile(0.75)
iqr = q3 - q1
outlier_mask = (df[features] < q1 - 1.5 * iqr) | (df[features] > q3 + 1.5 * iqr)
outlier_counts = outlier_mask.sum().sort_values(ascending=False)
outlier_counts.to_csv(REP / "outlier_counts_iqr.csv", header=["outliers"])
print("Rows with >=1 IQR outlier:", int(outlier_mask.any(axis=1).sum()))
print(outlier_counts.head(5))

# --------------------------------------------------------- 5. Correlations
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Correlation Matrix - All 30 Features")
save("04_correlation_heatmap.png")

# Strongest correlated feature pairs (multicollinearity)
pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
             .stack().rename("corr").reset_index())
pairs.columns = ["feature_1", "feature_2", "corr"]
pairs["abs"] = pairs["corr"].abs()
pairs = pairs.sort_values("abs", ascending=False)
pairs.head(20).drop(columns="abs").round(3).to_csv(REP / "top_correlated_pairs.csv", index=False)
print("Pairs with |r|>0.9:", int((pairs["abs"] > 0.9).sum()))
print(pairs.head(5))

# Correlation of each feature with the diagnosis (malignant = 1)
y = (df["diagnosis"] == "Malignant").astype(int)
target_corr = df[features].corrwith(y).sort_values(key=np.abs, ascending=False)
target_corr.round(3).to_csv(REP / "correlation_with_malignancy.csv", header=["pearson_r"])
print(target_corr.head(8))

fig, ax = plt.subplots(figsize=(8, 10))
target_corr.sort_values().plot.barh(ax=ax, color=["#d1495b" if v > 0 else "#2e86ab" for v in target_corr.sort_values()])
ax.set_title("Correlation of Each Feature with Malignant Diagnosis")
ax.set_xlabel("Pearson r")
save("05_feature_target_correlation.png")

# Pairplot of top 4 predictors
top4 = list(target_corr.head(4).index)
g = sns.pairplot(df[top4 + ["diagnosis"]], hue="diagnosis", corner=True, plot_kws={"alpha": 0.6, "s": 25})
g.fig.suptitle("Top 4 Features Most Correlated with Diagnosis", y=1.02)
g.savefig(FIG / "06_pairplot_top_features.png", dpi=130, bbox_inches="tight")
plt.close("all")

# ------------------------------------------------ 6. Hypothesis testing
rows = []
for col in features:
    a = df.loc[df.diagnosis == "Malignant", col]
    b = df.loc[df.diagnosis == "Benign", col]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b)
    pooled = np.sqrt((a.var() + b.var()) / 2)
    d = (a.mean() - b.mean()) / pooled
    rows.append((col, a.mean(), b.mean(), t, p, pu, d))
tests = pd.DataFrame(rows, columns=["feature", "mean_malignant", "mean_benign",
                                    "welch_t", "p_value", "mannwhitney_p", "cohens_d"])
tests["abs_d"] = tests["cohens_d"].abs()
tests = tests.sort_values("abs_d", ascending=False).drop(columns="abs_d")
tests.round(4).to_csv(REP / "group_comparison_tests.csv", index=False)
print("Features significant at p<0.05:", int((tests["p_value"] < 0.05).sum()), "/ 30")
print(tests.head(6).round(3).to_string(index=False))

# ---------------------------------------- 7. Key influencing factors (RF)
Xtr, Xte, ytr, yte = train_test_split(df[features], y, test_size=0.25, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
acc = rf.score(Xte, yte)
print(f"Random forest hold-out accuracy: {acc:.3f}")
perm = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=42)
imp = pd.DataFrame({
    "feature": features,
    "rf_importance": rf.feature_importances_,
    "permutation_importance": perm.importances_mean,
}).sort_values("rf_importance", ascending=False)
imp.round(4).to_csv(REP / "feature_importance.csv", index=False)
print(imp.head(8).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 7))
top = imp.head(15).iloc[::-1]
ax.barh(top["feature"], top["rf_importance"], color="#2e86ab")
ax.set_title(f"Top 15 Feature Importances (Random Forest, hold-out acc = {acc:.1%})")
save("07_feature_importance.png")

# --------------------------------------------------------------- 8. PCA
Xs = StandardScaler().fit_transform(df[features])
pca = PCA().fit(Xs)
evr = pca.explained_variance_ratio_
print("PC1, PC2 explained variance:", evr[:2].round(3), "| PCs for 95%:", int(np.argmax(np.cumsum(evr) >= 0.95) + 1))
comp = pca.transform(Xs)[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(x=comp[:, 0], y=comp[:, 1], hue=df["diagnosis"], alpha=0.7, ax=axes[0])
axes[0].set_xlabel(f"PC1 ({evr[0]:.1%})"); axes[0].set_ylabel(f"PC2 ({evr[1]:.1%})")
axes[0].set_title("PCA Projection")
axes[1].plot(range(1, 31), np.cumsum(evr), marker="o")
axes[1].axhline(0.95, ls="--", color="grey")
axes[1].set_xlabel("Number of components"); axes[1].set_ylabel("Cumulative explained variance")
axes[1].set_title("Explained Variance")
save("08_pca.png")

print("DONE")
