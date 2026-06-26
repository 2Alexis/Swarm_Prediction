"""
explore_fish.py — Exploration data + graphiques FAO Fish Production 2024.

Sections :
  1. Couverture (pays × années × espèces)
  2. Distributions (production totale, par source)
  3. Évolution temporelle mondiale + top 10 pays
  4. Heatmap pays × année (matrice large)
  5. Capture vs Aquaculture (mondial + top 15 pays)
  6. Diversité espèces par pays (ISSCAAP)
  7. Corrélation fish production vs features dataset principal
  8. Top espèces mondiales
  9. Variation par cluster climatique
"""
import os, sys, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

OUT_DIR  = "data/cleaned"
FIG_DIR  = "couche1_planete/reports/fish_figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ── 1. Chargement ─────────────────────────────────────────────────────────
print("[1] Chargement…")
df_total = pd.read_csv(f"{OUT_DIR}/fish_production_total.csv")
df_src   = pd.read_csv(f"{OUT_DIR}/fish_production_by_source.csv")
df_iss   = pd.read_csv(f"{OUT_DIR}/fish_production_by_isscaap.csv")
df_raw   = pd.read_csv(f"{OUT_DIR}/fish_production_raw_clean.csv", low_memory=False)
print(f"   total : {df_total.shape} ; by_source : {df_src.shape} ; by_isscaap : {df_iss.shape}")
print(f"   raw clean : {df_raw.shape}")


# ── 2. Couverture ─────────────────────────────────────────────────────────
print("\n[2] Couverture du dataset…")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
years_per_iso = df_total.groupby("ISO")["Annee"].nunique().sort_values(ascending=False)
ax.hist(years_per_iso, bins=30, color="steelblue", edgecolor="white")
ax.axvline(years_per_iso.median(), color="red", ls="--", label=f"med={years_per_iso.median():.0f}")
ax.set_title("Nombre d'années couvertes par pays", weight="bold")
ax.set_xlabel("Années"); ax.legend()

ax = axes[1]
isos_per_year = df_total.groupby("Annee")["ISO"].nunique()
ax.plot(isos_per_year.index, isos_per_year.values, lw=2, color="forestgreen")
ax.fill_between(isos_per_year.index, isos_per_year.values, alpha=0.3, color="forestgreen")
ax.set_title("Nombre de pays reportant par année", weight="bold")
ax.set_xlabel("Année"); ax.set_ylabel("Nb pays")

ax = axes[2]
sp_per_iso = df_total.groupby("ISO")["fish_species_count"].mean().sort_values(ascending=False)
ax.hist(sp_per_iso, bins=30, color="coral", edgecolor="white")
ax.axvline(sp_per_iso.median(), color="blue", ls="--", label=f"med={sp_per_iso.median():.0f}")
ax.set_title("Diversité moyenne d'espèces par pays", weight="bold")
ax.set_xlabel("Nb espèces"); ax.legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_coverage.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/01_coverage.png")


# ── 3. Distribution production totale ──────────────────────────────────────
print("\n[3] Distributions production…")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
recent = df_total[df_total["Annee"] >= 2010]
ax.hist(np.log10(recent["fish_total_t"].clip(lower=1)), bins=40,
        color="steelblue", edgecolor="white")
ax.set_xlabel("log10(tonnes totales par pays/année, 2010-2022)")
ax.set_ylabel("Fréquence")
ax.set_title("Distribution production (log)", weight="bold")

ax = axes[1]
# Par source 2022
src_cols = [c for c in df_src.columns if c.startswith("fish_")]
src_2022 = df_src[df_src["Annee"]==2022][src_cols].sum()
ax.pie(src_2022.values, labels=[c.replace("fish_","").replace("_t","") for c in src_2022.index],
       autopct="%1.1f%%", colors=plt.cm.Set3.colors)
ax.set_title("Production mondiale 2022 — répartition par source", weight="bold")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_distributions.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/02_distributions.png")


# ── 4. Évolution temporelle mondiale ──────────────────────────────────────
print("\n[4] Évolution temporelle mondiale…")
world = df_src.groupby("Annee")[src_cols].sum() / 1e6  # M tonnes

fig, ax = plt.subplots(figsize=(13, 6))
world.plot(kind="area", stacked=True, ax=ax, alpha=0.85,
            color=["#1f77b4","#ff7f0e","#2ca02c","#d62728"])
ax.set_title("Production halieutique mondiale 1950-2022 (M tonnes)", weight="bold", fontsize=14)
ax.set_xlabel("Année"); ax.set_ylabel("Millions de tonnes")
ax.legend([c.replace("fish_","").replace("_t","") for c in src_cols], loc="upper left")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_evolution_mondiale.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/03_evolution_mondiale.png")


# ── 5. Top 10 pays — évolution ────────────────────────────────────────────
print("\n[5] Top 10 pays…")
top10 = df_total[df_total["Annee"]==2022].nlargest(10, "fish_total_t")["ISO"].tolist()

fig, ax = plt.subplots(figsize=(13, 7))
for iso in top10:
    sub = df_total[df_total["ISO"]==iso].sort_values("Annee")
    ax.plot(sub["Annee"], sub["fish_total_t"]/1e6, lw=2, marker="o", ms=3, label=iso, alpha=0.9)
ax.set_title("Top 10 pays producteurs (2022) — évolution 1950-2022", weight="bold", fontsize=13)
ax.set_xlabel("Année"); ax.set_ylabel("Production (M tonnes)")
ax.legend(ncol=5, loc="upper left", fontsize=9)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_top10_pays.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/04_top10_pays.png")


# ── 6. Capture vs Aquaculture (top 15 pays) ────────────────────────────────
print("\n[6] Capture vs aquaculture top 15…")
top15 = df_total[df_total["Annee"]==2022].nlargest(15, "fish_total_t")["ISO"].tolist()
df15 = df_src[(df_src["ISO"].isin(top15)) & (df_src["Annee"]==2022)].set_index("ISO").loc[top15]

# Préparer colonnes
capt = df15.get("fish_Capture_t", 0)
aqua = (df15.get("fish_Aquaculture_Freshwater_t", 0) +
        df15.get("fish_Aquaculture_Marine_t", 0) +
        df15.get("fish_Aquaculture_Brackish_t", 0))

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(top15))
ax.bar(x - 0.2, capt/1e6, width=0.4, label="Capture", color="steelblue", alpha=0.85)
ax.bar(x + 0.2, aqua/1e6, width=0.4, label="Aquaculture", color="forestgreen", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(top15, rotation=0)
ax.set_ylabel("Production 2022 (M tonnes)")
ax.set_title("Capture vs Aquaculture — Top 15 pays (2022)", weight="bold", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_capture_vs_aquaculture.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/05_capture_vs_aquaculture.png")


# ── 7. Heatmap groupes ISSCAAP × pays top 20 ─────────────────────────────
print("\n[7] Heatmap ISSCAAP × top 20 pays…")
top20 = df_total[df_total["Annee"]==2022].nlargest(20, "fish_total_t")["ISO"].tolist()
iss_2022 = df_iss[(df_iss["ISO"].isin(top20)) & (df_iss["Annee"]==2022)]
iss_2022 = iss_2022.set_index("ISO").loc[top20]
iss_cols = [c for c in iss_2022.columns if c.startswith("fish_isscaap_")]
mat = iss_2022[iss_cols] / 1e3  # k tonnes
mat.columns = [c.replace("fish_isscaap_","")[:25] for c in mat.columns]
# Log scale pour lisibilité
log_mat = np.log10(mat.clip(lower=1))

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(log_mat, cmap="YlOrRd", annot=False, cbar_kws={"label": "log10(k tonnes)"}, ax=ax)
ax.set_title("Production par groupe ISSCAAP × Top 20 pays (2022, log)", weight="bold", fontsize=13)
ax.set_xlabel(""); ax.set_ylabel("Pays")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_heatmap_isscaap.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/06_heatmap_isscaap.png")


# ── 8. Top 15 espèces mondiales (2018-2022 moyenne) ───────────────────────
print("\n[8] Top espèces mondiales…")
recent_raw = df_raw[(df_raw["Annee"] >= 2018) & (df_raw["Tonnes"] > 0)]
top_sp = (recent_raw.groupby("Espece_En")["Tonnes"].sum()
           .sort_values(ascending=False).head(20) / 1e6)

fig, ax = plt.subplots(figsize=(12, 8))
top_sp.sort_values().plot(kind="barh", color="navy", alpha=0.85, ax=ax)
ax.set_title("Top 20 espèces mondiales 2018-2022 cumulé (M tonnes)", weight="bold", fontsize=13)
ax.set_xlabel("M tonnes (5 ans cumulés)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_top_species.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/07_top_species.png")


# ── 9. Corrélations avec dataset principal (intégration future) ───────────
print("\n[9] Corrélations avec dataset principal V8…")
df_main = pd.read_csv(f"{OUT_DIR}/dataset_final_v8_honest.csv", low_memory=False)
df_main = df_main.dropna(subset=["ISO"])

# Joindre fish_total_t
merged = df_main.merge(df_total, on=["ISO","Annee"], how="left")
print(f"   Lignes merged : {len(merged):,}")
print(f"   fish_total_t couverture : {merged['fish_total_t'].notna().sum():,}")

# Top corrélations avec fish_total_t (log)
merged["log_fish"] = np.log1p(merged["fish_total_t"].fillna(0))
num = merged.select_dtypes(include="number").columns.tolist()
excl = {"ISO","Annee","T_ref","P_ref","cluster","fish_total_t",
        "fish_species_count","fish_isscaap_count","log_fish"}
feats = [c for c in num if c not in excl]
cors = merged[feats].corrwith(merged["log_fish"]).abs().sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(11, 8))
cors.sort_values().plot(kind="barh", color="teal", alpha=0.85, ax=ax)
ax.set_title("Top 20 |corrélations| features dataset principal → fish_total_t (log)",
              weight="bold", fontsize=12)
ax.set_xlabel("|corrélation|")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_correlations.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/08_correlations.png")

# Matrice de corrélation interne fish + quelques features clés
key_feats = ["fish_total_t","fish_species_count","fish_isscaap_count"]
for c in ["Population","GDP_pc","dist_to_coast_km","Urban_pct","nasa_t2m",
          "nasa_prectotcorr","forest_share_pct","Engrais_kgha"]:
    if c in merged.columns:
        key_feats.append(c)
mat = merged[key_feats].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Matrice corrélation : fish vs features clés", weight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_corr_matrix.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"   → {FIG_DIR}/09_corr_matrix.png")


# ── 10. Variation par cluster climatique ──────────────────────────────────
if "cluster" in merged.columns:
    print("\n[10] Variation par cluster…")
    recent_merged = merged[merged["Annee"] >= 2010]
    by_cluster = recent_merged.groupby("cluster").agg(
        median_total=("fish_total_t", "median"),
        median_species=("fish_species_count", "median"),
        countries=("ISO", "nunique"),
    ).round(0)
    print(by_cluster.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    cluster_data = [recent_merged[recent_merged["cluster"]==c]["fish_total_t"].dropna()
                     for c in sorted(recent_merged["cluster"].dropna().unique())]
    bp = ax.boxplot(cluster_data, labels=sorted(recent_merged["cluster"].dropna().unique().astype(int)),
                     patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], sorted(recent_merged["cluster"].dropna().unique().astype(int))):
        patch.set_facecolor(plt.cm.tab10(c)); patch.set_alpha(0.7)
    ax.set_yscale("log"); ax.set_xlabel("Cluster"); ax.set_ylabel("Production (tonnes, log)")
    ax.set_title("Production par cluster climatique (2010-2022)", weight="bold")

    ax = axes[1]
    sp_data = [recent_merged[recent_merged["cluster"]==c]["fish_species_count"].dropna()
                for c in sorted(recent_merged["cluster"].dropna().unique())]
    bp = ax.boxplot(sp_data, labels=sorted(recent_merged["cluster"].dropna().unique().astype(int)),
                     patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], sorted(recent_merged["cluster"].dropna().unique().astype(int))):
        patch.set_facecolor(plt.cm.tab10(c)); patch.set_alpha(0.7)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Diversité espèces")
    ax.set_title("Diversité d'espèces par cluster", weight="bold")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/10_by_cluster.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"   → {FIG_DIR}/10_by_cluster.png")


print(f"\n[DONE] 10 figures générées dans {FIG_DIR}/")
print("   À utiliser : intégrer fish_total_t / fish_Capture_t / fish_Aquaculture_*_t")
print("   comme nouvelles cibles ou features dans la Couche 1.")
