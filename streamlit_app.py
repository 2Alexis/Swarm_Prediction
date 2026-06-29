# -*- coding: utf-8 -*-
"""
Swarm_Prediction — Dashboard « De la terre aux gens »
Tableau de bord interactif organisé par Couche -> Sous-couche.
Pour chaque sous-couche : performance des modèles (R²) + exploration des vraies données
(cartes choroplèthes, distributions, top pays, évolution temporelle).

Lancer :  streamlit run streamlit_app.py
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
V15 = os.path.join(BASE, "data", "cleaned", "dataset_final_v15_couche1.csv")
C2_FILE = os.path.join(BASE, "data", "couche2", "dataset_couche2.csv")
RES1 = os.path.join(BASE, "couche1_planete", "reports", "results.csv")
RES2 = os.path.join(BASE, "couche2_sang", "reports", "results.csv")

st.set_page_config(page_title="Swarm_Prediction — De la terre aux gens",
                   page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Thème visuel (cohérent avec la présentation : « couches de terre »)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,700;9..144,900&family=Inter:wght@400;500;600&display=swap');
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#f3ece0 !important; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; color:#2a2018; }
/* forcer le texte sombre dans la zone principale (anti « blanc sur blanc » en thème sombre) */
[data-testid="stMain"] p, [data-testid="stMain"] span, [data-testid="stMain"] label,
[data-testid="stMain"] li, [data-testid="stMain"] div, [data-testid="stMain"] td,
[data-testid="stMain"] th, [data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
.stSelectbox div, .stRadio label { color:#2a2018; }
h1,h2,h3,h4 { font-family:'Fraunces',serif !important; color:#3c2a1e !important; letter-spacing:-.01em; }
section[data-testid="stSidebar"] { background:#3c2a1e; }
section[data-testid="stSidebar"] * { color:#f3e6cf !important; }
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2 { color:#fff !important; }
/* le selectbox a un fond clair -> son texte doit rester sombre */
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-baseweb="input"] input { color:#2a2018 !important; }
ul[role="listbox"] li, [data-baseweb="popover"] li { color:#2a2018 !important; }
.hero { background:linear-gradient(110deg,#6b4a2f,#3c2a1e); color:#f3ece0; padding:18px 26px;
        border-radius:16px; margin-bottom:14px; }
.hero h1 { color:#fff !important; margin:0; font-size:1.9rem; }
.hero p  { margin:.25rem 0 0; color:#e7d6bb; font-size:.95rem; }
.subdesc { background:#fbf6ec; border:1px solid rgba(60,42,30,.15); border-left:4px solid #c97b3c;
           border-radius:12px; padding:12px 16px; margin-bottom:10px; color:#4a3829; font-size:.95rem; }
div[data-testid="stMetric"] { background:#fbf6ec; border:1px solid rgba(60,42,30,.15);
           border-radius:12px; padding:12px 14px; }
div[data-testid="stMetricValue"] { font-family:'Fraunces',serif; color:#3c2a1e; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:#efe4d2; border-radius:8px 8px 0 0; padding:6px 16px; }
.stTabs [aria-selected="true"] { background:#c97b3c; color:#fff; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Conversion ISO -> alpha-3 (pour les cartes)
@st.cache_data(show_spinner=False)
def iso3_map(codes, names):
    try:
        import pycountry
    except Exception:
        return {}
    out = {}
    for code, name in zip(codes, names):
        a3 = None
        c = str(code).strip()
        try:
            if len(c) == 2:
                o = pycountry.countries.get(alpha_2=c)
                a3 = o.alpha_3 if o else None
            elif len(c) == 3 and c.isalpha():
                o = pycountry.countries.get(alpha_3=c)
                a3 = o.alpha_3 if o else c
        except Exception:
            a3 = None
        if a3 is None and isinstance(name, str):
            try:
                o = pycountry.countries.lookup(name)
                a3 = o.alpha_3
            except Exception:
                a3 = None
        out[code] = a3
    return out

# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement Couche 1 (La Planète)…")
def load_c1():
    use = lambda c: c in ("ISO", "Annee", "Country", "cluster") or c.startswith("target_")
    df = pd.read_csv(V15, usecols=use, low_memory=False)
    df = df.dropna(subset=["ISO"]).copy()
    df["Annee"] = df["Annee"].astype(int)
    name_by_iso = (df.drop_duplicates("ISO").set_index("ISO")["Country"].to_dict()
                   if "Country" in df.columns else {})
    isos = list(name_by_iso.keys()) or df["ISO"].unique().tolist()
    names = list(name_by_iso.values()) or [""] * len(isos)
    df["ISO3"] = df["ISO"].map(iso3_map(isos, names))
    if "Country" not in df.columns:
        df["Country"] = df["ISO"]
    return df

@st.cache_data(show_spinner="Chargement Couche 2 (Le Sang)…")
def load_c2():
    df = pd.read_csv(C2_FILE)
    df["Annee"] = df["Annee"].astype(int)
    codes = df.drop_duplicates("Code_Pays")[["Code_Pays", "Pays"]]
    m = iso3_map(codes["Code_Pays"].tolist(), codes["Pays"].tolist())
    df["ISO3"] = df["Code_Pays"].map(m)
    return df

@st.cache_data(show_spinner=False)
def load_results():
    r1 = pd.read_csv(RES1); r2 = pd.read_csv(RES2)
    return r1, r2

# ----------------------------------------------------------------------------
# Mapping des 7 sous-couches THÉMATIQUES de la Couche 1 (demande utilisateur)
SOUS_TO_THEME = {
    "Émissions atmosphériques": "🌫️ Atmosphère",
    "Écologie": "🪨 Sols & écologie",
    "Énergie": "⚡ Énergies",
    "Céréales & racines": "🌾 Agriculture",
    "Oléagineux (par culture)": "🌾 Agriculture",
    "Fruits (par culture)": "🌾 Agriculture",
    "Légumes (par culture)": "🌾 Agriculture",
    "Légumineuses (par culture)": "🌾 Agriculture",
    "Élevage": "🐄 Élevage",
}
TARGET_OVERRIDE = {
    "target_thermal_anomaly": "🌫️ Atmosphère",
    "target_water_stress": "💧 Hydrologie",
    "target_water_access": "💧 Hydrologie",
    "target_soil_moisture_root": "💧 Hydrologie",
    "target_soil_degradation": "🪨 Sols & écologie",
    "target_aquaculture": "🐟 Pêche",
}
C1_DESC = {
    "🌫️ Atmosphère": "Gaz à effet de serre (CO₂, CH₄, N₂O) et anomalies de température : ce que respire le pays.",
    "💧 Hydrologie": "L'eau du pays : stress hydrique, accès à l'eau potable, humidité des sols.",
    "🪨 Sols & écologie": "L'état des sols et du couvert végétal : forêts, dégradation, déforestation.",
    "🌾 Agriculture": "Les rendements agricoles, culture par culture (céréales, fruits, légumes, oléagineux, légumineuses).",
    "🐄 Élevage": "La production animale : lait, œufs, viandes (carcasses bovine/poulet/porc/ovin).",
    "🐟 Pêche": "La production halieutique : aquaculture (+ exploration FAO capture / espèces ISSCAAP).",
    "⚡ Énergies": "La production énergétique : charbon, pétrole, gaz, solaire, éolien, hydroélectrique.",
}
C1_ORDER = ["🌫️ Atmosphère", "💧 Hydrologie", "🪨 Sols & écologie", "🌾 Agriculture",
            "🐄 Élevage", "🐟 Pêche", "⚡ Énergies"]

# Couche 2
C2_SUB = {
    "👶 Démographie": {
        "res": "Démographie",
        "desc": "Natalité, fécondité, mortalité, migration — prédites depuis la seule Terre (variables socio-éco interdites).",
        "explore": [("Natalité (‰)", "Natalite_pour1000"),
                    ("Fécondité (enfants/femme)", "Fecondite_enf_par_femme"),
                    ("Mortalité infantile <5 ans (‰)", "MortMoins5ans_pour1000"),
                    ("Mortalité brute (‰)", "MortBrute_pour1000"),
                    ("Migration nette", "MigrationNette"),
                    ("Densité (hab/km²)", "Densite_hab_km2")],
    },
    "🩹 Santé & nutrition": {
        "res": "Santé",
        "desc": "Malnutrition (carence alimentaire) et retard de croissance (stunting).",
        "explore": [("Carence / malnutrition (%)", "Carence_pct")],
    },
    "🌪️ Catastrophes humaines": {
        "res": "Catastrophes humaines",
        "desc": "Décès et personnes affectées par les catastrophes naturelles (EM-DAT, échelle log).",
        "explore": [],
    },
}

def qual_color(r):
    if r is None or (isinstance(r, float) and np.isnan(r)): return "#b7a98c"
    if r >= 0.7: return "#3f7d4e"
    if r >= 0.4: return "#c98a2e"
    if r >= 0:   return "#b7a98c"
    return "#a83c34"

# ----------------------------------------------------------------------------
def perf_chart(sub_df, title):
    d = sub_df.copy()
    d["R2"] = d["R² Global"]
    d = d.sort_values("R2")
    d["col"] = d["R2"].apply(qual_color)
    fig = px.bar(d, x="R2", y="Cible", orientation="h",
                 title=title, text=d["R2"].map(lambda v: f"{v:.2f}".replace(".", ",")))
    fig.update_traces(marker_color=d["col"], textposition="outside", cliponaxis=False)
    fig.update_layout(height=max(260, 34*len(d)+90), xaxis_title="R² (test sur pays jamais vus)",
                      yaxis_title="", margin=dict(l=10, r=30, t=50, b=10),
                      plot_bgcolor="#fbf6ec", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#3c2a1e"), title_font=dict(size=16))
    fig.add_vline(x=0, line_color="#8a7560", line_width=1)
    return fig

def map_chart(df, col, label, scale):
    d = df.dropna(subset=[col, "ISO3"])
    if d.empty: return None
    yr = int(d.loc[d[col].notna(), "Annee"].max())
    dy = d[d["Annee"] == yr]
    fig = px.choropleth(dy, locations="ISO3", color=col, hover_name="Country" if "Country" in dy.columns else "Pays",
                        color_continuous_scale=scale, locationmode="ISO-3",
                        title=f"{label} — carte mondiale ({yr})")
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=46, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#3c2a1e"),
                      geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False))
    fig.update_coloraxes(colorbar_title="")
    return fig, yr

def dist_chart(df, col, label):
    d = df.dropna(subset=[col])
    if d.empty: return None
    fig = px.histogram(d, x=col, nbins=40, title=f"Distribution — {label}")
    fig.update_traces(marker_color="#c97b3c")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=46, b=10), bargap=.05,
                      plot_bgcolor="#fbf6ec", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#3c2a1e"), yaxis_title="pays·années", xaxis_title=label)
    return fig

def top_chart(df, col, label, namecol):
    d = df.dropna(subset=[col, "ISO3"])
    if d.empty: return None
    yr = int(d.loc[d[col].notna(), "Annee"].max())
    dy = d[d["Annee"] == yr].sort_values(col, ascending=False).head(12)
    fig = px.bar(dy, x=col, y=namecol, orientation="h", title=f"Top 12 pays ({yr}) — {label}")
    fig.update_traces(marker_color="#42692f")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=46, b=10),
                      yaxis=dict(autorange="reversed"), xaxis_title=label, yaxis_title="",
                      plot_bgcolor="#fbf6ec", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#3c2a1e"))
    return fig

def evol_chart(df, col, label):
    d = df.dropna(subset=[col])
    if d.empty: return None
    g = d.groupby("Annee")[col].mean().reset_index()
    fig = px.line(g, x="Annee", y=col, title=f"Évolution mondiale (moyenne) — {label}", markers=True)
    fig.update_traces(line_color="#9b2d2d")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=46, b=10),
                      plot_bgcolor="#fbf6ec", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#3c2a1e"), xaxis_title="", yaxis_title=label)
    return fig

# ----------------------------------------------------------------------------
r1, r2 = load_results()

# Construire le mapping thème -> cibles (Couche 1) depuis results
r1 = r1.copy()
r1["theme"] = [TARGET_OVERRIDE.get(t, SOUS_TO_THEME.get(s, "🌾 Agriculture"))
               for t, s in zip(r1["Technique"], r1["Sous-couche"])]

# ----------------------------------------------------------------------------
# SIDEBAR
st.sidebar.markdown("## 🌍 Swarm_Prediction")
st.sidebar.caption("De la terre aux gens — tableau de bord par couches")
couche = st.sidebar.radio("Couche", ["🌍 Couche 1 — La Planète", "🩸 Couche 2 — Le Sang"])

if couche.startswith("🌍"):
    sous = st.sidebar.selectbox("Sous-couche", C1_ORDER)
else:
    sous = st.sidebar.selectbox("Sous-couche", list(C2_SUB.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("**Lecture du R²**")
st.sidebar.markdown("🟢 ≥ 0,7 fort &nbsp; 🟠 0,4–0,7 &nbsp; ⚪ < 0,4 &nbsp; 🔴 négatif")
st.sidebar.caption("R² évalué par **GroupShuffleSplit** : test sur des pays jamais vus.")
st.sidebar.markdown("---")
st.sidebar.caption("Données 100 % réelles : FAO · World Bank · WorldClim · NASA · OWID · Hansen · Pew · NOAA.")

# ----------------------------------------------------------------------------
# HERO
st.markdown(f"""<div class="hero"><h1>{couche}</h1>
<p>{'62 cibles environnement & agriculture, réparties en 7 sous-couches.' if couche.startswith('🌍')
   else 'La démographie et la santé prédites à partir de la seule Couche 1.'}</p></div>""",
            unsafe_allow_html=True)

# ============================================================================
if couche.startswith("🌍"):           # ---------- COUCHE 1 ----------
    df = load_c1()
    sub = r1[r1["theme"] == sous].copy()
    st.subheader(sous)
    st.markdown(f"<div class='subdesc'>{C1_DESC[sous]}</div>", unsafe_allow_html=True)

    rr = sub["R² Global"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cibles", len(sub))
    c2.metric("Meilleur R²", f"{rr.max():.2f}".replace(".", ","))
    c3.metric("R² moyen", f"{rr.mean():.2f}".replace(".", ","))
    c4.metric("R² ≥ 0,7", int((rr >= 0.7).sum()))

    t1, t2 = st.tabs(["📊 Performance des modèles", "🗺️ Exploration des données"])
    with t1:
        st.plotly_chart(perf_chart(sub, f"R² par cible — {sous}"), use_container_width=True)
        st.dataframe(sub[["Cible", "R² Global", "MAE", "N obs", "N pays test"]]
                     .rename(columns={"R² Global": "R²"}).sort_values("R²", ascending=False),
                     use_container_width=True, hide_index=True)
    with t2:
        opts = {row["Cible"]: row["Technique"] for _, row in sub.iterrows()
                if row["Technique"] in df.columns}
        if not opts:
            st.info("Pas de données géographiques disponibles pour cette sous-couche.")
        else:
            label = st.selectbox("Choisir une cible à explorer", list(opts.keys()))
            col = opts[label]
            scale = "YlGn" if sous in ("🪨 Sols & écologie", "🌾 Agriculture") else "YlOrBr"
            mc = map_chart(df, col, label, scale)
            if mc:
                st.plotly_chart(mc[0], use_container_width=True)
            a, b = st.columns(2)
            f1 = top_chart(df, col, label, "Country"); f2 = dist_chart(df, col, label)
            if f1: a.plotly_chart(f1, use_container_width=True)
            if f2: b.plotly_chart(f2, use_container_width=True)
            f3 = evol_chart(df, col, label)
            if f3: st.plotly_chart(f3, use_container_width=True)

else:                                  # ---------- COUCHE 2 ----------
    df2 = load_c2()
    conf = C2_SUB[sous]
    sub = r2[r2["Sous-couche"] == conf["res"]].copy()
    st.subheader(sous)
    st.markdown(f"<div class='subdesc'>{conf['desc']}</div>", unsafe_allow_html=True)

    rr = sub["R² Global"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cibles", len(sub))
    c2.metric("Meilleur R²", f"{rr.max():.2f}".replace(".", ",") if len(rr) else "—")
    c3.metric("R² moyen", f"{rr.mean():.2f}".replace(".", ",") if len(rr) else "—")

    t1, t2 = st.tabs(["📊 Performance des modèles", "🗺️ Exploration des données"])
    with t1:
        st.plotly_chart(perf_chart(sub, f"R² par cible — {sous}"), use_container_width=True)
        st.dataframe(sub[["Cible", "R² Global", "MAE", "N obs", "N pays test"]]
                     .rename(columns={"R² Global": "R²"}).sort_values("R²", ascending=False),
                     use_container_width=True, hide_index=True)
    with t2:
        ex = [(lab, c) for lab, c in conf["explore"] if c in df2.columns]
        if not ex:
            st.info("Sous-couche modélisée à partir de sources externes (EM-DAT) — pas de carte ici. "
                    "Voir l'onglet performance.")
        else:
            labels = {lab: c for lab, c in ex}
            label = st.selectbox("Choisir une variable à explorer", list(labels.keys()))
            col = labels[label]
            mc = map_chart(df2, col, label, "OrRd")
            if mc:
                st.plotly_chart(mc[0], use_container_width=True)
            a, b = st.columns(2)
            f1 = top_chart(df2, col, label, "Pays"); f2 = dist_chart(df2, col, label)
            if f1: a.plotly_chart(f1, use_container_width=True)
            if f2: b.plotly_chart(f2, use_container_width=True)
            f3 = evol_chart(df2, col, label)
            if f3: st.plotly_chart(f3, use_container_width=True)

st.markdown("---")
st.caption("Swarm_Prediction · De la terre aux gens — données réelles & publiques · "
           "R² évalué par pays (GroupShuffleSplit) · XGBoost.")
