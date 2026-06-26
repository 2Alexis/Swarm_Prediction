"""Téléchargement religion (Pew/OWID) + meat consumption per type."""
import os, sys, io, urllib.request, json
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"


def fetch_text(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def owid_csv(slug, name, keep_all_num=False):
    try:
        url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
        txt = fetch_text(url, timeout=120)
        df = pd.read_csv(io.StringIO(txt))
        df.columns = [c.strip() for c in df.columns]
        ent = "Entity" if "Entity" in df.columns else df.columns[0]
        yr  = "Year"   if "Year"   in df.columns else df.columns[2]
        val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
        num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols: return False
        if keep_all_num and len(num_cols) > 1:
            # Garder toutes les colonnes numériques (pour datasets multi-valeurs)
            out = df[[ent, yr] + num_cols].rename(columns={ent: "Pays", yr: "Annee"})
            out = out.dropna(subset=["Pays", "Annee"])
            out.to_csv(f"{D}/{name}.csv", index=False)
            print(f"  + {name}: {len(out)} lignes, {len(num_cols)} colonnes [{', '.join(num_cols)}]")
        else:
            out = df[[ent, yr, num_cols[0]]].rename(columns={ent: "Pays", yr: "Annee", num_cols[0]: "Valeur"})
            out = out.dropna(subset=["Pays", "Annee", "Valeur"])
            out.to_csv(f"{D}/{name}.csv", index=False)
            print(f"  + {name}: {len(out)} lignes")
        return True
    except Exception as e:
        print(f"  ✗ {slug}: {e}")
        return False


# ── 1. Religion (Pew / OWID) ───────────────────────────────────────────────
print("[1] Religion (Pew/OWID)…")
RELIGION_SLUGS = [
    ("religious-affiliation",                        "owid_religion_affiliation"),
    ("religion-pew",                                  "owid_religion_pew"),
    ("religious-population-by-country",              "owid_religion_pop"),
    ("religion-projections",                          "owid_religion_projections"),
    ("share-of-population-by-religion",              "owid_religion_share"),
    ("muslim-percentage-of-population",              "owid_muslim_pct"),
    ("christianity-pct-pop",                          "owid_christian_pct"),
    ("religiosity-religion-importance-life",          "owid_religiosity"),
    ("religious-composition-by-country",              "owid_religion_composition"),
]
for slug, name in RELIGION_SLUGS:
    owid_csv(slug, name, keep_all_num=True)


# ── 2. Meat consumption détaillé par type ─────────────────────────────────
print("\n[2] Meat consumption par type…")
MEAT_SLUGS = [
    ("per-capita-meat-consumption-by-type-kilograms-per-year", "owid_meat_by_type"),
    ("meat-supply-per-person",                                  "owid_meat_supply_pp"),
    ("per-capita-poultry-consumption",                          "owid_poultry_consumption"),
    ("per-capita-beef-consumption",                             "owid_beef_consumption"),
    ("per-capita-pork-consumption",                             "owid_pork_consumption"),
    ("per-capita-sheep-and-goat-consumption",                   "owid_sheepgoat_consumption"),
    ("per-capita-fish-and-seafood-consumption",                 "owid_fish_consumption_pp"),
    ("per-capita-milk-consumption",                             "owid_milk_consumption"),
    ("per-capita-egg-consumption-kilograms-per-year",           "owid_egg_consumption"),
    ("meat-production-tonnes",                                   "owid_meat_production_t"),
]
for slug, name in MEAT_SLUGS:
    owid_csv(slug, name, keep_all_num=True)


# ── 3. Pew Research direct (tentative) ─────────────────────────────────────
print("\n[3] Pew Research direct…")
# Pew n'a pas d'URL CSV publique stable, mais Wikipedia maintient
# une bonne liste. Alternative : OWID religion-by-country
# Let's try the Wikipedia-like sources via OWID
WIKI_SLUGS = [
    ("share-religious-affiliation",                  "owid_relig_share_alt"),
    ("share-of-population-that-is-religious",        "owid_religious_share"),
]
for slug, name in WIKI_SLUGS:
    owid_csv(slug, name, keep_all_num=True)


# ── 4. Alimentation / régime ──────────────────────────────────────────────
print("\n[4] Alimentation supplémentaire…")
FOOD_SLUGS = [
    ("daily-caloric-supply-from-meat-eggs-fish-dairy", "owid_calories_animal"),
    ("share-of-calories-from-animal-vs-plant",         "owid_share_animal_calories"),
    ("vegetable-supply-per-person",                    "owid_veg_supply"),
    ("fruit-supply-per-person",                        "owid_fruit_supply"),
    ("food-supply-kcal-per-day-per-person",            "owid_food_supply_kcal"),
]
for slug, name in FOOD_SLUGS:
    owid_csv(slug, name, keep_all_num=True)


print("\n[DONE]")
