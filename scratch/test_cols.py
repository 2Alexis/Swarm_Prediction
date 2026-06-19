import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
from clean_data import find_col, load_csv

path = 'data/raw/pedologie_sols/sol_nutritif/Environnement_Bilan_nutritif_des_sols_F_Toutes_les_Données_(Normalisé).csv'
df = load_csv(path)
cols = list(df.columns)

def escape_str(s):
    if s is None:
        return 'None'
    return str(s).encode('ascii', 'backslashreplace').decode('ascii')

print("All columns:", [escape_str(c) for c in cols])

print("col_unite:", escape_str(find_col(df, 'Unité', 'Unit')))
print("col_element:", escape_str(find_col(df, 'Élément', 'Element')))
print("col_zone:", escape_str(find_col(df, 'Zone', 'Area')))
print("col_annee:", escape_str(find_col(df, 'Année', 'Year')))
print("col_valeur:", escape_str(find_col(df, 'Valeur', 'Value')))
print("col_produit:", escape_str(find_col(df, 'Produit', 'Item')))
