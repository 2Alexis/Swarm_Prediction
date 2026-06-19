import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df_prod = pd.read_csv('data/cleaned/production_cultures.csv')
df_fert = pd.read_csv('data/cleaned/fertilizers_nutrient.csv')
df_temp = pd.read_csv('data/cleaned/mean_temperature.csv')
df_precip = pd.read_csv('data/cleaned/precipitations.csv')

def search_country(df, name, df_name):
    unique_names = df['Pays'].unique()
    matches = [c for c in unique_names if name.lower() in str(c).lower()]
    print(f"Matches in {df_name} for '{name}': {matches}")

for df_name, df in [('prod', df_prod), ('fert', df_fert), ('temp', df_temp), ('precip', df_precip)]:
    search_country(df, 'eston', df_name)
    search_country(df, 'letton', df_name)
    search_country(df, 'litua', df_name)
    search_country(df, 'liby', df_name)
