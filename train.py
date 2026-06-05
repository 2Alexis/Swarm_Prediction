"""
Script d'entraînement du pipeline de Machine Learning pour la prédiction du rendement agricole.
Prend en compte les relations non-linéaires, les intrants, l'effet temporel, et évite les fuites de données.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Configurer les chemins
DATA_DIR = "data/cleaned"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def clean_columns(df):
    """Nettoie les colonnes mal encodées."""
    rename_dict = {}
    for col in df.columns:
        if 'l' in col.lower() and 'm' in col.lower() and 'nt' in col.lower():
            rename_dict[col] = 'Element'
        elif 'unit' in col.lower():
            rename_dict[col] = 'Unite'
    return df.rename(columns=rename_dict)

def load_and_merge_data():
    print("[INFO] Chargement et fusion des donnees...")
    
    # Chargement des datasets
    df_prod = clean_columns(pd.read_csv(f"{DATA_DIR}/production_cultures.csv"))
    df_temp = pd.read_csv(f"{DATA_DIR}/mean_temperature.csv").rename(columns={'Valeur': 'Temperature_C'})
    df_precip = pd.read_csv(f"{DATA_DIR}/precipitations.csv").rename(columns={'Valeur': 'Precipitations_mm'})
    df_fert = pd.read_csv(f"{DATA_DIR}/fertilizers_nutrient.csv")
    df_pest = pd.read_csv(f"{DATA_DIR}/pesticides.csv")
    df_sols = pd.read_csv(f"{DATA_DIR}/bilan_nutritif_sols.csv").rename(columns={'Valeur': 'Bilan_sols_kgha'})
    
    # 1. Préparer la cible (Rendement)
    df_yield = df_prod[df_prod['Element'] == 'Rendement'].copy()
    df_yield = df_yield.rename(columns={'Valeur': 'Rendement_kgha'})
    # Supprimer les outliers extrêmes découverts lors de l'EDA
    df_yield = df_yield[df_yield['Rendement_kgha'] <= 100000]
    df_yield = df_yield[df_yield['Rendement_kgha'] > 0] # Echelle log requiert des valeurs > 0
    
    # 2. Agrégation des intrants par pays/année
    df_fert_agg = df_fert.groupby(['Pays', 'Annee'])['Valeur'].sum().reset_index().rename(columns={'Valeur': 'Engrais_kgha'})
    df_pest_agg = df_pest.groupby(['Pays', 'Annee'])['Valeur'].sum().reset_index().rename(columns={'Valeur': 'Pesticides_kgha'})
    
    # Agrégation climat (au cas où il y a des doublons)
    df_temp_agg = df_temp.groupby(['Pays', 'Annee'])['Temperature_C'].mean().reset_index()
    df_precip_agg = df_precip.groupby(['Pays', 'Annee'])['Precipitations_mm'].mean().reset_index()
    
    # 3. Fusion des données
    # Jointure crop-specific (Rendement + Bilan sols)
    df_master = pd.merge(
        df_yield[['Pays', 'Produit', 'Annee', 'Rendement_kgha']],
        df_sols[['Pays', 'Produit', 'Annee', 'Bilan_sols_kgha']],
        on=['Pays', 'Produit', 'Annee'],
        how='left'
    )
    
    # Jointures géographiques/temporelles
    df_master = pd.merge(df_master, df_temp_agg, on=['Pays', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_precip_agg, on=['Pays', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_fert_agg, on=['Pays', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_pest_agg, on=['Pays', 'Annee'], how='left')
    
    print(f"[SUCCESS] Donnees fusionnees avec succes. Nombre total de lignes : {len(df_master):,}")
    return df_master

def engineer_features(df):
    print("[INFO] Feature Engineering...")
    
    # Création des features non-linéaires basées sur les conclusions EDA
    df['Temperature_C_sq'] = df['Temperature_C'] ** 2
    df['Precipitations_mm_sq'] = df['Precipitations_mm'] ** 2
    df['Engrais_Temp_interaction'] = df['Engrais_kgha'] * df['Temperature_C']
    
    # Transformation de la cible
    df['log_Rendement'] = np.log1p(df['Rendement_kgha'])
    
    # Drop rows without targets
    df = df.dropna(subset=['log_Rendement'])
    
    return df

def build_preprocessing_pipeline(num_cols, cat_cols):
    # Pipeline pour les variables numériques
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline pour les variables catégorielles
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combiner les deux
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    return preprocessor

def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    print(f"\n===== Evaluation du modele : {name} =====")
    
    # Prédictions sur l'échelle log
    y_pred_train_log = model.predict(X_train)
    y_pred_test_log = model.predict(X_test)
    
    # Conversion inverse vers l'échelle réelle
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)
    y_pred_train_orig = np.expm1(y_pred_train_log)
    y_pred_test_orig = np.expm1(y_pred_test_log)
    
    # Calcul des métriques sur l'échelle d'origine (Rendement réel en kg/ha)
    r2_train = r2_score(y_train_orig, y_pred_train_orig)
    r2_test = r2_score(y_test_orig, y_pred_test_orig)
    
    mae_train = mean_absolute_error(y_train_orig, y_pred_train_orig)
    mae_test = mean_absolute_error(y_test_orig, y_pred_test_orig)
    
    rmse_train = root_mean_squared_error(y_train_orig, y_pred_train_orig)
    rmse_test = root_mean_squared_error(y_test_orig, y_pred_test_orig)
    
    print(f"Train R2 : {r2_train:.4f} | Test R2 : {r2_test:.4f}")
    print(f"Train MAE: {mae_train:.2f} kg/ha | Test MAE: {mae_test:.2f} kg/ha")
    print(f"Train RMSE: {rmse_train:.2f} kg/ha | Test RMSE: {rmse_test:.2f} kg/ha")
    
    return {
        'model': model,
        'r2_test': r2_test,
        'metrics': {'R2': r2_test, 'MAE': mae_test, 'RMSE': rmse_test}
    }

def main():
    # 1. Charger et nettoyer
    df = load_and_merge_data()
    df = engineer_features(df)
    
    # Définir les colonnes
    num_cols = [
        'Annee', 'Temperature_C', 'Temperature_C_sq', 
        'Precipitations_mm', 'Precipitations_mm_sq', 
        'Engrais_kgha', 'Pesticides_kgha', 'Bilan_sols_kgha',
        'Engrais_Temp_interaction'
    ]
    cat_cols = ['Pays', 'Produit']
    
    # 2. Split temporel (Train <= 2013 | Test > 2013)
    # Ce split évite le data leakage temporel
    print("\n[INFO] Division temporelle des donnees...")
    df_train = df[df['Annee'] <= 2013]
    df_test = df[df['Annee'] > 2013]
    
    X_train_raw = df_train[num_cols + cat_cols]
    y_train = df_train['log_Rendement']
    
    X_test_raw = df_test[num_cols + cat_cols]
    y_test = df_test['log_Rendement']
    
    print(f"   Train set : {X_train_raw.shape[0]:,} lignes (annees <= 2013)")
    print(f"   Test set  : {X_test_raw.shape[0]:,} lignes (annees > 2013)")
    
    # 3. Pipeline de Preprocessing
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    
    # 4. Définir et entraîner les modèles
    models = {
        "Baseline Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    }
    
    best_r2 = -float('inf')
    best_model_name = None
    best_pipeline = None
    
    results = {}
    
    for name, model in models.items():
        print(f"\n[INFO] Entrainement de {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Entraîner le pipeline complet
        pipeline.fit(X_train_raw, y_train)
        
        # Évaluer
        eval_res = evaluate_model(pipeline, X_train_raw, y_train, X_test_raw, y_test, name)
        results[name] = eval_res
        
        # Enregistrer le meilleur modèle
        if eval_res['r2_test'] > best_r2:
            best_r2 = eval_res['r2_test']
            best_model_name = name
            best_pipeline = pipeline

    # 5. Sauvegarder le meilleur modèle
    print(f"\n[SUCCESS] Meilleur modele : {best_model_name} avec un Test R2 de {best_r2:.4f}")
    model_path = f"{MODEL_DIR}/best_model_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"💾 Pipeline complet sauvegarde dans : {model_path}")
    
    # Sauvegarder un échantillon de test pour vérification future
    test_sample_path = f"{DATA_DIR}/test_sample.csv"
    df_test.sample(min(1000, len(df_test)), random_state=42).to_csv(test_sample_path, index=False)
    print(f"📝 Echantillon de test sauvegarde dans : {test_sample_path}")

if __name__ == "__main__":
    main()
