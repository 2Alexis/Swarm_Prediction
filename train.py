"""
Script d'entraînement du pipeline de Machine Learning pour la prédiction du rendement agricole.
Prend en compte les caractéristiques physiques planétaires (Layer 1), les intrants, 
les indicateurs de développement, et évite les fuites de données par un split temporel.
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

def load_data():
    dataset_path = os.path.join(DATA_DIR, "dataset_final_modelisation.csv")
    print(f"[INFO] Chargement du dataset final : {dataset_path}...")
    df = pd.read_csv(dataset_path)
    return df

def engineer_features(df):
    print("[INFO] Feature Engineering...")
    
    # Réintroduire les relations quadratiques et les interactions basées sur l'analyse
    df['Temperature_C_sq'] = df['Temperature_C'] ** 2
    df['Precipitations_mm_sq'] = df['Precipitations_mm'] ** 2
    df['Engrais_Temp_interaction'] = df['Engrais_kgha'] * df['Temperature_C']
    
    # Transformation de la cible
    df['log_Rendement'] = np.log1p(df['Rendement_kgha'])
    
    # Suppression des lignes n'ayant pas de cible valide
    df = df.dropna(subset=['log_Rendement'])
    
    return df

def build_preprocessing_pipeline(num_cols, cat_cols):
    # Pipeline pour les variables numériques
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    transformers = [('num', num_transformer, num_cols)]
    
    # Pipeline pour les variables catégorielles (si spécifiées)
    if cat_cols:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, cat_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers)
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
    # 1. Charger et préparer les données
    df = load_data()
    df = engineer_features(df)
    
    # Détecter dynamiquement les colonnes de type One-Hot-Encoded (pd.get_dummies) de build_dataset.py
    dummy_prefixes = ('closest_volcano_type_', 'closest_energy_type_', 'closest_mineral_type_', 'closest_fossil_type_')
    dummy_cols = [col for col in df.columns if col.startswith(dummy_prefixes)]
    
    # Définition de toutes les colonnes numériques incluant les features physiques et les variables d'aléa encodées
    num_cols = [
        # Variables temporelles et intrants
        'Annee', 'Engrais_kgha', 'Pesticides_kgha', 'Bilan_sols_kgha',
        'log_Engrais_kgha', 'log_Pesticides_kgha',
        
        # Climat annuel localisé
        'Temperature_C', 'Temperature_C_sq', 
        'Precipitations_mm', 'Precipitations_mm_sq', 
        'Engrais_Temp_interaction',
        
        # Socio-démographie & Santé
        'GDP_pc', 'Life_Exp', 'Child_Mort', 'HDI', 'Malaria_Incidence', 'Water_Withdrawal_pct',
        
        # Features Physiques du Moteur (Climat de référence)
        'temp_mean', 'temp_max', 'temp_min', 'precip_mean', 'precip_seasonality',
        'wind_speed_mean', 'solar_radiation_mean', 'vapor_pressure_mean',
        
        # Topographie & Relief
        'elevation', 'slope_pct', 'aspect_deg', 'roughness_m',
        
        # Hydrologie
        'dist_to_river_km', 'dist_to_lake_km', 'dist_to_coast_km', 'dist_to_freshwater_km', 'groundwater_depth_m',
        
        # Sols (Qualité de base)
        'clay_pct', 'silt_pct', 'sand_pct', 'soil_pH', 'organic_carbon_pct',
        
        # Risques physiques
        'dist_to_fault_km', 'Seismic_Hazard_Index', 'dist_to_volcano_km', 'Volcanic_Hazard_Index',
        
        # Proximité ressources
        'dist_to_mineral_resource_km', 'dist_to_fossil_resource_km', 'dist_to_energy_source_km',
        
        # Écologie native
        'npp_g_m2_yr', 'estimated_wood_density_g_cm3', 'fauna_herbivore_biomass_kg_km2', 'fauna_predator_biomass_kg_km2',
        
        # Marées, astronomie, vecteurs
        'tide_amplitude_m', 'vector_suitability_index',
        'photoperiod_summer_solstice_hours', 'photoperiod_winter_solstice_hours', 'photoperiod_range_hours'
    ] + dummy_cols
    
    # Exclure Pays_EN, Produit, ISO de X en gardant cat_cols vide
    cat_cols = []
    
    # 2. Split temporel (Train <= 2013 | Test > 2013) pour éviter le data leakage temporel
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
    model_path = os.path.join(MODEL_DIR, "best_model_pipeline.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Pipeline complet sauvegarde dans : {model_path}")
    
    # Sauvegarder un échantillon de test pour vérification future
    test_sample_path = os.path.join(DATA_DIR, "test_sample.csv")
    df_test.sample(min(1000, len(df_test)), random_state=42).to_csv(test_sample_path, index=False)
    print(f"Echantillon de test sauvegarde dans : {test_sample_path}")

if __name__ == "__main__":
    main()
