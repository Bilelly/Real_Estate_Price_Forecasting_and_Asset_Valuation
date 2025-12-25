from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np

def evaluate_with_cross_val(models_dict, X, y, cv=5):
    """
    Effectue une validation croisée pour une liste de modèles et affiche les scores.
    
    Args:
        models_dict (dict): Dictionnaire de noms de modèles et leurs pipelines respectifs.
        X: Features (données d'entraînement).
        y: Target (variable cible).
        cv (int): Nombre de plis (folds) pour la validation croisée.
    """
    results = []
    
    print(f"=== Validation Croisée (K={cv}) ===")
    
    # Définition des métriques à calculer
    scoring = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    for name, model in models_dict.items():
        # Exécution de la validation croisée
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Extraction des scores (on inverse le signe pour RMSE et MAE car sklearn les renvoie en négatif)
        mean_rmse = -cv_results['test_rmse'].mean()
        std_rmse = cv_results['test_rmse'].std()
        mean_mae = -cv_results['test_mae'].mean()
        mean_r2 = cv_results['test_r2'].mean()
        
        results.append({
            'Modèle': name,
            'RMSE Moyenne': mean_rmse,
            'RMSE Std': std_rmse,
            'MAE Moyenne': mean_mae,
            'R² Moyen': mean_r2
        })
        
        print(f"Terminé pour : {name}")

    # Transformation en DataFrame pour une lecture facile
    df_results = pd.DataFrame(results).sort_values(by='RMSE Moyenne')
    return df_results