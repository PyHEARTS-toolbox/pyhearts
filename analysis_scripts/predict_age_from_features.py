#!/usr/bin/env python3
"""
Predict age from PyHEARTS extracted ECG features using QTDB dataset.

This script:
1. Extracts age information from QTDB header files
2. Aggregates PyHEARTS features per subject (mean, std, median across cycles)
3. Builds regression models to predict age
4. Evaluates model performance with cross-validation
5. Visualizes results and feature importance
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import json
import re

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Visualization
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style('whitegrid')
except ImportError:
    pass  # seaborn is optional

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
QTDB_DATA_DIR = PROJECT_ROOT / "data" / "qtdb" / "1.0.0"
RESULTS_DIR = PROJECT_ROOT / "results" / "qtdb_full_20260107_093822"

# Create output directory for age prediction results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "results" / f"age_prediction_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {OUTPUT_DIR}")


def extract_age_from_header(subject_name):
    """
    Extract age and sex from QTDB header file.
    
    Returns:
        tuple: (age, sex) or (None, None) if not found
    """
    header_file = QTDB_DATA_DIR / f"{subject_name}.hea"
    
    if not header_file.exists():
        return None, None
    
    try:
        with open(header_file, 'r') as f:
            lines = f.readlines()
        
        # Look for age in header comments
        for line in lines:
            line = line.strip()
            
            # Format 1: #Age: 43  Sex: M
            match = re.search(r'#Age:\s*(-?\d+)\s+Sex:\s*([MF-])', line)
            if match:
                age_str = match.group(1)
                sex = match.group(2) if match.group(2) != '-' else None
                try:
                    age = int(age_str)
                    if age > 0 and age < 150:  # Reasonable age range
                        return age, sex
                except ValueError:
                    pass
            
            # Format 2: # 69 M 1085 1629 x1 (first number is age)
            match = re.search(r'#\s*(-?\d+)\s+([MF-])\s', line)
            if match:
                age_str = match.group(1)
                sex = match.group(2) if match.group(2) != '-' else None
                try:
                    age = int(age_str)
                    if age > 0 and age < 150:  # Reasonable age range
                        return age, sex
                except ValueError:
                    pass
    
    except Exception as e:
        print(f"  Warning: Could not read {header_file}: {e}")
    
    return None, None


def aggregate_features_per_subject(subject_name):
    """
    Load PyHEARTS output CSV and aggregate features across cycles.
    
    Returns:
        pd.Series: Aggregated features for the subject
    """
    output_file = RESULTS_DIR / f"{subject_name}_output.csv"
    
    if not output_file.exists():
        return None
    
    try:
        df = pd.read_csv(output_file)
        
        if len(df) == 0:
            return None
        
        # Exclude non-feature columns
        exclude_cols = [
            'cycle_trend',  # Keep this as it's informative
            'R_global_center_idx', 'P_global_center_idx', 'Q_global_center_idx',
            'S_global_center_idx', 'T_global_center_idx',
            'P_global_le_idx', 'P_global_ri_idx', 'Q_global_le_idx', 'Q_global_ri_idx',
            'R_global_le_idx', 'R_global_ri_idx', 'S_global_le_idx', 'S_global_ri_idx',
            'T_global_le_idx', 'T_global_ri_idx',
            'P_fwhm_global_le_idx', 'P_fwhm_global_ri_idx', 'Q_fwhm_global_le_idx', 'Q_fwhm_global_ri_idx',
            'R_fwhm_global_le_idx', 'R_fwhm_global_ri_idx', 'S_fwhm_global_le_idx', 'S_fwhm_global_ri_idx',
            'T_fwhm_global_le_idx', 'T_fwhm_global_ri_idx',
            'P_center_idx', 'P_le_idx', 'P_ri_idx', 'Q_center_idx', 'Q_le_idx', 'Q_ri_idx',
            'R_center_idx', 'R_le_idx', 'R_ri_idx', 'S_center_idx', 'S_le_idx', 'S_ri_idx',
            'T_center_idx', 'T_le_idx', 'T_ri_idx',
            'P_fwhm_le_idx', 'P_fwhm_ri_idx', 'Q_fwhm_le_idx', 'Q_fwhm_ri_idx',
            'R_fwhm_le_idx', 'R_fwhm_ri_idx', 'S_fwhm_le_idx', 'S_fwhm_ri_idx',
            'T_fwhm_le_idx', 'T_fwhm_ri_idx',
            'P_gauss_center', 'Q_gauss_center', 'R_gauss_center', 'S_gauss_center', 'T_gauss_center',
        ]
        
        # Compute aggregations: mean, std, median, min, max
        aggregated = {}
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            values = df[col].dropna()
            
            if len(values) == 0:
                aggregated[f"{col}_mean"] = np.nan
                aggregated[f"{col}_std"] = np.nan
                aggregated[f"{col}_median"] = np.nan
                aggregated[f"{col}_min"] = np.nan
                aggregated[f"{col}_max"] = np.nan
                aggregated[f"{col}_count"] = 0
            else:
                aggregated[f"{col}_mean"] = values.mean()
                aggregated[f"{col}_std"] = values.std()
                aggregated[f"{col}_median"] = values.median()
                aggregated[f"{col}_min"] = values.min()
                aggregated[f"{col}_max"] = values.max()
                aggregated[f"{col}_count"] = len(values)
        
        # Add cycle-level stats
        aggregated['n_cycles'] = len(df)
        aggregated['r_squared_mean'] = df['r_squared'].mean() if 'r_squared' in df.columns else np.nan
        aggregated['r_squared_std'] = df['r_squared'].std() if 'r_squared' in df.columns else np.nan
        aggregated['rmse_mean'] = df['rmse'].mean() if 'rmse' in df.columns else np.nan
        
        return pd.Series(aggregated, name=subject_name)
    
    except Exception as e:
        print(f"  Warning: Could not load {output_file}: {e}")
        return None


def build_dataset():
    """
    Build dataset with age as target and aggregated features as predictors.
    
    Returns:
        pd.DataFrame: Dataset with age and features
    """
    print("\n" + "="*80)
    print("Building Dataset")
    print("="*80)
    
    # Get all subjects with output files
    output_files = list(RESULTS_DIR.glob("*_output.csv"))
    subjects = [f.stem.replace("_output", "") for f in output_files]
    
    print(f"Found {len(subjects)} subjects with output files")
    
    # Extract age and build feature matrix
    data_rows = []
    
    for subject in subjects:
        age, sex = extract_age_from_header(subject)
        
        if age is None:
            continue
        
        features = aggregate_features_per_subject(subject)
        
        if features is None:
            continue
        
        # Combine age, sex, and features
        row = features.copy()
        row['age'] = age
        row['sex'] = sex
        row['subject'] = subject
        
        data_rows.append(row)
    
    if len(data_rows) == 0:
        raise ValueError("No subjects with both age and features found!")
    
    df = pd.DataFrame(data_rows)
    df = df.set_index('subject')
    
    print(f"\nDataset created: {len(df)} subjects with age data")
    print(f"Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    print(f"Mean age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years")
    
    if 'sex' in df.columns:
        sex_counts = df['sex'].value_counts()
        print(f"Sex distribution: {dict(sex_counts)}")
    
    print(f"Total features: {len(df.columns) - 2}")  # Excluding age and sex
    
    return df


def prepare_features(df, use_feature_selection=True, n_features=50):
    """
    Prepare feature matrix and target vector with optional feature selection.
    
    Returns:
        X: Feature matrix
        y: Target vector (age)
        feature_names: List of feature names
    """
    # Separate target
    y = df['age'].values
    sex = df['sex'].values if 'sex' in df.columns else None
    
    # Get feature columns (exclude age, sex, subject)
    feature_cols = [col for col in df.columns if col not in ['age', 'sex', 'subject']]
    X = df[feature_cols].values
    
    # Handle missing values: fill with median
    X_df = pd.DataFrame(X, columns=feature_cols)
    X_df = X_df.fillna(X_df.median())
    X = X_df.values
    
    # Encode sex if available (add it first so it's included in selection)
    if sex is not None:
        sex_encoded = np.array([1 if s == 'M' else 0 if s == 'F' else 0.5 for s in sex])
        X = np.column_stack([X, sex_encoded])
        feature_cols = feature_cols + ['sex_M']
    
    # Feature selection to reduce overfitting
    if use_feature_selection and X.shape[1] > n_features:
        print(f"\nFeature selection: selecting top {n_features} features from {X.shape[1]}...")
        
        # Remove features with zero variance
        variance_threshold = 1e-8
        feature_variances = np.var(X, axis=0)
        valid_features = feature_variances > variance_threshold
        
        if np.sum(valid_features) < n_features:
            print(f"  Warning: Only {np.sum(valid_features)} features have non-zero variance")
            n_features = min(n_features, np.sum(valid_features))
        
        X_filtered = X[:, valid_features]
        feature_cols_filtered = [f for f, v in zip(feature_cols, valid_features) if v]
        
        # Use f-test to select top features
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_selected = selector.fit_transform(X_filtered, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        feature_cols = [feature_cols_filtered[i] for i in selected_indices]
        
        print(f"  Selected {len(feature_cols)} features with highest f-scores")
        
        X = X_selected
    else:
        print(f"\nUsing all {X.shape[1]} features (no selection)")
    
    return X, y, feature_cols


def evaluate_models(X, y, feature_names):
    """
    Evaluate multiple regression models with cross-validation.
    
    Returns:
        dict: Results for each model
    """
    print("\n" + "="*80)
    print("Model Evaluation")
    print("="*80)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Define models with more regularization for small sample size
    models = {
        'Ridge': Ridge(alpha=10.0),  # Increased regularization
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000),
        'RandomForest': RandomForestRegressor(
            n_estimators=50,  # Reduced to prevent overfitting
            max_depth=5,  # Shallow trees
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=50,  # Reduced
            max_depth=3,  # Shallow trees
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        ),
    }
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Cross-validation scores
        cv_scores_mae = -cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
        cv_scores_r2 = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        
        # Train on full training set and evaluate on test set
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
        
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'cv_mae_mean': cv_scores_mae.mean(),
            'cv_mae_std': cv_scores_mae.std(),
            'cv_r2_mean': cv_scores_r2.mean(),
            'cv_r2_std': cv_scores_r2.std(),
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test,
            'y_test': y_test,
        }
        
        print(f"  CV MAE: {cv_scores_mae.mean():.2f} ± {cv_scores_mae.std():.2f} years")
        print(f"  CV R²:  {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")
        print(f"  Test MAE: {test_mae:.2f} years")
        print(f"  Test RMSE: {test_rmse:.2f} years")
        print(f"  Test R²:  {test_r2:.3f}")
    
    return results, scaler


def analyze_feature_importance(model, feature_names, X_scaled, y, top_n=20):
    """
    Analyze feature importance for tree-based models.
    
    Returns:
        pd.DataFrame: Feature importance rankings
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        importances = np.abs(model.coef_)
    else:
        return None
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def visualize_results(results, importance_df, output_dir):
    """
    Create visualizations of model performance and feature importance.
    """
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    # 1. Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Age Prediction Model Performance', fontsize=16, fontweight='bold')
    
    # CV MAE comparison
    model_names = list(results.keys())
    cv_mae_means = [results[m]['cv_mae_mean'] for m in model_names]
    cv_mae_stds = [results[m]['cv_mae_std'] for m in model_names]
    
    axes[0, 0].bar(model_names, cv_mae_means, yerr=cv_mae_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_ylabel('Mean Absolute Error (years)')
    axes[0, 0].set_title('Cross-Validation MAE')
    axes[0, 0].grid(axis='y', alpha=0.3)
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # CV R² comparison
    cv_r2_means = [results[m]['cv_r2_mean'] for m in model_names]
    cv_r2_stds = [results[m]['cv_r2_std'] for m in model_names]
    
    axes[0, 1].bar(model_names, cv_r2_means, yerr=cv_r2_stds, capsize=5, alpha=0.7, color='green')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('Cross-Validation R²')
    axes[0, 1].grid(axis='y', alpha=0.3)
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Test set predictions vs actual (best model)
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
    best_result = results[best_model_name]
    
    axes[1, 0].scatter(best_result['y_test'], best_result['y_pred_test'], alpha=0.6, s=50)
    axes[1, 0].plot([best_result['y_test'].min(), best_result['y_test'].max()], 
                    [best_result['y_test'].min(), best_result['y_test'].max()], 
                    'r--', lw=2, label='Perfect prediction')
    axes[1, 0].set_xlabel('Actual Age (years)')
    axes[1, 0].set_ylabel('Predicted Age (years)')
    axes[1, 0].set_title(f'Test Set Predictions ({best_model_name})')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Add text with metrics
    mae_text = f"MAE: {best_result['test_mae']:.2f} years\n"
    mae_text += f"RMSE: {best_result['test_rmse']:.2f} years\n"
    mae_text += f"R²: {best_result['test_r2']:.3f}"
    axes[1, 0].text(0.05, 0.95, mae_text, transform=axes[1, 0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Feature importance (if available)
    if importance_df is not None and len(importance_df) > 0:
        top_features = importance_df.head(15)
        axes[1, 1].barh(range(len(top_features)), top_features['importance'].values[::-1])
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features['feature'].values[::-1], fontsize=8)
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top 15 Features (Importance)')
        axes[1, 1].grid(axis='x', alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'age_prediction_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: age_prediction_performance.png")
    plt.close()
    
    # 2. Residuals plot
    fig, ax = plt.subplots(figsize=(10, 6))
    residuals = best_result['y_test'] - best_result['y_pred_test']
    ax.scatter(best_result['y_pred_test'], residuals, alpha=0.6, s=50)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Age (years)')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.set_title(f'Residuals Plot ({best_model_name})')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'age_prediction_residuals.png', dpi=300, bbox_inches='tight')
    print(f"Saved: age_prediction_residuals.png")
    plt.close()


def main():
    print("="*80)
    print("Age Prediction from PyHEARTS Features")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Build dataset
    df = build_dataset()
    
    # Save dataset
    dataset_file = OUTPUT_DIR / "dataset_with_age.csv"
    df.to_csv(dataset_file)
    print(f"\nSaved dataset to: {dataset_file}")
    
    # Prepare features with selection for small sample size
    # With only 46 subjects, we need to limit features to avoid overfitting
    X, y, feature_names = prepare_features(df, use_feature_selection=True, n_features=30)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Evaluate models
    results, scaler = evaluate_models(X, y, feature_names)
    
    # Get best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*80}")
    print(f"Best Model: {best_model_name}")
    print(f"{'='*80}")
    print(f"Test MAE: {results[best_model_name]['test_mae']:.2f} years")
    print(f"Test RMSE: {results[best_model_name]['test_rmse']:.2f} years")
    print(f"Test R²: {results[best_model_name]['test_r2']:.3f}")
    
    # Feature importance
    X_scaled = scaler.transform(X)
    importance_df = analyze_feature_importance(best_model, feature_names, X_scaled, y, top_n=30)
    
    if importance_df is not None:
        importance_file = OUTPUT_DIR / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        print(f"\nSaved feature importance to: {importance_file}")
        print("\nTop 10 most important features:")
        print(importance_df.head(10).to_string(index=False))
    
    # Visualizations
    visualize_results(results, importance_df, OUTPUT_DIR)
    
    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_subjects': len(df),
        'age_range': [float(df['age'].min()), float(df['age'].max())],
        'age_mean': float(df['age'].mean()),
        'age_std': float(df['age'].std()),
        'n_features': len(feature_names),
        'best_model': best_model_name,
        'best_test_mae': float(results[best_model_name]['test_mae']),
        'best_test_rmse': float(results[best_model_name]['test_rmse']),
        'best_test_r2': float(results[best_model_name]['test_r2']),
        'model_results': {
            name: {
                'cv_mae_mean': float(results[name]['cv_mae_mean']),
                'cv_mae_std': float(results[name]['cv_mae_std']),
                'cv_r2_mean': float(results[name]['cv_r2_mean']),
                'cv_r2_std': float(results[name]['cv_r2_std']),
                'test_mae': float(results[name]['test_mae']),
                'test_rmse': float(results[name]['test_rmse']),
                'test_r2': float(results[name]['test_r2']),
            }
            for name in results.keys()
        }
    }
    
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_file}")
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

