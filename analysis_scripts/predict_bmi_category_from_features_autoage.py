#!/usr/bin/env python3
"""
Predict BMI category (health proxy) from PyHEARTS extracted ECG features using Autoage dataset.

This script:
1. Extracts BMI information from Autoage subject info CSV file
2. Creates BMI categories (Normal, Overweight, Obese) as health proxy
3. Aggregates PyHEARTS features per subject (mean, std, median across cycles)
4. Builds classification models to predict BMI category
5. Evaluates model performance with cross-validation
6. Visualizes results and feature importance

Note: The autoage dataset does not contain explicit health/disease labels.
BMI category is used as a proxy for health status.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import json

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif

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
AUTOAGE_DATA_DIR = PROJECT_ROOT / "data" / "autoage"
SUBJECT_INFO_FILE = AUTOAGE_DATA_DIR / "subject_info_clean.csv"

# Find the latest autoage results directory
autoage_results_dirs = sorted(PROJECT_ROOT.glob("results/autoage_*"))
if not autoage_results_dirs:
    raise ValueError("No autoage results directory found! Please run process_autoage.py first.")
RESULTS_DIR = autoage_results_dirs[-1]

# Create output directory for BMI prediction results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "results" / f"bmi_prediction_autoage_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {OUTPUT_DIR}")
print(f"Using PyHEARTS results from: {RESULTS_DIR}")


def extract_bmi_from_subject_info():
    """
    Extract BMI and create BMI category labels from Autoage subject info CSV.
    
    BMI Categories:
    - Normal: 18.5 <= BMI < 25
    - Overweight: 25 <= BMI < 30
    - Obese: BMI >= 30
    - Underweight: BMI < 18.5 (excluded or merged with Normal)
    
    Returns:
        dict: Mapping from subject_id (4-digit string) to (bmi, bmi_category) tuple
    """
    if not SUBJECT_INFO_FILE.exists():
        raise FileNotFoundError(f"Subject info file not found: {SUBJECT_INFO_FILE}")
    
    try:
        df = pd.read_csv(SUBJECT_INFO_FILE)
        
        # Create mapping: subject_id -> (bmi, bmi_category)
        bmi_map = {}
        
        for _, row in df.iterrows():
            subject_id = str(int(row['ID'])).zfill(4)  # Format as 4-digit string (e.g., "0001")
            
            # Get BMI
            bmi = row.get('BMI', None)
            if pd.isna(bmi) or bmi <= 0:
                continue
            
            # Categorize BMI
            if bmi < 18.5:
                category = 'Underweight'
            elif bmi < 25:
                category = 'Normal'
            elif bmi < 30:
                category = 'Overweight'
            else:
                category = 'Obese'
            
            bmi_map[subject_id] = (bmi, category)
        
        return bmi_map
    
    except Exception as e:
        print(f"  ERROR: Could not read subject info file: {e}")
        raise


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
    Build dataset with BMI category as target and aggregated features as predictors.
    
    Returns:
        pd.DataFrame: Dataset with BMI, BMI category, and features
    """
    print("\n" + "="*80)
    print("Building Dataset")
    print("="*80)
    
    # Load BMI mapping from subject info
    print("Loading subject BMI information...")
    bmi_map = extract_bmi_from_subject_info()
    print(f"  Loaded BMI info for {len(bmi_map)} subjects")
    
    # Get all subjects with output files
    output_files = list(RESULTS_DIR.glob("*_output.csv"))
    subjects = [f.stem.replace("_output", "") for f in output_files]
    
    print(f"Found {len(subjects)} subjects with output files")
    
    # Extract BMI and build feature matrix
    data_rows = []
    
    for subject in subjects:
        bmi_info = bmi_map.get(subject)
        
        if bmi_info is None:
            continue
        
        bmi, bmi_category = bmi_info
        
        features = aggregate_features_per_subject(subject)
        
        if features is None:
            continue
        
        # Combine BMI, BMI category, and features
        row = features.copy()
        row['bmi'] = bmi
        row['bmi_category'] = bmi_category
        row['subject'] = subject
        
        data_rows.append(row)
    
    if len(data_rows) == 0:
        raise ValueError("No subjects with both BMI and features found!")
    
    df = pd.DataFrame(data_rows)
    df = df.set_index('subject')
    
    print(f"\nDataset created: {len(df)} subjects with BMI data")
    print(f"\nBMI Statistics:")
    print(f"  Mean: {df['bmi'].mean():.2f} ± {df['bmi'].std():.2f}")
    print(f"  Range: {df['bmi'].min():.2f} - {df['bmi'].max():.2f}")
    
    print(f"\nBMI Category Distribution:")
    category_counts = df['bmi_category'].value_counts()
    for category, count in category_counts.items():
        pct = 100 * count / len(df)
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    print(f"Total features: {len(df.columns) - 2}")  # Excluding bmi and bmi_category
    
    return df


def prepare_features(df, use_feature_selection=True, n_features=100):
    """
    Prepare feature matrix and target vector with optional feature selection.
    
    Returns:
        X: Feature matrix
        y: Target vector (BMI category)
        feature_names: List of feature names
        label_encoder: Mapping from category to integer
    """
    # Separate target
    y_categorical = df['bmi_category'].values
    
    # Create label encoder
    unique_categories = sorted(df['bmi_category'].unique())
    label_to_int = {label: idx for idx, label in enumerate(unique_categories)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y = np.array([label_to_int[cat] for cat in y_categorical])
    
    # Get feature columns (exclude bmi, bmi_category, subject)
    feature_cols = [col for col in df.columns if col not in ['bmi', 'bmi_category', 'subject']]
    X = df[feature_cols].values
    
    # Handle missing values: fill with median
    X_df = pd.DataFrame(X, columns=feature_cols)
    X_df = X_df.fillna(X_df.median())
    X = X_df.values
    
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
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X_filtered, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        feature_cols = [feature_cols_filtered[i] for i in selected_indices]
        
        print(f"  Selected {len(feature_cols)} features with highest f-scores")
        
        X = X_selected
    else:
        print(f"\nUsing all {X.shape[1]} features (no selection)")
    
    return X, y, feature_cols, label_to_int, int_to_label


def evaluate_models(X, y, feature_names, int_to_label):
    """
    Evaluate multiple classification models with cross-validation.
    
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
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=2000, 
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        ),
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        # Train on full training set and evaluate on test set
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Get class names for reporting
        y_test_labels = [int_to_label[yi] for yi in y_test]
        y_pred_test_labels = [int_to_label[yi] for yi in y_pred_test]
        
        test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        results[name] = {
            'model': model,
            'cv_acc_mean': cv_scores.mean(),
            'cv_acc_std': cv_scores.std(),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'y_pred_test_labels': y_pred_test_labels,
            'y_test_labels': y_test_labels,
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        }
        
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")
        print(f"  Test Precision: {test_precision:.3f}")
        print(f"  Test Recall: {test_recall:.3f}")
        print(f"  Test F1: {test_f1:.3f}")
    
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
        # For linear models, use absolute coefficients (mean across classes)
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return None
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def visualize_results(results, importance_df, output_dir, int_to_label):
    """
    Create visualizations of model performance and feature importance.
    """
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    # 1. Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BMI Category Prediction Model Performance (Autoage Dataset)', fontsize=16, fontweight='bold')
    
    # CV Accuracy comparison
    model_names = list(results.keys())
    cv_acc_means = [results[m]['cv_acc_mean'] for m in model_names]
    cv_acc_stds = [results[m]['cv_acc_std'] for m in model_names]
    
    axes[0, 0].bar(model_names, cv_acc_means, yerr=cv_acc_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Cross-Validation Accuracy')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Test Accuracy comparison
    test_accs = [results[m]['test_acc'] for m in model_names]
    
    axes[0, 1].bar(model_names, test_accs, alpha=0.7, color='green')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Test Set Accuracy')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Confusion matrix (best model)
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_acc'])
    best_result = results[best_model_name]
    
    cm = best_result['confusion_matrix']
    class_names = [int_to_label[i] for i in sorted(int_to_label.keys())]
    
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].figure.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_names, yticklabels=class_names,
                   title=f'Confusion Matrix ({best_model_name})',
                   ylabel='True Label',
                   xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
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
    plt.savefig(output_dir / 'bmi_prediction_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: bmi_prediction_performance.png")
    plt.close()
    
    # 2. Classification report visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    best_result = results[best_model_name]
    
    # Create classification report
    from sklearn.metrics import classification_report
    report = classification_report(
        best_result['y_test'],
        best_result['y_pred_test'],
        target_names=class_names,
        output_dict=True
    )
    
    # Extract metrics for each class
    classes = []
    precisions = []
    recalls = []
    f1s = []
    
    for class_name in class_names:
        if class_name in report:
            classes.append(class_name)
            precisions.append(report[class_name]['precision'])
            recalls.append(report[class_name]['recall'])
            f1s.append(report[class_name]['f1-score'])
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.7)
    ax.bar(x, recalls, width, label='Recall', alpha=0.7)
    ax.bar(x + width, f1s, width, label='F1-Score', alpha=0.7)
    
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Performance ({best_model_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bmi_prediction_class_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: bmi_prediction_class_performance.png")
    plt.close()


def main():
    print("="*80)
    print("BMI Category Prediction from PyHEARTS Features (Autoage Dataset)")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nNote: Using BMI category as a proxy for health status.")
    print("The autoage dataset does not contain explicit health/disease labels.")
    
    # Build dataset
    df = build_dataset()
    
    # Save dataset
    dataset_file = OUTPUT_DIR / "dataset_with_bmi.csv"
    df.to_csv(dataset_file)
    print(f"\nSaved dataset to: {dataset_file}")
    
    # Prepare features
    X, y, feature_names, label_to_int, int_to_label = prepare_features(
        df, use_feature_selection=True, n_features=100
    )
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Classes: {list(int_to_label.values())}")
    
    # Evaluate models
    results, scaler = evaluate_models(X, y, feature_names, int_to_label)
    
    # Get best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_acc'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*80}")
    print(f"Best Model: {best_model_name}")
    print(f"{'='*80}")
    print(f"Test Accuracy: {results[best_model_name]['test_acc']:.3f}")
    print(f"Test Precision: {results[best_model_name]['test_precision']:.3f}")
    print(f"Test Recall: {results[best_model_name]['test_recall']:.3f}")
    print(f"Test F1: {results[best_model_name]['test_f1']:.3f}")
    
    # Print classification report
    print(f"\nClassification Report:")
    from sklearn.metrics import classification_report
    print(classification_report(
        results[best_model_name]['y_test'],
        results[best_model_name]['y_pred_test'],
        target_names=[int_to_label[i] for i in sorted(int_to_label.keys())]
    ))
    
    # Feature importance
    X_scaled = scaler.transform(X)
    importance_df = analyze_feature_importance(
        best_model, feature_names, X_scaled, y, top_n=50
    )
    
    if importance_df is not None:
        importance_file = OUTPUT_DIR / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        print(f"\nSaved feature importance to: {importance_file}")
        print("\nTop 10 most important features:")
        print(importance_df.head(10).to_string(index=False))
    
    # Visualizations
    visualize_results(results, importance_df, OUTPUT_DIR, int_to_label)
    
    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'autoage',
        'target': 'bmi_category',
        'note': 'BMI category used as proxy for health status (autoage dataset has no explicit health/disease labels)',
        'n_subjects': len(df),
        'bmi_mean': float(df['bmi'].mean()),
        'bmi_std': float(df['bmi'].std()),
        'bmi_range': [float(df['bmi'].min()), float(df['bmi'].max())],
        'category_distribution': df['bmi_category'].value_counts().to_dict(),
        'n_features': len(feature_names),
        'best_model': best_model_name,
        'best_test_acc': float(results[best_model_name]['test_acc']),
        'best_test_precision': float(results[best_model_name]['test_precision']),
        'best_test_recall': float(results[best_model_name]['test_recall']),
        'best_test_f1': float(results[best_model_name]['test_f1']),
        'model_results': {
            name: {
                'cv_acc_mean': float(results[name]['cv_acc_mean']),
                'cv_acc_std': float(results[name]['cv_acc_std']),
                'test_acc': float(results[name]['test_acc']),
                'test_precision': float(results[name]['test_precision']),
                'test_recall': float(results[name]['test_recall']),
                'test_f1': float(results[name]['test_f1']),
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
    print("\nNote: This analysis uses BMI category as a proxy for health status.")
    print("For true health/disease prediction, a dataset with explicit health labels would be needed.")


if __name__ == "__main__":
    main()

