import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
from utils import ensure_dir

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """Calculate evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred)
    }
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['ROC-AUC'] = np.nan
    return metrics

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot and save confusion matrix."""
    ensure_dir(output_dir)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_eda(df, output_dir):
    """Generate and save EDA plots."""
    ensure_dir(output_dir)
    
    # 1. Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Churn Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'churn_distribution.png'))
    plt.close()
    
    # 2. Churn by Contract
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
    plt.title('Churn by Contract Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contract_churn.png'))
    plt.close()
    
    # 3. Monthly Charges vs Churn
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, palette='Set2', bins=30)
    plt.title('Monthly Charges Distribution by Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_charges_churn.png'))
    plt.close()
    
    # 4. Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, palette='Set2', bins=30)
    plt.title('Tenure Distribution by Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tenure_churn.png'))
    plt.close()

def plot_feature_importance(model, preprocessor, output_dir):
    """Plot and save feature importance if the model supports it."""
    ensure_dir(output_dir)
    try:
        importances = None
        # Check if the model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            
        if importances is not None:
            # Extract feature names from ColumnTransformer
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = cat_encoder.get_feature_names_out()
            num_features = preprocessor.named_transformers_['num'].named_steps['scaler'].feature_names_in_
            
            all_features = list(num_features) + list(cat_features)
            
            feature_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
            feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_imp_df, x='Importance', y='Feature', palette='viridis')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()
        else:
            print("Model does not support feature_importances_ or coef_")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
