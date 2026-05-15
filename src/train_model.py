import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from data_preprocessing import load_data, prepare_data, get_preprocessor
from evaluate_model import evaluate_predictions, plot_confusion_matrix, plot_eda, plot_feature_importance
from utils import save_model, ensure_dir

# Configuration paths
DATA_PATH = os.path.join('data', 'telco_customer_churn.csv')
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'

def main():
    print("Starting Machine Learning Pipeline...")
    ensure_dir(MODEL_DIR)
    ensure_dir(OUTPUT_DIR)
    
    # 1. Data Loading & Preprocessing
    print("Loading data...")
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return
        
    print("Generating EDA visualizations...")
    plot_eda(df, OUTPUT_DIR)
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    preprocessor, _, _ = get_preprocessor(X_train)
    
    # Save preprocessor for deployment
    save_model(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    
    # 2. Model Definition
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = []
    best_model_name = None
    best_f1 = 0
    best_model_pipeline = None
    best_y_pred = None
    
    # 3. Model Training and Evaluation
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline to prevent data leakage
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
            
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics['Model'] = name
        results.append(metrics)
        
        # Determine best model based on F1-score
        if metrics['F1-score'] > best_f1:
            best_f1 = metrics['F1-score']
            best_model_name = name
            best_model_pipeline = pipeline
            best_y_pred = y_pred

    # 4. Save results and best model
    print(f"\nBest Model: {best_model_name} with F1-score: {best_f1:.4f}")
    
    results_df = pd.DataFrame(results)
    # Reorder columns
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    results_df = results_df[cols]
    
    results_df.to_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'), index=False)
    save_model(best_model_pipeline, os.path.join(MODEL_DIR, 'best_model.pkl'))
    
    print("Generating visualizations for the best model...")
    plot_confusion_matrix(y_test, best_y_pred, OUTPUT_DIR)
    
    best_clf = best_model_pipeline.named_steps['classifier']
    plot_feature_importance(best_clf, preprocessor, OUTPUT_DIR)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
