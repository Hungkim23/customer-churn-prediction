import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath):
    """Load the dataset and handle basic cleaning."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please place 'telco_customer_churn.csv' in the 'data/' directory.")

    # Convert TotalCharges to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Drop customerID as it's not predictive
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # Convert Churn target to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def get_preprocessor(X):
    """Create a scikit-learn ColumnTransformer for preprocessing."""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, numeric_features, categorical_features

def prepare_data(df):
    """Split data into features and target, and train/test sets."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test
