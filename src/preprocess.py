import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Fix TotalCharges — it's a string column with spaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Drop customerID — not useful for modeling
    df.drop('customerID', axis=1, inplace=True)

    # Engineer new features
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-12', '12-24', '24-48', '48-60', '60-72']
    )
    df['charges_per_month_ratio'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode all categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, y_train, y_test, X.columns.tolist()