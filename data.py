"""
dataset preparing and unpacking
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

scaler = StandardScaler()
# FILENAME = "datasets/heart.csv"
# FILENAME = "datasets/diabetes.csv"
FILENAME = "datasets/hospital_readmission.csv"
# FILENAME = 'datasets/working_hours.csv'


print(f"Завантаження {FILENAME}")
df = pd.read_csv(FILENAME)
# if big dataset
# df = pd.read_csv(FILENAME).sample(n=10000, random_state=42)


def convert_to_numeric(df):
    df = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.isna().sum() <= df[col].isna().sum():
                df[col] = converted
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.median(numeric_only=True))

    return df


df = df.drop_duplicates()
print(f"Датасет: {df.shape}")


if FILENAME == "datasets/hospital_readmission.csv":
    df = df.drop(columns="readmission_risk_score")


df = convert_to_numeric(df)
df = df.dropna()
target_name = df.columns[-1]
y = df[target_name]
X = df.drop(columns=[target_name])
X = pd.get_dummies(X, drop_first=True)

df = df.drop_duplicates()
features = df.columns
feature_types = dict(df.dtypes)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)
