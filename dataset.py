"""
dataset preparing and unpacking
"""

import pandas as pd
from sklearn.model_selection import train_test_split

#FILENAME = "datasets/heart.csv"
#FILENAME = "datasets/diabetes.csv"
FILENAME = 'datasets/hospital_readmission.csv'

df = pd.read_csv(FILENAME)

# only for hospital_readmission
if FILENAME == 'datasets/hospital_readmission.csv':
    df = df.drop(columns=['patient_id', 'admission_date'])
    df = pd.get_dummies(df, columns=["season"])
    df = pd.get_dummies(df, columns=['gender'])
    df = pd.get_dummies(df, columns=["region"])
    df = pd.get_dummies(df, columns=["primary_diagnosis"])
    df = pd.get_dummies(df, columns=["treatment_type"])
    df = pd.get_dummies(df, columns=["insurance_type"], drop_first=True)
    df = pd.get_dummies(df, columns=["discharge_disposition"])
    df = df.drop(columns='readmission_risk_score')
    #print(df["primary_diagnosis"].nunique())
    # print(df["treatment_type"].nunique())
    # print(df["treatment_type"].unique())
    # print(df["discharge_disposition"].nunique())
    # print(df["discharge_disposition"].value_counts().head(20))
    # print(df["gender"].value_counts(dropna=False))
    # print(df[df["gender"].isna()].head())
    target_name = "label"
    y = df[target_name]
    X = df.drop(columns=[target_name])


#target_name = df.columns[-1]
df = df.drop_duplicates()
features = df.columns
feature_types = dict(df.dtypes)
# targer variable always binary type (0 or 1)
#y = df[target_name]
#X = df.drop(target_name, axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)


X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
# print(X_train.isna().sum().sort_values(ascending=False).head(20))

# print(X_train)
# print(df)
