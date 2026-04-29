"""
dataset preparing and unpacking
"""

import pandas as pd
from sklearn.model_selection import train_test_split

#FILENAME = "datasets/heart.csv"
FILENAME = "datasets/diabetes.csv"


df = pd.read_csv(FILENAME)

target_name = df.columns[-1]
df = df.drop_duplicates()
features = df.columns
feature_types = dict(df.dtypes)
# targer variable always binary type (0 or 1)
y = df[target_name]
X = df.drop(target_name, axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

# print(X_train)
# print(df)
