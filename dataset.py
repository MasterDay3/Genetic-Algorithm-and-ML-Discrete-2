"""
dataset preparing and unpacking
"""

import pandas as pd
from sklearn.model_selection import train_test_split


FILENAME = "heart.csv"


df = pd.read_csv(FILENAME)
features = df.columns
feature_types = dict(df.dtypes)
# targer variable always binary type (0 or 1)
y = df["target"]
X = df.drop("target", axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

print(X_train)
