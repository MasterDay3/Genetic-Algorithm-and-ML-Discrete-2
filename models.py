from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DEFAULT_SCORING = "roc_auc"
# models.py

DEFAULT_MODEL = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# DEFAULT_MODEL = LogisticRegression(
#     max_iter=8000,
#     random_state=67,
#     solver="saga",
# )


def evaluate_baseline(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    return acc, f1, auc


log_reg = DEFAULT_MODEL
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# print(f"\033[32mStats for {FILENAME}\033[32m")

# acc_lr, f1_lr, auc_lr = evaluate_baseline(log_reg, X_train, y_train, X_test, y_test)
# print(
#     f"\033[32mLogistic Regression -> Accuracy: {acc_lr:.3f}, F1: {f1_lr:.3f}, ROC-AUC: {auc_lr:.3f}\033[32m"
# )

# acc_rf, f1_rf, auc_rf = evaluate_baseline(rf_model, X_train, y_train, X_test, y_test)
# print(
#     f"\033[32mRandom Forest       -> Accuracy: {acc_rf:.3f}, F1: {f1_rf:.3f}, ROC-AUC: {auc_rf:.3f}\033[32m"
# )


def fitness_function(
    chromosome: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    penalty: float = 0.05,
    cv: int = 5,
    scoring: str = DEFAULT_SCORING,
) -> float:
    selected_indices = np.where(chromosome == 1)[0]

    if len(selected_indices) == 0:
        return 0.0

    X_subset = X_train[:, selected_indices]
    scores = cross_val_score(model, X_subset, y_train, cv=cv, scoring=scoring)
    feature_ratio = len(selected_indices) / len(chromosome)

    return scores.mean() - penalty * feature_ratio


# def get_fitness_score(selected_features_mask):
#     """
#     selected_features_mask: список або масив булевих значень (True/False) для кожної фічі
#     """
#     X_subset = X.iloc[:, selected_features_mask]

#     # для вектору з нулів
#     if X_subset.shape[1] == 0:
#         return 0

#     # Можливо, алгоритм буде дуже довго працювати саме за рахунок RandomForest,
#     # можливо, варто спробувати Лінійну регресію або інший алгоритм,
#     # якщо з RandomForest буде довго виконуватись
#     model = RandomForestClassifier(n_estimators=50, random_state=42)
#     scores = cross_val_score(model, X_subset, y, cv=3, scoring="f1")

#     # штраф, якщо беремо багато ознак, число 0.1 є магічним, можливо варто буде змінити
#     # (перевірити експерементально) та встановити найкраще значення
#     penalty = sum(selected_features_mask) / len(selected_features_mask)

#     # не впевнений стосовно штрафу, поки закоменчений
#     return scores.mean()  - 0.05 * penalty


# mask = [False, True, True, True, True, True, True, True, True, False, True, True, True]
# print(mask)
# print("w mask", get_fitness_score(mask))
