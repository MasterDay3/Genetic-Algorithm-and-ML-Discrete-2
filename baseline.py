from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from dataset import X_train, X_test, y_train, y_test, X, y, FILENAME


def evaluate_baseline(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    roc_auc = roc_auc_score(y_te, y_prob)

    return acc, f1, roc_auc


log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


print(f"\033[32mStats for {FILENAME}\033[32m")

acc_lr, f1_lr, auc_lr = evaluate_baseline(log_reg, X_train, y_train, X_test, y_test)
print(
    f"\033[32mLogistic Regression -> Accuracy: {acc_lr:.3f}, F1: {f1_lr:.3f}, ROC-AUC: {auc_lr:.3f}\033[32m"
)

acc_rf, f1_rf, auc_rf = evaluate_baseline(rf_model, X_train, y_train, X_test, y_test)
print(
    f"\033[32mRandom Forest       -> Accuracy: {acc_rf:.3f}, F1: {f1_rf:.3f}, ROC-AUC: {auc_rf:.3f}\033[32m"
)


def get_fitness_score(selected_features_mask):
    """
    selected_features_mask: список або масив булевих значень (True/False) для кожної фічі
    """
    X_subset = X.iloc[:, selected_features_mask]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(model, X_subset, y, cv=3, scoring="f1")

    return scores.mean()
