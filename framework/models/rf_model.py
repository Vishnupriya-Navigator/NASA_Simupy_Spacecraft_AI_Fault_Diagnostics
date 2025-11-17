# framework/models/rf_model.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from framework.config import Config


def train(df: pd.DataFrame, cfg: Config = Config()):
    X = df[list(cfg.feature_names)]
    y = df[cfg.label_name]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=cfg.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)

    return clf, report, cm


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
