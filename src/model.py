import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def build_logistic_regression(random_state=42):
    return LogisticRegression(C=0.1, penalty='l2', max_iter=1000,
                              random_state=random_state, solver='liblinear')

def build_random_forest(random_state=42):
    return RandomForestClassifier(n_estimators=100, max_depth=10,
                                  min_samples_leaf=4, random_state=random_state, n_jobs=-1)

def build_xgboost(random_state=42):
    return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                         random_state=random_state, use_label_encoder=False,
                         eval_metric='logloss', n_jobs=-1)

def build_mlp(random_state=42):
    return MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                         solver='adam', max_iter=200, random_state=random_state,
                         early_stopping=True, n_iter_no_change=10)
