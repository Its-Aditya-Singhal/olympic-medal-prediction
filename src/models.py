"""Model zoo for the two-stage approach (paper Sections 4.2, 4.3, 5.3, 5.4).

Stage 1: a classifier predicts whether a country wins any medal.
Stage 2: a regressor, trained only on medal-winning rows, predicts how many.
"""
import numpy as np
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, PoissonRegressor, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from config import RANDOM_STATE


class WeightedLinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression with sample weights w = exp(-(y_i - y0)^2 / (2 tau^2)) (paper Eq. 4).

    Examples whose medal count is close to y0 get the most weight, which lets
    the fit favour the high-scoring countries that Eq. 2 cares about.
    """

    def __init__(self, y0: float = 50.0, tau: float = 25.0):
        self.y0 = y0
        self.tau = tau

    def fit(self, X, y):
        y = np.asarray(y, float)
        w = np.exp(-((y - self.y0) ** 2) / (2 * self.tau ** 2))
        self.model_ = LinearRegression().fit(X, y, sample_weight=np.maximum(w, 1e-6))
        return self

    def predict(self, X):
        return self.model_.predict(X)


def _pipe(estimator) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _grid(params: dict) -> dict:
    return {f"model__{k}": v for k, v in params.items()}


def _grids(grids: list) -> list:
    return [_grid(g) for g in grids]


# name -> (display name, pipeline, search space, "grid" or "random")
CLASSIFIERS = {
    "logreg": ("Logistic Regression", _pipe(LogisticRegression(max_iter=5000)),
               _grid({"C": [0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"]}), "grid"),
    "svc": ("SVM Classifier", _pipe(SVC()),
            _grids([{"kernel": ["linear"], "C": [0.1, 1, 10]},
                    {"kernel": ["rbf", "sigmoid"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]},
                    {"kernel": ["poly"], "degree": [2, 3], "C": [0.1, 1, 10]}]), "grid"),
    "gnb": ("Gaussian Naive Bayes", _pipe(GaussianNB()),
            _grid({"var_smoothing": np.logspace(-12, -1, 12)}), "grid"),
    "mlp": ("MLP Classifier", _pipe(MLPClassifier(max_iter=3000, random_state=RANDOM_STATE)),
            _grid({"hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
                   "activation": ["relu", "tanh", "logistic"],
                   "alpha": loguniform(1e-5, 1e-1), "learning_rate_init": loguniform(1e-4, 1e-2)}), "random"),
    "rf": ("Random Forest Classifier", _pipe(RandomForestClassifier(random_state=RANDOM_STATE)),
           _grid({"n_estimators": [100, 200, 400], "max_depth": [None, 4, 8, 12, 16],
                  "min_samples_leaf": [1, 2, 4, 8], "max_features": ["sqrt", 0.5, 1.0]}), "random"),
}

REGRESSORS = {
    "linear": ("Linear Regression", _pipe(LinearRegression()),
               _grid({"fit_intercept": [True, False]}), "grid"),
    "weighted_lr": ("Weighted Linear Regression", _pipe(WeightedLinearRegression()),
                    _grid({"y0": [10, 25, 50, 75, 100], "tau": [1, 5, 10, 25, 50, 100]}), "grid"),
    "ridge": ("Ridge", _pipe(Ridge()), _grid({"alpha": np.logspace(-3, 3, 13)}), "grid"),
    "lasso": ("Lasso", _pipe(Lasso(max_iter=100000)), _grid({"alpha": np.logspace(-3, 1, 9)}), "grid"),
    "poisson": ("Poisson Regression", _pipe(PoissonRegressor(max_iter=10000)),
                _grid({"alpha": [0, 1e-4, 1e-3, 1e-2, 0.1, 1]}), "grid"),
    "svr": ("Support Vector Regression", _pipe(SVR()),
            _grids([{"kernel": ["linear"], "C": [0.1, 1, 10, 100], "epsilon": [0.1, 0.5, 1]},
                    {"kernel": ["poly"], "degree": [2, 3], "C": [1, 10, 100], "epsilon": [0.1, 1]},
                    {"kernel": ["rbf", "sigmoid"], "C": [1, 10, 100, 1000],
                     "gamma": ["scale", 0.01, 0.1], "epsilon": [0.1, 1]}]), "grid"),
    "rf": ("Random Forest Regressor", _pipe(RandomForestRegressor(random_state=RANDOM_STATE)),
           _grid({"n_estimators": [100, 200, 400], "max_depth": [None, 4, 8, 12, 16],
                  "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5, 1.0]}), "random"),
}


def make(zoo: dict, name: str, params: dict | None = None):
    """Fresh (unfitted) pipeline for a model, optionally with tuned params."""
    model = clone(zoo[name][1])
    return model.set_params(**params) if params else model


def two_stage_predict(classifier, regressor, X) -> np.ndarray:
    """Countries the classifier says win nothing get 0, the rest get the regressor's count."""
    medal = classifier.predict(X).astype(bool) if classifier is not None else np.ones(len(X), bool)
    pred = np.zeros(len(X))
    if medal.any():
        pred[medal] = np.clip(regressor.predict(X[medal]), 0, None)
    return pred


class TwoStageModel(BaseEstimator, RegressorMixin):
    """Classifier + regressor, fit and used as one estimator. classifier=None -> plain regression."""

    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        won = y > 0
        if self.classifier is None:
            self.regressor_ = clone(self.regressor).fit(X, y)
            self.classifier_ = None
        else:
            self.classifier_ = clone(self.classifier).fit(X, won.astype(int))
            self.regressor_ = clone(self.regressor).fit(X[won], y[won])
        return self

    def predict(self, X):
        return two_stage_predict(self.classifier_, self.regressor_, np.asarray(X, float))
