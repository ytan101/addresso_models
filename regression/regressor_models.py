from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from collections import defaultdict

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB

results_dict = defaultdict(dict)

enable_grid_search = 1


def fit_model(
    model, x, Y, x_validation, validation_identifier, original_model_name=None
):
    model.fit(x, Y)
    y_pred = model.predict(x_validation)
    y_pred = y_pred.round()
    model_name = type(model).__name__
    results_dict[model_name]["accuracy"] = accuracy_score(validation_identifier, y_pred)
    print(y_pred)
    # results_dict[model_name] = model.best_params_


def linear_regressor(X, y, X_validation, validation_identifier):
    linear_r = LinearRegression()
    fit_model(linear_r, X, y, X_validation, validation_identifier)


def random_forest_regressor(X, y, X_validation, validation_identifier):
    rf_r = RandomForestRegressor(max_depth=2, random_state=0)
    fit_model(rf_r, X, y, X_validation, validation_identifier)


def mlp_regressor(X, y, X_validation, validation_identifier):
    mlp_r = MLPRegressor(
        solver="adam",
        batch_size=1000,
        alpha=1e-5,
        activation="tanh",
        max_iter=500,
        early_stopping=False,
        verbose=True,
        hidden_layer_sizes=(50, 50),
        random_state=100,
    )
    fit_model(mlp_r, X, y, X_validation, validation_identifier)


def grid_search(model, parameters, x, Y, x_validation, validation_identifier):

    original_model_name = type(model).__name__
    grid_search_clf = GridSearchCV(model, parameters, scoring="balanced_accuracy")
    fit_model(
        grid_search_clf, x, Y, x_validation, validation_identifier, original_model_name
    )
