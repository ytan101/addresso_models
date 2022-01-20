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

from utils import concordance_correlation_coefficient

# Create nested defaultdicts
results_dict = defaultdict(lambda: defaultdict(dict))

enable_grid_search = 1

data_folder = ""


def fit_model(
    model, x, Y, x_validation, validation_identifier, original_model_name=None
):
    model.fit(x, Y)
    y_pred = model.predict(x_validation)
    y_pred = y_pred.round()
    model_name = type(model).__name__
    results_dict[model_name]["Pred"] = y_pred
    results_dict[model_name]["Actual"] = validation_identifier
    results_dict[model_name]["MSE"] = mean_squared_error(
        validation_identifier, y_pred)
    results_dict[model_name]["CCC"] = concordance_correlation_coefficient(
        validation_identifier, y_pred)
    results_dict[model_name]["Pearson R"]["R"], results_dict[model_name]["Pearson R"]["p-value"] = pearsonr(
        validation_identifier, y_pred)
    return results_dict


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
        hidden_layer_sizes=(50, 100, 50),
        random_state=100,
    )
    fit_model(mlp_r, X, y, X_validation, validation_identifier)

def sgd_regressor(X, y, X_validation, validation_identifier):
    sgd_r = SGDRegressor(max_iter=1000, tol=1e-3)
    fit_model(sgd_r, X, y, X_validation, validation_identifier)

def gradient_boosting_regressor(X, y, X_validation, validation_identifier):
    gb_r = GradientBoostingRegressor(random_state=0)
    fit_model(gb_r, X, y, X_validation, validation_identifier)
