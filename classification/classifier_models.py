from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from collections import defaultdict

from sklearn.neural_network import MLPRegressor, MLPClassifier
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import neighbors
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB

mydict = lambda: defaultdict(mydict)
results_dict = mydict()

enable_grid_search = 1


def fit_model(
    model, x, Y, x_validation, validation_identifier, original_model_name=None
):
    model.fit(x, Y)
    y_pred = model.predict(x_validation)

    if original_model_name is not None:  # Adds CV as a separate value for base model
        results_dict[original_model_name]["Gridsearch"][
            "accuracy"
        ] = f"{accuracy_score(validation_identifier, y_pred)}"
        results_dict[original_model_name]["Gridsearch"][
            "best_params"
        ] = model.best_params_
    else:
        model_name = type(model).__name__
        results_dict[model_name]["accuracy"] = accuracy_score(
            validation_identifier, y_pred
        )
    # results_dict[model_name] = model.best_params_


def mlp_classifier(x, Y, x_validation, validation_identifier):

    mlp_clf = MLPClassifier(
        solver="adam",
        batch_size=1000,
        alpha=1e-5,
        activation="tanh",
        max_iter=500,
        early_stopping=False,
        verbose=True,
        hidden_layer_sizes=(100, 50, 100),
        random_state=100,
    )
    fit_model(mlp_clf, x, Y, x_validation, validation_identifier)


def clp_classifier(x, Y, x_validation, validation_identifier):

    clp = neighbors.KNeighborsClassifier(166, weights="uniform")
    fit_model(clp, x, Y, x_validation, validation_identifier)


def rf_classifier(x, Y, x_validation, validation_identifier):

    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_features="auto",
        max_depth=100,
        min_samples_split=10,
        random_state=00,
        verbose=2,
        warm_start=True,
    )
    fit_model(rf_clf, x, Y, x_validation, validation_identifier)


def ada_classifier(x, Y, x_validation, validation_identifier):

    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=50),
        n_estimators=250,
        learning_rate=1,
        random_state=100,
    )
    fit_model(ada_clf, x, Y, x_validation, validation_identifier)


def sgd_classifier(x, Y, x_validation, validation_identifier):

    sgd_clf = SGDClassifier(loss="modified_huber", verbose=1)
    fit_model(sgd_clf, x, Y, x_validation, validation_identifier)


def gb_classifier(x, Y, x_validation, validation_identifier):

    gb_clf = GradientBoostingClassifier(
        loss="deviance",
        learning_rate=0.1,
        n_estimators=250,
        subsample=1,
        criterion="friedman_mse",
        min_samples_split=10,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=50,
        init=None,
        random_state=00,
        max_features=None,
        verbose=2,
        max_leaf_nodes=None,
        warm_start=False,
    )
    fit_model(gb_clf, x, Y, x_validation, validation_identifier)


def bag_classifier(x, Y, x_validation, validation_identifier):

    bag_clf = BaggingClassifier(
        base_estimator=None,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=1,
        random_state=00,
        verbose=2,
    )
    fit_model(bag_clf, x, Y, x_validation, validation_identifier)

    if enable_grid_search == 1:
        bag_clf_parameters = [
            {
                "n_estimators": [500],
                "max_samples": [0.5, 0.6, 0.7],
                "max_features": [1.0],
            }
        ]
        grid_search(
            bag_clf, bag_clf_parameters, x, Y, x_validation, validation_identifier
        )


def grid_search(model, parameters, x, Y, x_validation, validation_identifier):

    original_model_name = type(model).__name__
    grid_search_clf = GridSearchCV(model, parameters, scoring="balanced_accuracy")
    fit_model(
        grid_search_clf, x, Y, x_validation, validation_identifier, original_model_name
    )
