from __future__ import print_function, absolute_import, division

from collections import namedtuple
from itertools import combinations_with_replacement
import logging
import numpy as np
from pandas import DataFrame

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .turbine_file import (enum_files, load_single_file, load_data,
                           normalize_column, preprocess_data, split_data_set,
                           extract_data_set, FEATURES, TARGET)

from .dual_linear_regression import DualLinearModel


LearningResult = namedtuple('LearningResult',
                            ['r2_train',
                             'r2_test',
                             'rms_train',
                             'rms_test',
                             'polynomial'])


def detect_outliers(data):
    clf = IsolationForest()
    X = data[FEATURES + [TARGET]]
    clf.fit(X)
    Z = clf.decision_function(X)
    data['OUTLIER'] = Z


def polynomialize_data(X, degree=2):
    poly_factory = PolynomialFeatures(degree=degree, include_bias=False)
    poly_X = poly_factory.fit_transform(X)

    feature_names = poly_factory.get_feature_names(list(X))
    feature_names = [name.replace(" ", "*") for name in feature_names]

    return DataFrame(data=poly_X, index=X.index, columns=feature_names)


def linear_regression(X, y):
    reg = linear_model.LinearRegression()
    reg = reg.fit(X, y)
    return reg


def do_evaluate(training_data, test_data, reg_mod, degree=2):
    r2_train = reg_mod.score(*training_data)
    r2_test  = reg_mod.score(*test_data)
    rms_train = mean_squared_error(training_data[1], reg_mod.predict(training_data[0]))
    rms_test  = mean_squared_error(test_data[1], reg_mod.predict(test_data[0]))
    polynomial = generate_polynomial(reg_mod, features=list(training_data[0]))
    res = LearningResult(r2_train=r2_train,
                         r2_test=r2_test,
                         rms_train=rms_train,
                         rms_test=rms_test,
                         polynomial=polynomial)
    return res


def print_result(learning_result):
    logging.info("R^2 training: %.5f" % learning_result.r2_train)
    logging.info("R^2 test:     %.5f" % learning_result.r2_test)

    logging.info("RMS training: %.5f" % learning_result.rms_train)
    logging.info("RMS test:     %.5f" % learning_result.rms_test)

    logging.info("Generated polynomial:\n\t %s" % learning_result.polynomial)


def evaluate(training_data, test_data, reg_mod, degree=2):
    reg_res = do_evaluate(training_data, test_data, reg_mod, degree=2)
    print_result(reg_res)
    return reg_res


def regression(data_files, training_fraction=0.6, degree=2, limits=None, normalize=()):
    dataset = None

    data = load_data(data_files)
    N = len(data)
    logging.info(" >> Loaded %d data points" % N)

    dataset = data
    data = preprocess_data(data, limits=limits, normalize=normalize)
    logging.info(" >> Using  %d / %d data points after preprocessing (deleted %d points)" %
          (len(data), N, N-len(data)))

    data = extract_data_set(data)
    data[0] = polynomialize_data(data[0])
    data, training_data, test_data = split_data_set(data,
                                                    training_fraction=training_fraction)

    train_size, test_size = training_data[0].shape[0], test_data[0].shape[0]
    logging.info(" >> Training on %d, testing on %d" % (train_size, test_size))
    if not (data and training_data and test_data):
        logging.WARN(" >> Did not have enough data to do regression")
        return

    reg_mod = linear_regression(*training_data)

    reg_res = evaluate(training_data, test_data, reg_mod, degree=degree)
    return dataset, (data, training_data, test_data, reg_mod), reg_res


def fetch_and_split_data(data_files, training_fraction=0.6, degree=2, limits=None, normalize=()):
    data = load_data(data_files)
    logging.info(" >> Loaded %d data points" % len(data))

    data = preprocess_data(data, limits=limits, normalize=normalize)
    logging.info(" >> After preprocessing %d data points remaining" % len(data))

    if len(data) == 0:
        raise ValueError("No data left after preprocessing.")

    data = extract_data_set(data)
    data[0] = polynomialize_data(data[0], degree=degree)

    return split_data_set(data, training_fraction=training_fraction)

def fetch_data(data_files, degree=1, dual_model=False, limits=None, normalize=()):
    data = load_data(data_files)
    logging.info(" >> Loaded %d data points" % len(data))

    load_features = FEATURES+["TURBINE_TYPE"]
    data = preprocess_data(data, features=load_features, limits=limits, normalize=normalize)
    logging.info(" >> After preprocessing %d data points remaining" % len(data))

    X, y = data[FEATURES], data[TARGET]

    if degree > 1:
        X = polynomialize_data(X, degree=degree)

    if dual_model:
        X = DualLinearModel.format(X, data["TURBINE_TYPE"])

    return (X, y)

def train_and_evaluate_single_file(data_file,
                                   training_fraction=0.6,
                                   degree=2,
                                   limits=None,
                                   normalize=()):
    [data, training_data, test_data] = fetch_and_split_data(data_file,
                                                            training_fraction=training_fraction,
                                                            degree=degree,
                                                            limits=limits,
                                                            normalize=normalize)

    reg_mod = linear_regression(*training_data)
    evaluate(training_data, test_data, reg_mod, degree=degree)
    return data, training_data, test_data, reg_mod


def individual_cross_validation(data_file,
                                k=5,
                                degree=2,
                                limits=None,
                                normalize=()):
    data = fetch_data([data_file], degree=degree, limits=limits, normalize=normalize)
    # Split dataset into k consecutive folds (without shuffling).
    scores = cross_val_score(linear_model.LinearRegression(), *data, cv=k)
    logging.info("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    logging.info("Scores:   %s" % ", ".join(['%.4f' % s for s in scores]))


def pca(data_file, limits=None, normalize=()):
    data = load_data(data_file)
    data = preprocess_data(data, limits=limits, normalize=normalize)
    X = data[FEATURES + [TARGET]]
    y = data[TARGET]
    pca = PCA(n_components=2)
    X_2D = pca.fit(X).transform(X)
    explained = pca.explained_variance_ratio_
    logging.info('Explained variance ratio: %.5f, %.5f' % (explained[0], explained[1]))

    X_1 = [x[0] for x in X_2D]
    X_2 = [x[1] for x in X_2D]
    return X_1, X_2, y


def compute_learning_progress(data_files, steps=10, degree=2, limits=None, normalize=()):
    """Trains a polynomial with n/steps fraction increments.  Returns error."""

    data = load_data(data_files)
    N = len(data)
    logging.info(" >> Loaded %d data points" % N)

    progress = []
    step_size = 1/(steps+1)
    for i in range(steps):
        fraction = (i+1) * step_size
        data = load_data(data_files)
        data = preprocess_data(data, limits=limits, normalize=normalize)
        logging.info(" >> Using  %d / %d data points after preprocessing (deleted %d points)" %
                     (len(data), N, N-len(data)))

        data = extract_data_set(data)
        data[0] = polynomialize_data(data[0])
        data, training_data, test_data = split_data_set(data,
                                                        training_fraction=fraction)

        if not (data and training_data and test_data):
            logging.WARN(" >> Did not have enough data to do regression")
            continue

        reg_mod = linear_regression(*training_data)

        reg_res = do_evaluate(training_data, test_data, reg_mod, degree=2)
        progress.append(reg_res)
    return progress



def generate_polynomial(linear_model, features):
    float_fmt = '%.4f'

    polypoly = float_fmt % (linear_model.coef_[0] + linear_model.intercept_)
    for variable, coef in zip(features, linear_model.coef_):
        polypoly += "\n\t+ " if coef >= 0 else "\n\t- "
        polypoly += float_fmt % abs(coef) + "*" + variable

    return polypoly


def filebased_cross_validation(data_files,
                               test_data_files,
                               degree=2,
                               dual_model=False,
                               limits=None,
                               normalize=()):

    logging.info("\nTraining on data_files %s " % ", ".join(data_files))

    data = fetch_data(data_files, degree=degree,
                      dual_model=dual_model, limits=limits, normalize=normalize)

    if dual_model:
        reg_mod = DualLinearModel.dual_regression(*data)
    else:
        reg_mod = linear_regression(*data)

    r2_scores = [reg_mod.score(*data)]
    learning_results = []

    logging.info("Testing on test_data_files %s" % ", ".join(test_data_files))
    for input_file in test_data_files:
        test_data = fetch_data(input_file, degree=degree,
                               dual_model=dual_model, limits=limits, normalize=())

        learn_res = evaluate(data, test_data, reg_mod, degree=degree)
        r2_scores.append(reg_mod.score(*test_data))
        learning_results.append(learn_res)

    return r2_scores, learning_results

def file_cross_val(data_files,
                   k=2,
                   degree=2,
                   dual_model=False,
                   limits=None,
                   normalize=()):
    test_data = []
    learning_results = []

    for training_set, test_set in enum_files(data_files, k):
        r2_scores, res = filebased_cross_validation(training_set,
                                                    test_set,
                                                    degree=degree,
                                                    dual_model=dual_model,
                                                    limits=limits,
                                                    normalize=normalize)

        test_data.append((training_set, test_set, r2_scores))
        learning_results.append(res)
    return test_data, learning_results
