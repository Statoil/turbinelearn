from __future__ import print_function, absolute_import, division

import numpy as np

from pandas import DataFrame

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from itertools import combinations_with_replacement

from .turbine_file import (enum_files, load_single_file, load_data,
                           normalize_column, preprocess_data, split_data_set,
                           extract_data_set, FEATURES, TARGET)

from .dual_linear_regression import DualLinearModel


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

    return DataFrame(data=poly_X, index=X.index, columns=feature_names)


def linear_regression(X, y):
    reg = linear_model.LinearRegression()
    reg = reg.fit(X, y)
    return reg


def evaluate(training_data, test_data, reg_mod, degree=2):
    r2_train = reg_mod.score(*training_data)
    r2_test  = reg_mod.score(*test_data)
    print("R^2 training: %.5f" % reg_mod.score(*training_data))
    print("R^2 test:     %.5f" % reg_mod.score(*test_data))

    rms_train = mean_squared_error(training_data[1], reg_mod.predict(training_data[0]))
    rms_test  = mean_squared_error(test_data[1], reg_mod.predict(test_data[0]))
    print("RMS training: %.5f" % rms_train)
    print("RMS test:     %.5f" % rms_test)

    print("Generated polynomial:\n\t %s" %
                        generate_polynomial(reg_mod, features=list(training_data[0])))


def regression(data_files, training_fraction=0.6, degree=2, limits=None, normalize=(),):
    dataset = None

    data = load_data(data_files)
    N = len(data)
    print(" >> Loaded %d data points" % N)

    dataset = data
    data = preprocess_data(data, limits=limits, normalize=normalize)
    print(" >> Using  %d / %d data points after preprocessing (deleted %d points)" %
          (len(data), N, N-len(data)))

    data = extract_data_set(data)
    data[0] = polynomialize_data(data[0])
    data, training_data, test_data = split_data_set(data,
                                                    training_fraction=training_fraction)

    print(" >> Training on %d, testing on %d" % (training_data[0].shape[0], test_data[0].shape[0]))
    if not (data and training_data and test_data):
        print(" >> Did not have enough data to do regression")
        return

    reg_mod = linear_regression(*training_data)

    evaluate(training_data, test_data, reg_mod, degree=degree)
    return dataset, (data, training_data, test_data, reg_mod)


def fetch_and_split_data(data_files, training_fraction=0.6, degree=2, limits=None, normalize=()):
    data = load_data(data_files)
    print(" >> Loaded %d data points" % len(data))

    data = preprocess_data(data, limits=limits, normalize=normalize)
    print(" >> After preprocessing %d data points remaining" % len(data))

    if len(data) == 0:
        raise AssertionError("No data left after preprocessing.")

    data = extract_data_set(data)
    data[0] = polynomialize_data(data[0], degree=degree)

    return split_data_set(data, training_fraction=training_fraction)

def fetch_data(data_files, degree=1, dual_model=False, limits=None, normalize=()):
    data = load_data(data_files)
    print(" >> Loaded %d data points" % len(data))

    load_features = FEATURES+["TURBINE_TYPE"]
    data = preprocess_data(data, features=load_features, limits=limits, normalize=normalize)
    print(" >> After preprocessing %d data points remaining" % len(data))

    X, y = data[FEATURES], data[TARGET]

    if degree > 1:
        X = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)

    if dual_model:
        X = DualLinearModel.format(X, np.array(data["TURBINE_TYPE"]))

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
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("Scores:   %s" % ", ".join(['%.4f' % s for s in scores]))


def pca(data_file, limits=None, normalize=()):
    data = load_data(data_file)
    data = preprocess_data(data, limits=limits, normalize=normalize)
    X = data[FEATURES + [TARGET]]
    y = data[TARGET]
    pca = PCA(n_components=2)
    X_2D = pca.fit(X).transform(X)
    explained = pca.explained_variance_ratio_
    print('Explained variance ratio: %.5f, %.5f' % (explained[0], explained[1]))

    X_1 = [x[0] for x in X_2D]
    X_2 = [x[1] for x in X_2D]
    return X_1, X_2, y


def generate_polynomial(linear_model, features):
    float_fmt = '%.4f'

    polypoly = float_fmt % (linear_model.coef_[0] + linear_model.intercept_)
    for variable, coef in zip(features, linear_model.coef_):
        polypoly += "\n\t+ " if coef >= 0 else "\n\t- "
        polypoly += float_fmt % coef + "*" + variable

    return polypoly



def filebased_cross_validation(data_files,
                               test_data_files,
                               degree=2,
                               dual_model=False,
                               limits=None,
                               normalize=()):

    print("\nTraining on data_files %s " % ", ".join(data_files))

    data = fetch_data(data_files, degree=degree,
                      dual_model=dual_model, limits=limits, normalize=normalize)

    if dual_model:
        reg_mod = DualLinearModel.dual_regression(*data)
    else:
        reg_mod = linear_regression(*data)

    r2_scores = [reg_mod.score(*data)]

    print("Testing on test_data_files %s" % ", ".join(test_data_files))
    for input_file in test_data_files:
        test_data = fetch_data(input_file, degree=degree,
                               dual_model=dual_model, limits=limits, normalize=())

        evaluate(data, test_data, reg_mod, degree=degree)
        r2_scores.append(reg_mod.score(*test_data))

    return r2_scores

def file_cross_val(data_files,
                   k=2,
                   degree=2,
                   dual_model=False,
                   limits=None,
                   normalize=()):
    test_data = []

    for training_set, test_set in enum_files(data_files, k):
        r2_scores = filebased_cross_validation(training_set, test_set,
                                               degree=degree, dual_model=dual_model,
                                               limits=limits, normalize=normalize)

        test_data.append((training_set, test_set, r2_scores))
    return test_data
