from __future__ import print_function, absolute_import, division

import numpy as np

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


def polynomialize_data(data_sets, degree=2):
    data_transformer = lambda X: PolynomialFeatures(degree=degree).fit_transform(X)
    return [(data_transformer(data[0]), data[1]) for data in data_sets]


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

    print("Generated polynomial:\n\t %s" % generate_polynomial(reg_mod, degree))


def regression(data_files, test_data_files=None, training_fraction=0.6, degree=2, limits=None, normalize=()):
    dataset = None
    if test_data_files is None:
        data = load_data(data_files)
        N = len(data)
        print(" >> Loaded %d data points" % N)

        dataset = data
        data = preprocess_data(data, limits=limits, normalize=normalize)
        print(" >> Using  %d / %d data points after preprocessing (deleted %d points)" %
              (len(data), N, N-len(data)))

        data = extract_data_set(data)
        data, training_data, test_data = split_data_set(data,
                                                        training_fraction=training_fraction)

        print(" >> Training on %d, testing on %d" % (training_data[0].shape[0], test_data[0].shape[0]))
        if not (data and training_data and test_data):
            print(" >> Did not have enough data to do regression")
            return

    else:
        raise NotImplementedError("This functionality is yet to be implemented")

    [data, training_data, test_data] = polynomialize_data([data, training_data, test_data], degree=degree)

    reg_mod = linear_regression(*training_data)

    evaluate(training_data, test_data, reg_mod, degree=degree)
    return dataset, (data, training_data, test_data, reg_mod)


def read_and_split_files(data_files, training_fraction=0.6, degree=2, limits=None, normalize=()):
    data = load_data(data_files)
    print(" >> Loaded %d data points" % len(data))

    data = preprocess_data(data, limits=limits, normalize=normalize)
    print(" >> After preprocessing %d data points remaining" % len(data))
    if training_fraction > 0:
        data = extract_data_set(data)
        data, training_data, test_data = split_data_set(data, training_fraction=training_fraction)

        if not (data and training_data and test_data):
            print(" >> Did not have enough data to do regression")
            return

        return polynomialize_data([data, training_data, test_data], degree=degree)
    else:
        data = extract_data_set(data)
        return polynomialize_data([data], degree=degree)

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
    [data, training_data, test_data] = read_and_split_files(data_file,
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
    [data] = read_and_split_files(data_file, training_fraction=0, degree=degree, limits=limits, normalize=normalize)
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


def generate_polynomial(linear_model, degree, features=FEATURES):
    float_fmt = '%.4f'
    polypoly = float_fmt % (linear_model.coef_[0] + linear_model.intercept_)

    term = lambda coef, vars: "*".join([float_fmt % abs(coef)] + list(vars))

    variables = list(combinations_with_replacement(FEATURES, degree))
    for variable, coef in zip(variables, linear_model.coef_[1:]):
        variable = map(lambda X: "<" + X + ">", variable)
        polypoly += "\n\t+ " if coef >= 0 else "\n\t- "
        polypoly += term(coef, variable)

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
