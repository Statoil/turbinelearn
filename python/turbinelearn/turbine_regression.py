from __future__ import print_function, absolute_import, division

from collections import namedtuple
from itertools import combinations_with_replacement
import logging
import numpy as np
from pandas import DataFrame, concat

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from .turbine_file import (enum_files, load_single_file, load_data,
                           normalize_column, preprocess_data, split_data_set,
                           extract_data_set, FEATURES, TARGET)
from .turbine_file import POLYNOMIAL_MAP

from .dual_linear_regression import DualLinearModel


LearningResult = namedtuple('LearningResult',
                            ['r2_train',
                             'r2_test',
                             'rms_train',
                             'rms_test',
                             'polynomial'])



def polynomialize_data(X, degree=2):
    poly_factory = PolynomialFeatures(degree=degree, include_bias=False)
    poly_X = poly_factory.fit_transform(X)

    feature_names = poly_factory.get_feature_names(list(X))
    feature_names = [name.replace(" ", "*") for name in feature_names]

    return DataFrame(data=poly_X, index=X.index, columns=feature_names)


def linear_regression(X, y, dual_model=False, ridge=False):
    if dual_model:
        return DualLinearModel.dual_regression(X, y)

    reg = linear_model.LinearRegression()
    if ridge:
        reg = linear_model.Ridge()
    reg = reg.fit(X, y)
    return reg


def do_evaluate(training_data, test_data, reg_mod, degree=2, purge=0):
    r2_train = reg_mod.score(*training_data)
    r2_test  = reg_mod.score(*test_data)
    rms_train = mean_squared_error(training_data[1], reg_mod.predict(training_data[0]))
    rms_test  = mean_squared_error(test_data[1], reg_mod.predict(test_data[0]))
    polynomial = generate_polynomial(reg_mod, features=list(training_data[0]), purge=purge)
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


def evaluate(training_data, test_data, reg_mod, degree=2, purge=0):
    if purge > 0:
        # if purge > 0 we want to output the R^2 score also before purging
        reg_res = do_evaluate(training_data, test_data, reg_mod, degree=2, purge=0)
        logging.info("R^2 test pre-purge:  %.5f" % reg_res.r2_test)
    reg_res = do_evaluate(training_data, test_data, reg_mod, degree=2, purge=purge)
    print_result(reg_res)
    return reg_res


def regression(data_files,
               training_fraction=0.6,
               degree=2,
               limits=None,
               normalize=(),
               ridge=False):
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

    reg_mod = linear_regression(*training_data, ridge=ridge)

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


def _xy_concat(Xy1, Xy2):
    """Takes two pairs Xy1 = (X1, y1) and Xy2 = (X2, y2) and pairwise concatenates
    into (X1|X2, y1|y2).

    """
    return (concat((Xy1[0], Xy2[0]), ignore_index=True),
            concat((Xy1[1], Xy2[1]), ignore_index=True))

def fetch_and_join_data(data_files,
                        training_fraction=0.6,
                        degree=2,
                        dual_model=False,
                        limits=None,
                        normalize=()):
    """Concatenates the training part of all files into on training part, and same
    with test part.

    Beware that this function returns "((None, None), training, test)" since X,y
    is not necessarily sensible; it would contain duplicate indices.

    """

    data = load_data(data_files, concat=False)
    training_frame = None
    test_frame = None
    for f in data:
        load_features = FEATURES + ["TURBINE_TYPE"]
        f = preprocess_data(f, features=load_features, limits=limits, normalize=normalize)
        logging.info(" >> After preprocessing %d data points remaining" % len(f))
        if len(f) == 0:
            raise ValueError("No data left after preprocessing.")

        X, y = f[FEATURES], f[TARGET]
        if degree > 1:
            X = polynomialize_data(X, degree=degree)
        if dual_model:
            X = DualLinearModel.format(X, f["TURBINE_TYPE"])

        _, training_data, test_data = split_data_set((X,y), training_fraction=training_fraction)
        if training_frame is None:
            training_frame = training_data
            test_frame = test_data
        else:
            training_frame = _xy_concat(training_frame, training_data)
            test_frame = _xy_concat(test_frame, test_data)

    return (None, None), training_frame, test_frame


def joined_regression(data_files,
                      training_fraction=0.6,
                      degree=2,
                      dual_model=False,
                      limits=None,
                      normalize=(),
                      purge=0,
                      ridge=False):
    data, train, test = fetch_and_join_data(data_files,
                                            training_fraction=training_fraction,
                                            degree=degree,
                                            dual_model=dual_model,
                                            limits=limits,
                                            normalize=normalize)

    reg_mod = linear_regression(*train, dual_model=dual_model, ridge=ridge)
    reg_res = evaluate(train, test, reg_mod, degree=degree, purge=purge)
    return data_files, (data, train, test, reg_mod), reg_res



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
                                   normalize=(),
                                   ridge=False):
    [data, training_data, test_data] = fetch_and_split_data(data_file,
                                                            training_fraction=training_fraction,
                                                            degree=degree,
                                                            limits=limits,
                                                            normalize=normalize)

    reg_mod = linear_regression(*training_data, ridge=ridge)
    evaluate(training_data, test_data, reg_mod, degree=degree)
    return data, training_data, test_data, reg_mod


def individual_cross_validation(data_file,
                                k=5,
                                degree=2,
                                limits=None,
                                normalize=(),
                                ridge=False):
    data = fetch_data([data_file], degree=degree, limits=limits, normalize=normalize)
    # Split dataset into k consecutive folds (without shuffling).
    model = linear_model.Ridge() if ridge else linear_model.LinearRegression()
    scores = cross_val_score(model, *data, cv=k)
    return scores


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


def compute_learning_progress(data_files,
                              steps=10,
                              degree=2,
                              limits=None,
                              normalize=(),
                              ridge=False):
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

        reg_mod = linear_regression(*training_data, ridge=ridge)

        reg_res = do_evaluate(training_data, test_data, reg_mod, degree=2)
        progress.append(reg_res)
    return progress


def _translate(s, mapping):
    for k in mapping:
        s = s.replace(k, mapping[k])
    return s

def generate_polynomial_terms(linear_model, features, purge=0):
    """Notice that if you call this function with purge > 0, the linear model may be
    altered.

    """
    yield (linear_model.coef_[0] + linear_model.intercept_)
    features = [_translate(f, POLYNOMIAL_MAP) for f in features]
    for i in range(len(features)):
        coef, variable = linear_model.coef_[i], features[i]
        if abs(coef) >= purge:
            yield (coef, variable)
        else:
            if isinstance(linear_model.coef_, np.ndarray):
                linear_model.coef_[i] = 0

def generate_polynomial(linear_model, features, purge=0):
    float_fmt = '%.4f'
    terms = list(generate_polynomial_terms(linear_model, features, purge=purge))
    polypoly = float_fmt % terms[0]
    for coef, variable in terms[1:]:
        polypoly += " "
        polypoly += "+ " if coef >= 0 else "- "
        polypoly += float_fmt % abs(coef) + "*" + variable

    return polypoly


def filebased_cross_validation(data_files,
                               test_data_files,
                               degree=2,
                               dual_model=False,
                               limits=None,
                               normalize=(),
                               ridge=False):

    logging.info("\nTraining on data_files %s " % ", ".join(data_files))

    data = fetch_data(data_files, degree=degree,
                      dual_model=dual_model, limits=limits, normalize=normalize)

    reg_mod = linear_regression(*data, dual_model=dual_model, ridge=ridge)

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
                   normalize=(),
                   ridge=False):
    test_data = []
    learning_results = []

    for training_set, test_set in enum_files(data_files, k):
        r2_scores, res = filebased_cross_validation(training_set,
                                                    test_set,
                                                    degree=degree,
                                                    dual_model=dual_model,
                                                    limits=limits,
                                                    normalize=normalize,
                                                    ridge=ridge)

        test_data.append((training_set, test_set, r2_scores))
        learning_results.append(res)
    return test_data, learning_results
