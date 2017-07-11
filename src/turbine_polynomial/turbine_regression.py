#!/usr/bin/env python
from __future__ import print_function, absolute_import, division
from os.path import split as path_split
from sys import argv
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from itertools import combinations
import argparse

from sklearn.decomposition import PCA

DATASET_HT = ["data/LOCO_B_HTA.csv",
              "data/LOCO_B_HTB.csv",
              "data/LOCO_C_HTA.csv",
              "data/LOCO_C_HTB.csv"]
DATASET_HG = ["data/LOCO_B_HGA.csv",
              "data/LOCO_B_HGB.csv",
              "data/LOCO_C_HGA.csv",
              "data/LOCO_C_HGB.csv"]
DATASET_ALL = DATASET_HG + DATASET_HT

LIMITS =   {'SPEED'          : (5000, 10000),
            'DISCHARGE_PRES' : (5, 20),
            'SIMULATED_EFF'  : (0, 100),
            'AIR_IN_PRES'    : (0.8, 1.1),
            'DISCHARGE_TEMP' : (0, 600)}

feature_map = {
    "Air In.Phase - Temperature.Overall.Overall"        : "AIR_IN_TEMP",
    "Air In.Phase - Pressure.Overall.Overall"           : "AIR_IN_PRES",
    "Discharge.Phase - Temperature.Overall.Overall"     : "DISCHARGE_TEMP",
    "Discharge.Phase - Pressure.Overall.Overall"        : "DISCHARGE_PRES",
    "K-100.Polytropic Efficiency.Polytropic Efficiency" : "SIMULATED_EFF",
    "GAS GENERATOR SPEED"                               : "SPEED"
}

FEATURES = ["AIR_IN_TEMP", "AIR_IN_PRES", "DISCHARGE_TEMP", "DISCHARGE_PRES"]
TARGET = "SIMULATED_EFF"


def detect_outliers(data):
    clf = IsolationForest()
    X = data[FEATURES + [TARGET]]
    clf.fit(X)
    Z = clf.decision_function(X)
    data['OUTLIER'] = Z


def load_single_file(filename):
    data = pandas.read_csv(filename, sep=";", header=0)
    data = data.rename(columns=feature_map)
    data["TIME"] = pandas.to_datetime(data["TIME"])
    # Marks the dataset as comming from a variable (HG) or full (HT) -speed turbine
    data["TURBINE_TYPE"] = 0 if "HG" in filename.upper() else 1
    return data


def load_data(files):
    if (isinstance(files,basestring)):
        files = [files]
    return pandas.concat(map(load_single_file, files))


def normalize_column(data, field):
    if field not in data:
        return
    min_, max_ = min(data[field]), max(data[field])
    dif_ = max_ - min_
    data[field] -= min_
    data[field] /= dif_


def preprocess_data(data, normalize=(), limits={}):
    for key in limits:
        min_, max_ = limits[key]
        data = data[(min_ <= data[key]) & (data[key] <= max_)]
    data = data[FEATURES + [TARGET, "TIME", 'SPEED']]
    data = data.dropna(axis=0, how="any")
    data = data.reset_index()

    for field in normalize:
        normalize_column(data, field)
    data = data.set_index("TIME")
    return data


def split_data_set(data, training_fraction = 0.9):
    split_index = int(len(data)*training_fraction)

    if split_index >= len(data):
        return [[]] * 3
    # This is probably a bit inefficient
    split_time = data.reset_index()["TIME"][split_index]

    X = data[FEATURES]
    y = data[TARGET]

    training_data = (X[:split_time:], y[:split_time:])
    test_data = (X[split_time::], y[split_time::])

    return (X, y), training_data, test_data


def polynomialize_data(data_sets, degree=2):
    data_transformer = lambda X : PolynomialFeatures(degree=degree).fit_transform(X)
    return [(data_transformer(data[0]), data[1]) for data in data_sets]


def linear_regression(X, y):
    reg = linear_model.LinearRegression()
    reg = reg.fit(X, y)
    return reg


def visualize(data, training_data, test_data, reg_mod):
    data_timeline = data[1].index
    training_data_timeline = training_data[1].index
    test_data_timeline = test_data[1].index

    plt.plot(
        data_timeline, data[1], "ro",
        training_data_timeline, reg_mod.predict(training_data[0]), "bo",
        test_data_timeline, reg_mod.predict(test_data[0]), "go",
        markersize=2)


def evaluate(data, training_data, test_data, reg_mod):
    r2_train = reg_mod.score(*training_data)
    r2_test  = reg_mod.score(*test_data)
    print("R^2 training: %.5f" % reg_mod.score(*training_data))
    print("R^2 test:     %.5f" % reg_mod.score(*test_data))

    rms_train = mean_squared_error(training_data[1], reg_mod.predict(training_data[0]))
    rms_test  = mean_squared_error(test_data[1], reg_mod.predict(test_data[0]))
    print("RMS training: %.5f" % rms_train)
    print("RMS test:     %.5f" % rms_test)


def main(data_files, test_data_files=None, training_fraction=0.9, degree=1, limits={}):
    dataset = None
    if test_data_files is None:
        data = load_data(data_files)
        N = len(data)
        print(" >> Loaded %d data points" % N)

        dataset = data
        data = preprocess_data(data, limits=limits)
        print(" >> Using  %d / %d data points after preprocessing (deleted %d points)" %
              (len(data), N, N-len(data)))

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

    evaluate(data, training_data, test_data, reg_mod)
    visualize(data, training_data, test_data, reg_mod)
    return dataset


def read_and_split_files(data_files,training_fraction=0.9, degree=1):
    data = load_data(data_files)
    print(" >> Loaded %d data points" % len(data))

    data = preprocess_data(data, limits=LIMITS)
    print(" >> After preprocessing %d data points remaining" % len(data))
    if training_fraction > 0:
        data, training_data, test_data = split_data_set(data, training_fraction=training_fraction)
        if not (data and training_data and test_data):
            print(" >> Did not have enough data to do regression")
            return
        return polynomialize_data([data, training_data, test_data], degree=degree)
    else:
        X = data[FEATURES]
        y = data[TARGET]
        data=(X,y)
        return polynomialize_data([data], degree=degree)


def train_and_evaluate_single_file(data_file, training_fraction=0.9, degree=1):
    [data, training_data, test_data] = read_and_split_files(data_file, training_fraction=training_fraction, degree=degree)
    reg_mod = linear_regression(*training_data)
    evaluate(
            data,
            training_data,
            test_data,
            reg_mod
            )
    visualize(data, training_data, test_data, reg_mod)


def individual_cross_validation(data_file, k=5, degree=1):
    [data] = read_and_split_files(data_file, training_fraction=0, degree=degree)
    #Split dataset into 5 consecutive folds (without shuffling by default).
    scores = cross_val_score(linear_model.LinearRegression(), *data, cv = k )
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Scores:%s"%str(scores))


def pca(data_file):
    data = load_data(data_file)
    data = preprocess_data(data, limits=LIMITS)
    X = data[FEATURES + [TARGET]]
    y = data[TARGET]
    pca = PCA(n_components=2)
    X_2D = pca.fit(X).transform(X)
    explained = pca.explained_variance_ratio_
    print('Explained variance ratio: %.5f, %.5f' % (explained[0], explained[1]))

    X_1 = [x[0] for x in X_2D]
    X_2 = [x[1] for x in X_2D]

    ### Plot the 2 dimensions with y as color of circle
    cmap = sns.cubehelix_palette(as_cmap=True)

    points = plt.scatter(X_1, X_2, c=y, cmap=cmap)



def filebased_cross_validation(data_files, test_data_files, degree=1):
    print("\nTraining on data_files %s " % ", ".join(data_files))

    [data] = read_and_split_files(data_files, training_fraction=0, degree=degree)
    reg_mod = linear_regression(*data)

    print("Testing on test_data_files %s" % ", ".join(test_data_files))
    for input_file in test_data_files:
        [data] = read_and_split_files(input_file,training_fraction=0, degree=degree)
        score = reg_mod.score(*data)
        print("Score on %s is %.4f" % (input_file, score))


def file_cross_val(data_files,k=2,degree=2):
    in_size = len(data_files)
    data_files=set(data_files)
    for training_set in combinations(data_files,in_size-k):
        test_set = data_files.difference(training_set)
        filebased_cross_validation(training_set,test_set,degree=degree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learns a polynomial.')
    parser.add_argument('--method', dest='method',choices=['fcv', 'icv', 'simple', 'pca', 'reg'], required=True,
                        help='Uses either filebased cross validation (fcv), individual file cross validation (icv), run a single train-test run on each file (simple) or do PCA (pca)')
    parser.add_argument('--degree', dest='degree',type=int, choices=range(1, 10), default=2,
                        help='The degree of the polynomial to train, defaults to 2')

    parser.add_argument('--training-fraction', dest='training_fraction',type=float, default=0.6,
                        help='The fraction of data to use as training data, only useed when method=simple')
    parser.add_argument('--k', dest='k',type=int, choices=range(1, 10), default=2,
                        help='The number of folds to keep out when using cross validation. For method=fcv it is the number of files to keep out')
    parser.add_argument('--dataset', dest='dataset', choices=['HG','HT','all'], default='all',
                        help='The dataset to use, HG, HT, or all.')

    args = parser.parse_args()

    DATASET = DATASET_ALL
    if args.dataset == 'HG':
        DATASET = DATASET_HG
    elif args.dataset == 'HT':
        DATASET = DATASET_HT

    degree = args.degree
    training_fraction = args.training_fraction
    k = args.k

    if args.method == 'fcv':
        file_cross_val(DATASET,k=k,degree=degree)
    if args.method == 'icv':
        for input_file in DATASET:
            print("\nDoing individual cross validation for %s" % input_file)
            individual_cross_validation(input_file, degree=degree)
    if args.method == 'simple':
        for input_file in DATASET:
            print("\nDoing regression for %s" % input_file)
            train_and_evaluate_single_file(input_file,
                                           training_fraction = training_fraction,
                                           degree=degree)


    if args.method == "pca":
        plt.title('Turbine polynomial PCA')
        i = 0
        for input_file in DATASET:
            fname = path_split(input_file)
            i = i + 1
            plt.subplot(len(DATASET)//2, 2, i)
            plt.ylabel(fname[1])
            pca(input_file)
        plt.show()

    if args.method == 'reg':
        plt.title('Turbine polynomial')
        i = 0
        for input_file in DATASET:
            fname = path_split(input_file)
            i = i + 1
            plt.subplot(len(DATASET), 2, i)
            plt.ylabel(fname[1])
            print("\nDoing regression for %s" % input_file)
            data = main(input_file, training_fraction=0.6, degree=3, limits=LIMITS)
            X = data[['SPEED', 'DISCHARGE_TEMP', 'DISCHARGE_PRES']] * [1, 10, 1000]
            i = i + 1
            plt.subplot(len(DATASET), 2, i)
            plt.ylim([0, 20*1000])
            plt.plot(data['TIME'], X, 'o', markersize=2)
        plt.show()
