from __future__ import print_function, absolute_import, division
import pandas

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from itertools import combinations


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


def evaluate(data, training_data, test_data, reg_mod):
    r2_train = reg_mod.score(*training_data)
    r2_test  = reg_mod.score(*test_data)
    print("R^2 training: %.5f" % reg_mod.score(*training_data))
    print("R^2 test:     %.5f" % reg_mod.score(*test_data))

    rms_train = mean_squared_error(training_data[1], reg_mod.predict(training_data[0]))
    rms_test  = mean_squared_error(test_data[1], reg_mod.predict(test_data[0]))
    print("RMS training: %.5f" % rms_train)
    print("RMS test:     %.5f" % rms_test)


def regression(data_files, test_data_files=None, training_fraction=0.9, degree=1, limits={}):
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
    return dataset, (data, training_data, test_data, reg_mod)


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
    evaluate(data, training_data, test_data, reg_mod)
    return data, training_data, test_data, reg_mod


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
    return X_1, X_2, y


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
