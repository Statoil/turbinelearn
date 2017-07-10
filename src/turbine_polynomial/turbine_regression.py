from __future__ import print_function, absolute_import, division
from os.path import split as path_split
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

DATASET = ["data/LOCO_B_HGA.csv",
           "data/LOCO_B_HGB.csv",
           "data/LOCO_B_HTA.csv",
           "data/LOCO_B_HTB.csv",
           "data/LOCO_C_HGA.csv",
           "data/LOCO_C_HGB.csv",
           "data/LOCO_C_HTA.csv",
           "data/LOCO_C_HTB.csv"]

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


def load_data(filename):
    data = pandas.read_csv(filename, sep=";", header=0)
    data = data.rename(columns=feature_map)
    data["TIME"] = pandas.to_datetime(data["TIME"])

    return data

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
    return data

def split_data_set(data, training_fraction = 0.9):
    split_index = int(len(data)*training_fraction)

    if split_index >= len(data):
        return [[]] * 3

    split_time = data["TIME"][split_index]
    data = data.set_index("TIME")

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
        markersize=2
    )


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

        data, training_data, test_data = split_data_set(
                                                    data,
                                                    training_fraction=training_fraction
                                                    )

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


def individual_cross_validation(data_files, training_fraction=0.9, degree=1):
    data = load_data(data_files)
    print(" >> Loaded %d data points" % len(data))

    data = preprocess_data(data)
    print(" >> After preprocessing %d data points remaining" % len(data))
    #This sets data on a (X,y) format. We ingor ethe training and test_data split
    data, training_data, test_data = split_data_set(
                                                    data,
                                                    training_fraction=training_fraction
                                                    )
    [data] = polynomialize_data([data], degree=degree)

    scores = cross_val_score(linear_model.LinearRegression(), *data, cv = 5 )
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)

if __name__ == "__main__":
    limits =   {'SPEED'         : (5000, 10000),
                'DISCHARGE_PRES': (5, 20),
                'SIMULATED_EFF' : (0, 100),
                'AIR_IN_PRES'   : (0.8, 1.1),
                'DISCHARGE_TEMP': (0, 600)}

    plt.title('Turbine polynomial')
    i = 1
    for input_file in DATASET:
        fname = path_split(input_file)
        plt.subplot(8, 2, i)
        plt.ylabel(fname[1])
        i = i + 1
        print("\nDoing regression for %s" % input_file)
        data = main(input_file, training_fraction=0.6, degree=3, limits=limits)
        plt.subplot(8, 2, i)
        plt.ylim([0, 20*1000])
        plt.ylabel(fname[1])
        plt.plot(data['TIME'], data[['SPEED', 'DISCHARGE_TEMP', 'DISCHARGE_PRES']] * [1, 10, 1000],
                 'o', markersize=2)
        i = i + 1
    plt.show()

    for input_file in data_files:
        print("\nDoing cross validation for %s" % input_file)
        individual_cross_validation(input_file, training_fraction=0.6, degree=2)

