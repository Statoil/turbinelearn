import pandas, numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

feature_map = {
        "Air In.Phase - Temperature.Overall.Overall"        : "AIR_IN_TEMP",
        "Air In.Phase - Pressure.Overall.Overall"           : "AIR_IN_PRES",
        "Discharge.Phase - Temperature.Overall.Overall"     : "DISCHARGE_TEMP",
        "Discharge.Phase - Pressure.Overall.Overall"        : "DISCHARGE_PRES",
        "K-100.Polytropic Efficiency.Polytropic Efficiency" : "SIMULATED_EFF"
        }

FEATURES = ["AIR_IN_TEMP", "AIR_IN_PRES", "DISCHARGE_TEMP", "DISCHARGE_PRES"]
TARGET = "SIMULATED_EFF"

def load_data(filename):
    data = pandas.read_csv(filename, sep=";", header=0)
    data = data.rename(columns=feature_map)
    data["TIME"] = pandas.to_datetime(data["TIME"])

    return data

def preprocess_data(data):
    data = data[FEATURES + [TARGET, "TIME"]]
    data = data.dropna(axis=0, how="any")
    data = data[data['DISCHARGE_PRES'] > 6]
    data = data[(0 <= data['SIMULATED_EFF']) & (data['SIMULATED_EFF'] <= 100)]
    data = data[(0.8 <= data['AIR_IN_PRES']) & (data['AIR_IN_PRES'] <= 1.1)]
    data = data.reset_index()

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
            test_data_timeline, reg_mod.predict(test_data[0]), "go"
            )

    plt.show()

def evaluate(data, training_data, test_data, reg_mod):
    r2_train = reg_mod.score(*training_data)
    r2_test  = reg_mod.score(*test_data)
    print("R^2 score on training set: %.5f" % reg_mod.score(*training_data))
    print("R^2 score on test set:     %.5f" % reg_mod.score(*test_data))

    rms_train = mean_squared_error(training_data[1], reg_mod.predict(training_data[0]))
    rms_test  = mean_squared_error(test_data[1], reg_mod.predict(test_data[0]))
    print("RMS score on training set: %.5f" % rms_train)
    print("RMS score on test set:     %.5f" % rms_test)

    visualize(data, training_data, test_data, reg_mod)

def main(data_files, test_data_files=None, training_fraction=0.9, degree=1):

    if test_data_files is None:
        data = load_data(data_files)
        print(" >> Loaded %d data points" % len(data))

        data = preprocess_data(data)
        print(" >> After preprocessing %d data points remaining" % len(data))

        data, training_data, test_data = split_data_set(
                                                    data,
                                                    training_fraction=training_fraction
                                                    )

        if not (data and training_data and test_data):
            print(" >> Did not have enough data to do regression")
            return

    else:
        raise NotImplemented("This functionality is yet to be implemented")

    [data, training_data, test_data] = polynomialize_data([data, training_data, test_data], degree=degree)

    reg_mod = linear_regression(*training_data)

    evaluate(
            data,
            training_data,
            test_data,
            reg_mod
            )


def individualCrossVal(data_files, training_fraction=0.9, degree=1):
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
    # main("data/turbin_data.csv", training_fraction=0.6, degree=2)
    data_files = ["data/LOCO_B_HGA.csv",
                  "data/LOCO_B_HGB.csv",
                  "data/LOCO_B_HTA.csv",
                  "data/LOCO_B_HTB.csv",
                  "data/LOCO_C_HGA.csv",
                  "data/LOCO_C_HGB.csv",
                  "data/LOCO_C_HTA.csv",
                  "data/LOCO_C_HTB.csv"]

    for input_file in data_files:
        print("\nDoing regression for %s" % input_file)
        individualCrossVal(input_file, training_fraction=0.6, degree=2)
