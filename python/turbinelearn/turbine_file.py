import pandas
from itertools import combinations

from datetime import datetime as dt

FEATURES = ["AIR_IN_TEMP", "AIR_IN_PRES", "DISCHARGE_TEMP", "DISCHARGE_PRES"]
TARGET = "SIMULATED_EFF"


LIMITS =   {'SPEED'          : (5000, 10000),
            'SIMULATED_EFF'  : (0, 100),
            'AIR_IN_TEMP'    : (-50, 50),
            'AIR_IN_PRES'    : (0.8, 1.1),
            'DISCHARGE_TEMP' : (0, 600),
            'DISCHARGE_PRES' : (5, 20),
            'TIME':            (dt(2016,01,21), dt(2100,1,1))  # [2016-01-01, 2016-01-21]  seems to be broken
}

FEATURE_MAP = {
    "Air In.Phase - Temperature.Overall.Overall"        : "AIR_IN_TEMP",
    "Air In.Phase - Pressure.Overall.Overall"           : "AIR_IN_PRES",
    "Discharge.Phase - Temperature.Overall.Overall"     : "DISCHARGE_TEMP",
    "Discharge.Phase - Pressure.Overall.Overall"        : "DISCHARGE_PRES",
    "K-100.Polytropic Efficiency.Polytropic Efficiency" : "SIMULATED_EFF",
    "GAS GENERATOR SPEED"                               : "SPEED"
}


def enum_files(dataset, k=2):
    """For cross validation: enumerates (training, test) for k-fold cv"""
    takeout = len(dataset) - k
    data_files = set(dataset)
    for training_set in combinations(dataset, takeout):
        test_set = data_files.difference(training_set)
        yield training_set, test_set


def load_single_file(filename):
    data = pandas.read_csv(filename, sep=";", header=0)
    data = data.rename(columns=FEATURE_MAP)
    data["TIME"] = pandas.to_datetime(data["TIME"])
    # Marks the dataset as comming from a variable (HG) or full (HT) -speed turbine
    data["TURBINE_TYPE"] = 0 if "HG" in filename.upper() else 1
    return data


def load_data(files, concat=True):
    if isinstance(files, str):
        files = [files]

    if concat:
        return pandas.concat(map(load_single_file, files))
    return map(load_single_file, files)


def normalize_column(data, field):
    if field not in data:
        return
    min_, max_ = min(data[field]), max(data[field])
    dif_ = max_ - min_
    data[field] -= min_
    data[field] /= dif_


def preprocess_data(data, features=FEATURES, target=TARGET, limits={}, normalize=()):
    if limits is None:
        limits = {}
    for key in limits:
        min_, max_ = limits[key]
        if key == 'TIME':
            continue
        data = data[(min_ <= data[key]) & (data[key] <= max_)]
    data = data[features + [target, "TIME"]]
    data = data.dropna(axis=0, how="any")
    data = data.reset_index()

    if len(data.index) == 0:
        raise ValueError('Empty dataset after preprocessing!')

    for field in normalize:
        normalize_column(data, field)
    data = data.sort_values('TIME')
    data = data.set_index("TIME")
    if 'TIME' in limits:
        min_, max_ = limits['TIME']
        start = data.index.searchsorted(min_)
        end   = data.index.searchsorted(max_)
        data  = data.ix[start:end]
    return data


def extract_data_set(data, features=FEATURES, target=TARGET):
    return [data[FEATURES], data[TARGET]]


def split_data_set(data, training_fraction=0.6):
    X, y = data[0], data[1]

    split_index = int(len(X)*training_fraction)

    if split_index >= len(X):
        return [[]] * 3

    training_data = (X.iloc[:split_index:], y.iloc[:split_index:])
    test_data = (X.iloc[split_index::], y.iloc[split_index::])

    return (X, y), training_data, test_data
