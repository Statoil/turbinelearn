#!/usr/bin/env python
from __future__ import print_function, absolute_import, division
from os.path import split as path_split
from collections import namedtuple
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from turbinelearn import *

LearningParameter = namedtuple('LearningParameter', ['dataset', 'degree', 'training_fraction', 'k'])


def visualize(data, training_data, test_data, reg_mod):
    data_timeline = data[1].index
    training_data_timeline = training_data[1].index
    test_data_timeline = test_data[1].index

    plt.plot(
        data_timeline, data[1], "ro",
        training_data_timeline, reg_mod.predict(training_data[0]), "bo",
        test_data_timeline, reg_mod.predict(test_data[0]), "go",
        markersize=2)


def create_argparse():
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
    return parser.parse_known_args()


def method_fcv(params):
    file_cross_val(params.dataset, k=params.k, degree=params.degree)


def method_icv(params):
    for input_file in params.dataset:
        print("\nDoing individual cross validation for %s" % input_file)
        individual_cross_validation(input_file, degree=params.degree)


def method_simple(params):
    for input_file in params.dataset:
        print("\nDoing regression for %s" % input_file)
        full_model= train_and_evaluate_single_file(input_file,
                                                   training_fraction=params.training_fraction,
                                                   degree=params.degree)
        visualize(*full_model)


def method_pca(params):
    plt.title('Turbine polynomial PCA')
    i = 0
    N = len(params.dataset)
    for input_file in params.dataset:
        fname = path_split(input_file)
        i = i + 1
        plt.subplot(N//2, 2, i)
        plt.ylabel(fname[1])
        X_1, X_2, y = pca(input_file)
        ### Plot the 2 dimensions with y as color of circle
        cmap = sns.cubehelix_palette(as_cmap=True)
        plt.scatter(X_1, X_2, c=y, cmap=cmap)
    plt.show()


def method_reg(params):
    plt.title('Turbine polynomial')
    i = 0
    for input_file in params.dataset:
        fname = path_split(input_file)
        i = i + 1
        plt.subplot(len(params.dataset), 2, i)
        plt.ylabel(fname[1])
        print("\nDoing regression for %s" % input_file)
        data, full_model = regression(input_file, training_fraction=0.6, degree=3, limits=LIMITS)

        visualize(*full_model)

        X = data[['SPEED', 'DISCHARGE_TEMP', 'DISCHARGE_PRES']] * [1, 10, 1000]
        i = i + 1
        plt.subplot(len(params.dataset), 2, i)
        plt.ylim([0, 20*1000])
        plt.plot(data['TIME'], X, 'o', markersize=2)
    plt.show()


def main(args, dataset):
    params = LearningParameter(dataset=dataset,
                               degree=args.degree,
                               training_fraction=args.training_fraction,
                               k=args.k)
    methods = {
        'fcv'    : method_fcv,
        'icv'    : method_icv,
        'simple' : method_simple,
        'pca'    : method_pca,
        'reg'    : method_reg
    }
    fn = methods[args.method]
    fn(params)

def filter_dataset(dataset, filter):
    """filter in (None, 'HG', HT')"""
    if filter not in (None, '', False, 'all', 'HG', 'HT'):
        raise KeyError('Unknown filter "%s"' % filter)
    if not filter or filter == 'all':
        return dataset
    return [fname for fname in dataset if filter in fname]

if __name__ == "__main__":
    args, dataset = create_argparse()
    dataset = filter_dataset(dataset, args.dataset)
    main(args, dataset)
