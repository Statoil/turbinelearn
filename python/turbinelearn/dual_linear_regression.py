import numpy as np
from sklearn import linear_model

class DualLinearModel:

    """
    A dual linear model supports the following:
        - partition the data set into two parts,
        - then retrieve linear models (submodels) for each of these parts,
        - combining these two submodels into one unifying model.

    The resulting model will agree with each of the submodels on their
    respective data.
    """

    def __init__():
        raise NotImplementedError("DualLinearModel is a static class")

    @classmethod
    def format(cls, X, split_feature):
        """
        Formats the data according to a specified partition.

        @data the data to be formatted
        @split_feature gives a partition of the data

        split_feature is assumed to be an array consisting of 0 and 1 and
        should be of the same length as the columns of data.

        Both split_feature and data is assumed to support standard numpy
        operations.
        """

        X_split = (X.T * split_feature).T
        sf_column = np.asmatrix(split_feature).T

        return np.concatenate((X, X_split, sf_column), axis=1)

    @classmethod
    def dual_regression(cls, X, y):
        """
        Builds a dual linear model fitted for X and y.

        @X features to be trained on
        @y target values

        It is assumed that X is formatted in a similar way as format does and
        that hence the X consists of submatrices A, At and t, where A are the
        original features, t is the partition of the columns represented by 0
        and 1 and At is the elementwise product of A and t.
        """

        if len(X[0])%2 == 0:
            raise ValueError(
                    "Expected the elements of X to have odd length, was %d" %
                    len(X[0])
                    )

        num_orig_features = X.shape[1]//2

        reg_mod = []
        for split_feature in range(2):
            row_filter = np.array((X[:,-1]==split_feature)).flatten()
            Xi = X[row_filter,:][:,range(num_orig_features)]
            yi = y[row_filter]
            regi = linear_model.LinearRegression()
            regi = regi.fit(Xi, yi)

            reg_mod.append(regi)

        dual_reg_mod = linear_model.LinearRegression()
        dual_reg_mod = dual_reg_mod.fit(X, y)

        dual_reg_mod.intercept_ = reg_mod[0].intercept_
        dual_reg_mod.coef_ = np.concatenate((
                                    reg_mod[0].coef_,
                                    reg_mod[1].coef_-reg_mod[0].coef_,
                                    np.array([reg_mod[1].intercept_-reg_mod[0].intercept_])
                                    ))

        return dual_reg_mod
