import numpy as np

from pandas import DataFrame

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

        @data the data to be formatted as a pandas.DataFrame
        @split_feature gives a partition of the data as a pandas.Series

        split_feature is assumed to consist of 0 and 1 and should be of the
        same length as data.
        """

        Xm, sfm = np.asarray(X), np.asarray(split_feature)

        X_split = (Xm.T * sfm).T
        sf_column = np.asmatrix(sfm).T
        data = np.concatenate((X, X_split, sf_column), axis=1)


        feature_names = list(X.columns.values)
        feature_names += [split_feature.name + "*" + name for name in feature_names]
        feature_names += [split_feature.name]

        return DataFrame(data=data, index=X.index, columns=feature_names)

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

        if len(X.columns)%2 == 0:
            raise ValueError(
                    "Expected X to have an odd number of columns, was %d" %
                    len(X.columns)
                    )

        split_feature_name = X.columns.values[-1]
        split_feature_range = [0, 1]

        num_orig_features = len(X.columns)//2
        orig_features = X.columns.values[:num_orig_features]

        reg_mod = []
        for split_feature_value in split_feature_range:
            split_filter = (X[split_feature_name] == split_feature_value)
            Xi = X[split_filter][orig_features]
            yi = y[split_filter]

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
