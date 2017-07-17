from unittest import TestCase
import numpy as np
from pandas import Series
from sklearn import linear_model

from turbinelearn import DualLinearModel
import turbinelearn as tblearn
from datasets import relpath

class TestLearning(TestCase):

    def setUp(self):
        self.fnames = [relpath('data', 'LOCO_B_HTA.csv'),
                       relpath('data', 'LOCO_B_HGA.csv')]


    def build_sub_models(self, data):
        X = []
        sub_models = []
        for i in range(2):
            datai = data[data["TURBINE_TYPE"]==i]
            datai = tblearn.preprocess_data(datai)
            Xi, yi = datai[tblearn.FEATURES], datai[tblearn.TARGET]

            reg = linear_model.LinearRegression()
            reg = reg.fit(Xi, yi)

            sub_models.append(reg)
            X.append(Xi)

        return sub_models, X


    def build_dual_model(self, data):
        extended_features = tblearn.FEATURES+["TURBINE_TYPE"]
        data = tblearn.preprocess_data(
                                       data,
                                       features=extended_features
                                       )

        turbine_type = data["TURBINE_TYPE"]
        X, y = data[tblearn.FEATURES], data[tblearn.TARGET]
        dual_X = DualLinearModel.format(X, turbine_type)

        return DualLinearModel.dual_regression(dual_X, y)


    def test_dlr(self):
        """
        This test partitions the data set according to turbine type and builds two
        regression models, one for each part. It then does dual linear
        regression via DualLinearModel on all the data and verifies that the
        returned model agrees with each of the two submodels on their
        respective training sets.
        """

        data = tblearn.load_data(self.fnames)

        sub_models, sub_X = self.build_sub_models(data)
        dual_model = self.build_dual_model(data)

        for i in range(2):
            split_feature = Series(
                                data=np.zeros(len(sub_X[i]))+i,
                                name="TURBINE_TYPE"
                                )

            dual_Xi = DualLinearModel.format(sub_X[i], split_feature)

            sub_pred = sub_models[i].predict(sub_X[i])
            dual_pred = dual_model.predict(dual_Xi)

            error = abs(sub_pred-dual_pred)
            self.assertTrue(max(error) < 1e-10)
