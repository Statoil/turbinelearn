import os
import itertools
from unittest import TestCase

import turbinelearn as tblearn
from datasets import relpath

class TestLearning(TestCase):

    def setUp(self):
        self.fname = relpath('old_data', 'turbin_data.csv')
        self.data = tblearn.load_data(self.fname)
        self.data_files = map(relpath, [
            'data/LOCO_B_HGA.csv', 'data/LOCO_B_HGB.csv',
            'data/LOCO_B_HTA.csv', 'data/LOCO_B_HTB.csv',
        ])

    def preprocess(self):
        self.data = tblearn.preprocess_data(self.data)

    def test_pca(self):
        X_1, X_2, y = tblearn.pca(self.fname, limits=tblearn.LIMITS)
        self.assertEqual(1438, len(X_1))
        self.assertEqual(len(X_1), len(X_2))
        self.assertEqual(len(X_1), len(y))

    def test_simple(self):
        train = tblearn.train_and_evaluate_single_file
        data, training_data, test_data, reg_mod = train(self.fname, degree=3, limits=tblearn.LIMITS)

        score = reg_mod.score(*training_data)
        self.assertTrue(
                score > 0.9,
                "Score on training data was %f, expected > 0.9." % score
                )

        score = reg_mod.score(*test_data)
        self.assertTrue(
                score > 0.9,
                "Score on test data was %f, expected > 0.9." % score
                )

        poly = tblearn.generate_polynomial(reg_mod, list(data[0]))
        self.assertTrue(poly)

    def test_fcv(self):
        quality_threshold = 0.1

        scenarios = itertools.product([False, True], [1], [2,3])
        for dual_model, k, degree in scenarios:
            test_data, _ = tblearn.file_cross_val(self.data_files,
                                                  k=k,
                                                  degree=degree,
                                                  dual_model=dual_model,
                                                  limits=tblearn.LIMITS)
            results = zip(*test_data)[2]

            min_training_score = min(zip(*results)[0])
            self.assertTrue(
                    min_training_score > quality_threshold,
                    "Minimum fcv training score was %f, expected > %f" %
                    (min_training_score, quality_threshold)
                    )

            min_test_score = min([min(result[1:]) for result in results])
            self.assertTrue(
                    min_test_score > quality_threshold,
                    "Minimum fcv test score was %f, expected > %f" %
                    (min_test_score, quality_threshold)
                    )
