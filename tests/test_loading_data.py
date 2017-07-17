import os
from datetime import datetime as dt
from unittest import TestCase

import turbinelearn as tblearn
from datasets import relpath

class TestLoadingData(TestCase):

    def setUp(self):
        fname = relpath('old_data', 'turbin_data.csv')
        self.data = tblearn.load_data(fname)

    def test_load(self):
        self.assertEqual(2922, len(self.data))

        loaded_headers = list(self.data.columns.values)
        self.assertEqual(list(self.data), list(self.data.columns.values))

        expected_headers = ["TIME", tblearn.TARGET] + tblearn.FEATURES
        for header in expected_headers:
            self.assertTrue(header in loaded_headers)

    def test_normalize(self):
        features = tblearn.FEATURES + ['SPEED']
        data = tblearn.preprocess_data(self.data,
                                       features=features,
                                       normalize=('SPEED', 'DISCHARGE_TEMP'))
        self.assertEqual(2920, len(data))

        self.assertTrue(0 <= min(data['SPEED']))
        self.assertTrue(max(data['SPEED']) <= 1)
        self.assertTrue(0 <= min(data['DISCHARGE_TEMP']))
        self.assertTrue(max(data['DISCHARGE_TEMP']) <= 1)
        self.assertTrue(max(data['DISCHARGE_PRES']) > 10)

    def test_limits(self):
        speed_min, speed_max = 7000, 8500
        limits = {'SPEED': (speed_min, speed_max)}
        features = tblearn.FEATURES + ['SPEED']
        data = tblearn.preprocess_data(self.data, features=features, limits=limits)
        self.assertEqual(1445, len(data))

        self.assertTrue(speed_min <= min(data['SPEED']))
        self.assertTrue(max(data['SPEED']) <= speed_max)

    def test_limit_time(self):
        window = dt(2016,12,01), dt(2017,01,01)
        limits = {'TIME': window}
        features = tblearn.FEATURES
        data = tblearn.preprocess_data(self.data, features=features, limits=limits)
        self.assertEqual(6*31, len(data))  # ~6 measures per 31 days (=186)
        # First row: 12/1/2016 2:00;17.248752;1.0051291;24.002418;1.0174102;...
        accuracy = 3 # 3 digits?
        self.assertAlmostEqual(data['AIR_IN_TEMP'][0],    17.2488, accuracy)
        self.assertAlmostEqual(data['AIR_IN_PRES'][0],    01.0051, accuracy)
        self.assertAlmostEqual(data['DISCHARGE_TEMP'][0], 24.0024, accuracy)
        self.assertAlmostEqual(data['DISCHARGE_PRES'][0], 01.0174, accuracy)
