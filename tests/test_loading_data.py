import os
from unittest import TestCase
import turbinelearn as tblearn

class TestLoadingData(TestCase):

    def setUp(self):
        self.data = tblearn.load_data('old_data/turbin_data.csv')

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
