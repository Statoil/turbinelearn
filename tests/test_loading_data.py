import os

from unittest import TestCase

import turbinelearn as tp

class TestLoadingData(TestCase):

    def test_load(self):
        data = tp.load_data("data/turbin_data.csv")

        self.assertEqual(2922, len(data))

        loaded_headers = list(data.columns.values)
        self.assertEqual(list(data), list(data.columns.values))

        expected_headers = ["TIME", tp.TARGET] + tp.FEATURES
        for header in expected_headers:
            self.assertTrue(
                    header in loaded_headers,
                    "%s was not in %r" % (header, loaded_headers)
                    )
