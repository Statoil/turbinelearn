from setuptools import setup


def turbine_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setup(
    name="turbinelearn",
    packages=["turbinelearn"],
    package_dir={"turbinelearn" : "python/turbinelearn"},
    test_suite='setup.turbine_test_suite'
)
