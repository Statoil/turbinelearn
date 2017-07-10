from setuptools import setup

setup(
    name="turbinelearn",
    packages=["turbinelearn"],
    package_dir={"turbinelearn" : "src/turbinelearn"},
    test_suite="tests",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    zip_safe=False
)
