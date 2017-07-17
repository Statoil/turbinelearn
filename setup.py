#!/usr/bin/env python

from os import path as ospath
from setuptools import setup
import unittest


def relpath(*args):
    """Return path of args relative to this file"""
    root = ospath.dirname(__file__)
    if isinstance(args, str):
        return ospath.join(root, args)
    return ospath.join(root, *args)

def requirements():
    reqs = []
    with open(relpath('requirements.txt'), 'r') as f:
        reqs = [req.strip() for req in f]
    return reqs

def turbine_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(relpath('tests'), pattern='test_*.py')
    return test_suite

setup(
    name="turbinelearn",
    packages=["turbinelearn"],
    package_dir={"turbinelearn" : relpath('python', 'turbinelearn')},
    scripts=[relpath('bin', 'turbine')],
    test_suite='setup.turbine_test_suite',
    install_requires=requirements()
)
