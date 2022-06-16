from setuptools import setup
from setuptools import find_packages

setup(
    name = "tunnelslam",
    version = "1.0.0",
    description = "bla bla",
    author = "bla bla",
    url = "https://github.com/AlonSpinner/Tunnelslam",
    packages = find_packages(exclude = ('tests*')),
    )
