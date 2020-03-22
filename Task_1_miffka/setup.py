from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="assignment1",
    packages=find_packages(),
    version="0.1.0",
    description="Auxiliary library for assignment 1",
    author="Anastasia Miroshnikova",
    license="",
    install_requires=required
)