# created By: Zhaoze Wang
from setuptools import setup, find_packages

setup(
    name='nn4n',
    version='0.0.1',
    description='Neural Networks for Neuroscience',
    author='Zhaoze Wang',
    license='MIT',

    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'IPython'
    ],
    python_requires='>=3.7',
)
