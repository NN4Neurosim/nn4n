# created By: Zhaoze Wang
from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='nn4n',
    version='1.1.0',
    description='Neural Networks for Neuroscience Research',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Zhaoze Wang',
    license='MIT',

    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'IPython',
        'scipy',
    ],
    python_requires='>=3.10',
)
