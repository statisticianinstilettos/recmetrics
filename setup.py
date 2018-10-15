from setuptools import setup, find_packages

setup(
    name='recommender_metrics',
    version='0.0.0',
    author='Claire Longo',
    author_email='longoclaire@gmail.com',
    packages=find_packages(),
    package_dir={'recommender_metrics': 'recommender_metrics'},
    description='metrics for recommender systems',
)
