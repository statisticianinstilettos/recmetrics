from setuptools import setup

setup(
    name='recmetrics',
    url='https://github.com/statisticianinstilettos/recommender_metrics',
    author='Claire Longo',
    author_email='longoclaire@gmail.com',
    packages=['recmetrics'],
    install_requires=['numpy',
        'pandas',
        'scikit-learn',
        'seaborn'],
    version='0.0.1',
    description='Evaluation metrics for recommender systems',
)
