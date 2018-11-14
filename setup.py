from setuptools import setup

setup(
    name='recmetrics',
    url='https://github.com/statisticianinstilettos/recommender_metrics',
    author='Claire Longo',
    author_email='longoclaire@gmail.com',
    packages=['recmetrics'],
    install_requires=['numpy==1.15.2',
        'pandas==0.23.4',
        'scikit-learn',
        'scipy',
        'seaborn'],
    version='0.0.0',
    description='Evaluation metrics for recommender systems',
)
