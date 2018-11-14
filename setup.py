from setuptools import setup

setup(
    name='recmetrics',
    url='https://github.com/statisticianinstilettos/recommender_metrics',
    author='Claire Longo',
    author_email='longoclaire@gmail.com',
    packages=['recmetrics'],
    install_requires=['numpy==1.15.2',
        'pandas==0.23.4',
        'scikit-learn==0.20.0',
        'scipy==1.1.0',
        'seaborn==0.9.0'],
    version='0.1',
    description='Evaluation metrics for recommender systems',
)
