from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='RecMetrics',
    url='https://github.com/statisticianinstilettos/recommender_metrics',
    author='Claire Longo',
    author_email='longoclaire@gmail.com',
    # Needed to actually package something
    packages=['recmetrics'],
    # Needed for dependencies
    install_requires=['numpy==1.15.2',
        'pandas==0.23.4',
        'scikit-learn==0.20.0',
        'scipy==1.1.0',
        'seaborn==0.9.0'],
    # *strongly* suggested for sharing
    version='0.1',
    description='Evaluation metrics for recommender systems',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
