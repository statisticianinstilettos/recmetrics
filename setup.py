from setuptools import setup
import io
import os


def read(file_name):
    """Read a text file and return the content as a string."""
    with io.open(os.path.join(os.path.dirname(__file__), file_name),
                 encoding='utf-8') as f:
        return f.read()

setup(
    name='recmetrics',
    url='https://github.com/statisticianinstilettos/recommender_metrics',
    author='Claire Longo',
    author_email='longoclaire@gmail.com',
    packages=['recmetrics'],
    install_requires=['numpy',
        'pandas',
        'scikit-learn',
        'seaborn',
        'surprise',
        'funcsigs',
    ],
    license='MIT',
    version='0.0.12',
    description='Evaluation metrics for recommender systems',
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
