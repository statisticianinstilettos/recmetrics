upload:
	# Upload new version to PyPI
	make clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*

clean:
	python setup.py clean --all
	rm -rf *.pyc __pycache__ build dist recmetrics.egg-info recmetrics/__pycache__ tests/__pycache__ tests/reports docs/build .pytest_cache .tox .coverage
