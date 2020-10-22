# Global variables
REPO_NAME=statisticianinstilletos
DOCKER_IMAGE_NAME=recmetrics
DOCKER_IMAGE_TAG=dev

upload:
	# Upload new version to PyPI
	make clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*

clean:
	python setup.py clean --all
	rm -rf *.pyc __pycache__ build dist recmetrics.egg-info recmetrics/__pycache__ tests/__pycache__ tests/reports docs/build .pytest_cache .tox .coverage

download_movielens:
	wget http://files.grouplens.org/datasets/movielens/ml-20m.zip && \
		unzip ml-20m.zip

build:
	docker build -t \
		${REPO_NAME}/${DOCKER_IMAGE_NAME} .

run: build
	docker run -it \
		${REPO_NAME}/${DOCKER_IMAGE_NAME} /bin/bash

test: clean build
	docker run \
		--rm \
		-v "$$PWD":/recmetrics \
		${REPO_NAME}/${DOCKER_IMAGE_NAME}:latest pytest --cov=recmetrics

build_demo:
	docker build \
		-f Dockerfile-demo \
		-t ${REPO_NAME}/${DOCKER_IMAGE_NAME}-demo .

run_demo: build_demo
	docker run \
		--rm \
		-p 8888:8888 \
		-v "$$PWD":/recmetrics \
		${REPO_NAME}/${DOCKER_IMAGE_NAME}-demo:latest jupyter lab
