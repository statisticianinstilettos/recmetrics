# Global variables
REPO_NAME=statisticianinstilletos
DOCKER_IMAGE_NAME=recmetrics
DOCKER_IMAGE_TAG=dev

# Upload new version to PyPI
upload: clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*

# Remove files from repo
clean:
	python3 setup.py clean --all
	rm -rf *.pyc __pycache__ build dist recmetrics.egg-info recmetrics/__pycache__ tests/__pycache__ tests/reports docs/build .pytest_cache .tox .coverage

# Download MovieLens data to repo
download_movielens:
	wget http://files.grouplens.org/datasets/movielens/ml-20m.zip && \
		unzip ml-20m.zip

# Create RecMetrics Docker image (Development)
build:
	docker build -t \
		${REPO_NAME}/${DOCKER_IMAGE_NAME} .

# Run RecMetrics Docker image (Development)
run: build
	docker run -it \
		${REPO_NAME}/${DOCKER_IMAGE_NAME} /bin/bash

# Test RecMetrics Docker image
test: clean build
	docker run \
		--rm \
		${REPO_NAME}/${DOCKER_IMAGE_NAME}:latest pytest --cov=recmetrics

# Build RecMetrics Docker image (Demo)
build_demo:
	docker build \
		-f Dockerfile-demo \
		-t ${REPO_NAME}/${DOCKER_IMAGE_NAME}-demo .

# Run RecMetrics Docker image (Demo)
run_demo: build_demo
	docker run \
		--rm \
		-p 8888:8888 \
		-v "$$PWD":/recmetrics \
		${REPO_NAME}/${DOCKER_IMAGE_NAME}-demo:latest jupyter lab
