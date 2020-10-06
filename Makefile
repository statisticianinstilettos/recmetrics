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

build:
	docker build -t \
		${REPO_NAME}/${DOCKER_IMAGE_NAME} .

test: build
	docker run \
		--rm \
		-v "$$PWD":/recmetrics \
		${REPO_NAME}/${DOCKER_IMAGE_NAME}:latest pytest --cov=recmetrics


build-demo:
	docker build \
		-f Dockerfile-demo
		-t ${REPO_NAME}/${DOCKER_IMAGE_NAME}-demo .


# demo:
# 	docker run \
# 		--rm \
# 		-v "$$PWD":/recmetrics \
# 		${REPO_NAME}/${DOCKER_IMAGE_NAME}:latest pytest --cov=recmetrics
