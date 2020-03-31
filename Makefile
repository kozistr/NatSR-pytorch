.PHONY: init check format requirements

init:
	pip3 install -U pipenv
	pipenv sync --dev

check:
	isort --recursive --check-only natsr
	black -S -l 79 --check natsr
	pylint natsr

format:
	isort -rc -y natsr
	black -S -l 79 natsr

requirements:
	pipenv lock -r > requirements.txt
	pipenv lock -dr > requirements-dev.txt

run:
	python3 -m natsr
