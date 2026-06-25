.PHONY: help install ingest index run api test eval lint docker-build docker-up clean all

help:
	@echo "Targets: install ingest index run api test eval lint docker-build docker-up clean all"

install:
	pip install -r requirements.txt

# One-command reproducible pipeline: download -> clean -> chunk -> embed -> index
all:
	python run.py pipeline

ingest:
	python run.py ingest

index:
	python run.py index

# Interactive single query, e.g.: make run Q="What was Irish inflation in 2023?"
run:
	python run.py query --question "$(Q)"

api:
	python run.py api

test:
	pytest

eval:
	python run.py eval

lint:
	ruff check src tests

docker-build:
	docker build -t ireland-cso-rag .

docker-up:
	docker compose up --build

clean:
	rm -rf data/processed/* .pytest_cache .ruff_cache **/__pycache__
