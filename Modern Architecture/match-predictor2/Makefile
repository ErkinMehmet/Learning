SHELL = /usr/bin/env bash -o pipefail

default: help

.PHONY: help
help:
	# Usage:
	@sed -n '/^\([a-z][^:]*\).*/s//    make \1/p' $(MAKEFILE_LIST)

.PHONY: backend/install
backend/install:
	python -m venv backend/env; \
	./backend/env/Scripts/activate; \
	pip install -r backend/requirements.txt; \

.PHONY: backend/types
backend/types:
	./backend/env/Scripts/activate; \
	mypy ./backend/matchpredictor ./backend/test ./backend/fakecsvprovider; \

.PHONY: backend/test
backend/test: backend/types
	./backend/env/Scripts/activate; \
	python -m unittest; \

.PHONY: backend/measure
backend/measure:
	./backend/env/bin/activate; \
	python -m unittest discover -p "measure_*.py"; \

.PHONY: backend/report
backend/report:
	./backend/env/bin/activate; \
	python ./backend/report.py; \

.PHONY: backend/run
backend/run:
	./backend/env/Scripts/activate; \
	cd backend; \
	python -m matchpredictor; \

.PHONY: frontend/lint
frontend/lint:
	npm --prefix frontend run lint

.PHONY: frontend/test
frontend/test: frontend/lint
	npm --prefix frontend test

.PHONY: frontend/install
frontend/install:
	npm --prefix frontend install

.PHONY: frontend/run
frontend/run:
	npm --prefix frontend start

.PHONY: integration/install
integration/install:
	npm --prefix integration-tests install

.PHONY: integration/test
integration/test:
	./backend/env/bin/activate; \
	cd integration-tests; \
	./run; \

.PHONY: integration/run
integration/run:
	./backend/env/bin/activate; \
	cd integration-tests; \
	KEEP_OPEN=true ./run; \

.PHONY: install
install: backend/install frontend/install integration/install

.PHONY: unittest
unittest: backend/test frontend/test

.PHONY: test
test: unittest integration/test

