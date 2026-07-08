.PHONY: install test run offline
install:
	pip install -e ".[dev,real]"
test:
	pytest -q
run:
	llm-judge-bias --json results.json
offline:
	llm-judge-bias --offline
