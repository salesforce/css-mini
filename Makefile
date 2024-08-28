.PHONY: clean prerequisites install develop lint package test
RM := rm -rf
PIP_UPGRADE := pip3 install -U --upgrade-strategy eager

clean:
	$(RM) $(filter-out .venv, $(wildcard *egg-info .coverage .mypy_cache .pytest_cache \
		.tox reports .ruff_cache))
	find . -name '__pycache__' -not -path './.venv/*' | xargs $(RM)
	find . -name '.ipynb_checkpoints' -not -path './.venv/*' | xargs $(RM)

prerequisites:
	$(PIP_UPGRADE) pip setuptools wheel setuptools_scm[toml]

install: prerequisites
	$(PIP_UPGRADE) .

develop: prerequisites
	$(PIP_UPGRADE) -e '.[dev]'

lint:
	pre-commit run --all-files --hook-stage manual

package: prerequisites
	$(RM) dist/
	$(PIP_UPGRADE) build
	pyproject-build --no-isolation

test:
	tox --recreate
