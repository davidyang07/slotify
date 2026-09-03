# A thin alias over the npm scripts, for readers who reach for `make` first.
# The npm scripts are the source of truth and are what CI runs; nothing here
# does anything they do not.

.PHONY: help install verify verify-node verify-ml evidence demo preflight

help:
	@echo "make install   - install both Node workspaces (the ML package is a venv; see README)"
	@echo "make verify    - everything checkable without collecting new human labels"
	@echo "make evidence  - regenerate the statistical artifacts and both evidence reports"
	@echo "make demo      - run the product locally"

install:
	npm run install:all

# The one command. Frontend lint/test/build, backend typecheck/test, the
# checkpoint-selection tests, then the ML suite, repository hygiene, both
# generated evidence reports re-derived and compared, and -- when a local corpus
# is present -- dataset validation, split-leakage checks, feature validation,
# queue integrity and the experiment readiness gate.
verify:
	npm run verify:all

verify-node:
	npm run verify

verify-ml:
	npm run verify:ml

evidence:
	npm run evidence
	npm run claim-evidence

demo:
	npm run demo

preflight:
	npm run preflight
