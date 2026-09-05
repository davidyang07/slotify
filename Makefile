# A thin alias over the npm scripts, for readers who reach for `make` first.
# The npm scripts are the source of truth and are what CI runs; nothing here
# does anything they do not.

.PHONY: help install verify verify-node verify-ml demo preflight

help:
	@echo "make install   - install both Node workspaces (the ML package is a venv; see README)"
	@echo "make verify    - frontend, backend, scripts and the ML suite"
	@echo "make demo      - run the product locally"

install:
	npm run install:all

# The one command. Frontend lint/test/build, backend typecheck/test, the
# checkpoint-selection tests, then the ML suite, repository hygiene, and -- when
# a local corpus is present -- dataset validation, split-leakage checks, feature
# validation and queue integrity.
verify:
	npm run verify:all

verify-node:
	npm run verify

verify-ml:
	npm run verify:ml

demo:
	npm run demo

preflight:
	npm run preflight
