.PHONY: setup demo serve test

setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install -e ".[dev]"

demo:
	.venv/Scripts/python -m media_intel.cli demo

serve:
	.venv/Scripts/python -m media_intel.cli serve --port 8000

test:
	.venv/Scripts/python -m pytest
