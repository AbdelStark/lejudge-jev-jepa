PY ?= .venv/bin/python
LEJUDGE ?= .venv/bin/lejudge
export LEJUDGE_MODE ?= offline
export SDL_VIDEODRIVER ?= dummy
export STABLEWM_HOME ?= $(HOME)/.stable-wm

.PHONY: test lint report paper clean-figures m0 probes

test:
	$(PY) -m pytest -q

lint:
	$(LEJUDGE) lint
	$(PY) -m ruff check lejudge tests

probes:
	$(LEJUDGE) probes train --kind linear
	$(LEJUDGE) probes train --kind mlp

m0:
	$(LEJUDGE) m0 --episodes 50

report:
	$(LEJUDGE) report --out paper/figures

paper: report
	$(PY) paper/fill.py
	cd paper && (tectonic main.filled.tex >/dev/null 2>&1 || pdflatex -interaction=nonstopmode main.filled.tex >/dev/null) && echo "paper/main.filled.pdf"

clean-figures:
	rm -f paper/figures/*.pdf paper/figures/*.png paper/figures/*.csv paper/figures/*.md
