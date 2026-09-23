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
	@# README figures are copies of the paper figures, so the two can never disagree
	cp paper/figures/paper/fig_decomposition.png docs/assets/error_decomposition.png
	cp paper/figures/paper/fig_paraphrase_heatmap.png docs/assets/paraphrase_heatmap.png
	cp paper/figures/paper/fig_forest_study1.png docs/assets/study1_success_vs_violation.png
	cp paper/figures/paper/fig_forest_study2.png docs/assets/study2_success_vs_violation.png

paper: report
	$(PY) paper/fill.py
	cd paper && (tectonic main.filled.tex >/dev/null 2>&1 || pdflatex -interaction=nonstopmode main.filled.tex >/dev/null) && cp main.filled.pdf LeJudge-natural-language-constraints-for-latent-world-model-planning.pdf && echo "paper/LeJudge-natural-language-constraints-for-latent-world-model-planning.pdf"

clean-figures:
	rm -f paper/figures/*.pdf paper/figures/*.png paper/figures/*.csv paper/figures/*.md
