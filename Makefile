.PHONY: setup run quick distilbert score-mind test

setup:
	pip install -r requirements.txt

run:
	python scripts/run_all.py

quick:
	python scripts/run_all.py --quick --out results_quick --models models_quick

distilbert:
	pip install transformers && python -m quality.finetune --out results/distilbert

# Writes quality scores for the content-recsys repo (expects it cloned next to this one)
score-mind:
	python scripts/score_mind.py \
		--news ../content-recsys/data/MINDsmall/train/news.tsv ../content-recsys/data/MINDsmall/dev/news.tsv \
		--out ../content-recsys/data/mind_quality.tsv

test:
	python -m pytest -q
