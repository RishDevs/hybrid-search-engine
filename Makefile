DATASETS ?= scifact nfcorpus fiqa
RUN = uv run

.PHONY: setup test download index tune eval bench route report gate baseline serve all

setup:
	uv sync --extra dev --python 3.11
	$(RUN) hse fix-openmp

test:
	$(RUN) pytest -q

download:
	$(RUN) hse download $(DATASETS) quora

index:
	$(RUN) hse index $(DATASETS)

tune:
	$(RUN) hse tune

eval:
	$(RUN) hse eval $(DATASETS)

bench:
	$(RUN) hse bench --dataset quora-200k

route:
	$(RUN) hse route $(DATASETS)

report:
	$(RUN) hse report

gate:
	$(RUN) hse gate

baseline:
	$(RUN) hse gate --update

serve:
	$(RUN) hse serve --dataset scifact

all: eval bench route report
