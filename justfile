default:
    just -l

lint:
    ruff check --fix
    ruff format

run:
    ./generative_models/run.sh -b

generate-circle-dataset:
    python -m generative_models.datasets.generate