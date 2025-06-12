default:
    just -l

lint:
    ruff check --fix
    ruff format

run:
    ./generative_models/run.sh