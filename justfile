default:
    just -l

lint:
    ruff check --select I --fix .
    ruff format

run:
    ./generative_models/run.sh -b

generate-circle-dataset:
    python -m generative_models.datasets.generate

visualize-circle-dataset:
    python -m streamlit run generative_models/datasets/visualize.py

train:
    python -m generative_models.train

inference TYPE='ddpm':
    python -m generative_models.inference -d {{TYPE}}