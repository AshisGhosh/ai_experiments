# AI Experiments

experiments on toy problems to explore the fundamentals of ml/deep learning/ai models and training strategies.

## purpose

this repository contains implementations of various generative models (diffusion, flow matching, transformers) on simple 2d datasets. the goal is to understand the core principles behind different generative modeling approaches through hands-on experimentation.

## project structure

```
ai_experiments/
├── generative_models/          # main experiment directory
│   ├── checkpoints/           # saved model weights
│   ├── datasets/              # data generation and visualization
│   ├── docker/                # containerization setup
│   ├── evaluators/            # model evaluation and sampling
│   ├── models/                # model implementations
│   ├── objectives/            # loss functions and training objectives
│   └── schedulers/            # noise scheduling for diffusion
├── utils/                     # shared utilities
└── train.py                   # main training script
```

## setup

### local dependencies
```bash
./install.sh  # installs just, linting tools, etc.
```

### run experiments
```bash
just run  # builds and launches docker container
```

once inside the container, use `just` commands for training and inference:
```bash
just train     # run training
just inference # run inference
```

## models

- **diffusion models**: denoising diffusion probabilistic models
- **flow matching**: continuous normalizing flows
- **transformers**: autoregressive sequence modeling

## datasets

- simple 2d geometric shapes (circles, etc.)
- synthetic data for controlled experimentation