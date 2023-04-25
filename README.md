# Conservative Agency

Reproduction of experiments from the [Conservative Agency paper](https://arxiv.org/abs/1902.09725).

This is an interesting paper, but I also had a few quibbles with it. I wanted to reproduce it to make sure I understood it, and then do a few further experiments.

Code for the original paper is [available here](https://github.com/alexander-turner/attainable-utility-preservation). It's based around an old DeepMind library, so I modified it to use [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) instead.

## Goals

1. reproduce the experiments in the original paper
2. develop a new AUP algorithm that _actually tries_ in the *survival* experiment while still maintaining off-switch functionality
3. support future investigations into combining AUP costs with the Kelly-criterion

## Setup

```
pip install -r requirements.txt
```

## Experiments

To produce gifs of representative runs through each game: `python -m experiments.ablation`

To produce traning charts: `python -m experiments.charts`

## TODO:

1. [X] set up environment
2. [X] make one environment
3. [X] set up rewards in environment
4. [X] train RL agent to solve environment
5. [X] make other environments
6. [X] reproduce plots from paper
7. [ ] fix _even trying_ in off-switch game
8. [ ] formalize AU as a "cost", use a kelly regularization factor in reward
