# Conservative Agency

Reproduction of experiments from the [Conservative Agency paper](https://arxiv.org/abs/1902.09725).

This is an interesting paper, but I also had a few quibbles with it. I wanted to reproduce it to make sure I understood it, and then do a few further experiments.

## Goals

1. reproduce the experiments in the original paper
2. develop a new AUP algorithm that _actually tries_ in the *survival* experiment while still maintaining off-switch functionality
3. support future investigations into combining AUP costs with the Kelly-criterion

## Setup

* gymnasium: https://github.com/Farama-Foundation/Gymnasium
* https://gymnasium.farama.org/api/registry/


see here for envs: 
* https://github.com/alexander-turner/attainable-utility-preservation/tree/master/ai_safety_gridworlds/environments

## TODO:

1. [X] set up environment
2. [X] make one environment
3. [ ] set up rewards in environment
4. [ ] train RL agent to solve environment
5. [ ] make other environments
6. [ ] reproduce plots from paper
7. [ ] fix _even trying_ in off-switch game
8. [ ] formalize AU as a "cost", use a kelly regularization factor in reward
