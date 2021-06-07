# HCA-Net/HMAG-QNet

This repository contains one of the projects I did during my PhD in Artificial Intelligence Applied to Automation and Robotics.  The idea was to propose a neural network architecture for cooperative multi-agent systems with heterogeneous agents, in which agents could be grouped into classes of homogeneous agents and policies as well as communication protocols could be learned faster by doing so.

- HMAG-QNet stands for __Heterogeneous Multi-Agent Graph Q-Network__. Paper and code information is available in the [v8 tag](https://github.com/douglasrizzo/hcanet/releases/tag/v8).
- HCA-Net stands for __Neural Network for Heterogeneous Communicative Agents__. Paper and code information is available in the [v9 tag](https://github.com/douglasrizzo/hcanet/releases/tag/v9).

Dependencies are listed inside `setup.py`. You can choose to install PyTorch and PyTorch Geometric before this package and configure GPU support.

To install, run:

```sh
pip install .
```

## Training time

On a PC with the following specs:

- 16~32 GB RAM
- i7 7th generation
- NVIDIA GTX 1070 GPU

It takes anywhere from 4h30 to 14h to train a neural network for 1 million to 1.5 million steps. In some cases, training has to run for around 5 to 6 million steps for some models to learn (see paper referenced in the [v9 tag](https://github.com/douglasrizzo/hcanet/releases/tag/v9)).

## Weights & Biases

I log my experiments on [Weights & Biases](https://wandb.ai/). I don't know how this code behaves if ran on another account or if W&B is not installed or not logged in.

## Maps I decide to use

- `3m` - a scenario with a single type of agent, to test all networks.
- `3s5z` - an easy scenario, but there is still room for VDN to improve.
- `MMM2` - an asymmetric heterogeneous environment in which VDN fails to win.
- `1c3s5z` - an easy scenario with high heterogeneity.
- `MMM` - a symmetric heterogeneous environment, should be easier than MMM2.
