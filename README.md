# Always-Sparse Training by Growing Connections with Guided Stochastic Exploration

The required dependiencies are listed in `requirements.txt`.

## Experiments

Our experiments can be run with the following commands:
```bash
# Ours Uniform
python our_training.py --model resnet56 --dataset cifar10 --prune magnitude --grow rigl-random --grow-subset 1.0 --sparsity 0.98

# Ours GraBo
python our_training.py --model resnet56 --dataset cifar10 --prune magnitude --grow rigl-grabo --grow-subset 1.0 --sparsity 0.98

# Ours GraEst
python our_training.py --model resnet56 --dataset cifar10 --prune magnitude --grow rigl-ams --grow-subset 1.0 --sparsity 0.98

```


The experiments for the related work can be run with the following commands:
```bash
# Lottery ticket hypothesis
python lottery_training.py --model resnet56 --dataset cifar10

# Gradual sparse training
python gradual_sparse_training.py --model resnet56 --dataset cifar10 --prune magnitude --sparsity 0.98

# Static random graph
python dynamic_sparse_training.py --model resnet56 --dataset cifar10 --prune none --grow none --sparsity 0.98

# SNIP
python prune_before_training.py --model resnet56 --dataset cifar10 --prune snip --sparsity 0.98

# GraSP
python prune_before_training.py --model resnet56 --dataset cifar10 --prune grasp --sparsity 0.98

# SynFlow
python prune_before_training.py --model resnet56 --dataset cifar10 --prune synflow --sparsity 0.98

# SET
python dynamic_sparse_training.py --model resnet56 --dataset cifar10 --prune magnitude --grow random --sparsity 0.98

# RigL
python dynamic_sparse_training.py --model resnet56 --dataset cifar10 --prune magnitude --grow rigl --sparsity 0.98

```

The following models and datasets are used in the paper results:

| Argument | Options |
| ---- | ---- |
|`--model` | `resnet56`, `vgg16-small`, `simple-vit-tiny`, `resnet50` |
| `--dataset` | `cifar10`, `cifar100`, `imagenet` |
