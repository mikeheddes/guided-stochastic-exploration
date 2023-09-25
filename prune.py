# Adapted from: https://github.com/ganguli-lab/Synaptic-Flow/blob/master/Pruners/pruners.py
from typing import Optional
import math
import torch
import torch.nn as nn
from torch import autograd
from torch.utils.data import DataLoader

from utils import Scores, LossFn, Scope, MaskParamIterable, PruningName
import utils


@torch.no_grad()
def random(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device
) -> Scores:

    scores = {}

    for _, p in mask_param_pairs:
        scores[id(p)] = torch.rand_like(p)

    return scores


@torch.no_grad()
def magnitude(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: Optional[LossFn] = None,
    data: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
) -> Scores:

    scores = {}

    for _, p in mask_param_pairs:
        scores[id(p)] = p.abs()

    return scores


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
def snip(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device
) -> Scores:

    scores = {}
    model.zero_grad()
    model.train()

    # compute gradient
    for inputs, targets in data:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

    # calculate score |g * theta|
    with torch.no_grad():
        for m, p in mask_param_pairs:
            scores[id(p)] = torch.abs(p.grad * p)

    return scores


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
def grasp(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device
) -> Scores:

    temp = 200

    scores = {}
    params = [p for (_, p) in mask_param_pairs]
    model.zero_grad()
    model.train()

    # first gradient vector without computational graph
    stopped_grads = 0
    for inputs, targets in data:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs) / temp
        loss = loss_fn(logits, targets)

        grads = autograd.grad(loss, params, create_graph=False)
        flat_grads = torch.cat([g.ravel() for g in grads])
        stopped_grads += flat_grads

    # second gradient vector with computational graph
    for inputs, targets in data:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs) / temp
        loss = loss_fn(logits, targets)

        grads = autograd.grad(loss, params, create_graph=True)
        flat_grads = torch.cat([g.ravel() for g in grads])

        gnorm = torch.sum(stopped_grads * flat_grads)
        gnorm.backward()

    # calculate score Hg * theta (negate to remove top percent)
    with torch.no_grad():
        for _, p in mask_param_pairs:
            scores[id(p)] = p.grad * p

    return scores


def synflow(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device
) -> Scores:

    scores = {}
    signs = {}
    model.zero_grad()

    # NOTE: this function assumes that the activation function is exactly linear in the positive domain

    # linearize the model
    with torch.no_grad():
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()

    revert_model = utils.prepare_synflow_model(model)
    model.double()
    # synflow performs better when the model is in eval mode
    model.eval()

    inputs, _ = next(iter(data))
    input_dim = list(inputs.shape)
    input_dim[0] = 1  # set batch size to 1
    inputs = torch.ones(input_dim, dtype=torch.double, device=device)

    logits = model(inputs)
    logit_sum = torch.sum(logits)
    # scale the output of the model to be 1 such that the gradients don't explode.
    logit_sum.div(logit_sum.detach()).backward()

    with torch.no_grad():
        for _, p in mask_param_pairs:
            scores[id(p)] = torch.abs(p.grad * p)

    revert_model()
    model.float()

    # revert the model to nonlinearized parameters
    with torch.no_grad():
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    return scores


score_fn_by_name = {
    "random": random,
    "magnitude": magnitude,
    "snip": snip,
    "grasp": grasp,
    "synflow": synflow,
}


def update_masks_globally(mask_param_pairs: MaskParamIterable, scores: Scores, sparsity: float) -> None:

    all_scores = torch.cat([v.ravel() for v in scores.values()])

    num_nonzero = 0
    for mask, _ in mask_param_pairs:
        num_nonzero += int(mask.sum().item())

    target_num_nonzero = int(all_scores.numel() * (1 - sparsity))

    pruned_sets = {}

    # if the number of nonzero parameters is already below the target, do nothing
    if num_nonzero <= target_num_nonzero:
        return pruned_sets

    # set the topk largest scores to one, i.e., keep the topk parameters
    _, indices = all_scores.topk(target_num_nonzero, largest=True)

    start = 0
    for mask, param in mask_param_pairs:
        # get the active connections for each layer with a sliding window
        end = start + mask.numel()
        is_local = (indices >= start) & (indices < end)
        local_indices = indices[is_local] - start

        prev_mask = mask.clone()

        # set active connections to 1
        mask.zero_()
        mask.view(-1)[local_indices] = 1.0

        pruned_sets[id(param)] = prev_mask != mask
        start = end

    return pruned_sets


def update_masks_locally(mask_param_pairs: MaskParamIterable, scores: Scores, sparsities: dict[int, float]) -> None:

    pruned_sets = {}

    for mask, param in mask_param_pairs:
        pid = id(param)
        score = scores[pid].ravel()
        sparsity = sparsities[pid]

        num_nonzero = int(mask.sum().item())
        target_num_nonzero = int(mask.numel() * (1 - sparsity))

        # if the number of nonzero parameters is already below the target, do nothing
        if num_nonzero <= target_num_nonzero:
            continue

        prev_mask = mask.clone()

        # set the topk largest scores to one, i.e., keep the topk parameters
        _, indices = score.topk(target_num_nonzero, largest=True)
        mask.zero_()
        mask.view(-1)[indices] = 1.0

        pruned_sets[id(param)] = prev_mask != mask

    return pruned_sets


def disable_grown_connections(scores: Scores, grown_sets) -> None:
    # ignore those parameters that were grown in this iteration
    for key in scores.keys():
        if key not in grown_sets:
            continue

        # set their score to inf
        scores[key][grown_sets[key]] = float("inf")


def get_prune_sparsity(prune_rate, sparsity, sparsity_by_layer, scope):
    if scope == "global":
        return 1 - (1 - sparsity) * (1 - prune_rate)
    
    elif scope == "local":
        sparsities = {}

        for k, v in sparsity_by_layer.items():
            sparsities[k] = 1 - (1 - v) * (1 - prune_rate)

        return sparsities


def compute_prune_score(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    method: PruningName,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device,
) -> Scores:

    score_fn = score_fn_by_name[method]
    scores = score_fn(model, mask_param_pairs, loss_fn, data, device)

    # set inactive connections to minimum value such that only active connections are pruned
    with torch.no_grad():
        for mask, params in mask_param_pairs:
            scores[id(params)][mask == 0.0] = float("-inf")

    return scores


def apply_pruning(
    scores: Scores,
    mask_param_pairs: MaskParamIterable,
    sparsity: float,
    scope: Scope,
) -> dict[int, torch.Tensor]:

    if scope == "global":
        assert type(sparsity) == float
        assert sparsity >= 0.0 and sparsity < 1.0

        if sparsity == 0.0:
            return {}
        
        return update_masks_globally(mask_param_pairs, scores, sparsity)

    elif scope == "local":
        for s in sparsity.values():
            assert s >= 0.0 and s < 1.0
            
        return update_masks_locally(mask_param_pairs, scores, sparsity)
