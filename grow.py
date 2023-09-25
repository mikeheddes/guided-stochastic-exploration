from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from layers import apply_masks, all_masked_modules, MaskedConv2d, MaskedLinear
from utils import Scores, LossFn, Scope, MaskParamIterable, GrowingName
import utils


@torch.no_grad()
def random(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device,
    subset_fraction: float,
    track_correlation=None
) -> Scores:

    scores = {}

    for _, p in mask_param_pairs:
        scores[id(p)] = torch.rand_like(p)

    return scores


def rigl(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device,
    subset_fraction: float,
    track_correlation=None
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

    # calculate score |g|
    with torch.no_grad():
        for _, p in mask_param_pairs:
            scores[id(p)] = torch.abs(p.grad)

    return scores


def accumulate_init(module):
    if isinstance(module, MaskedConv2d):
        param = module.weight
        in_features = module.in_channels * math.prod(module.kernel_size)

        prob_out = param.new_zeros(module.out_channels)
        prob_in = param.new_zeros(in_features)
        return prob_out, prob_in

    elif isinstance(module, MaskedLinear):
        param = module.weight

        prob_out = param.new_zeros(module.out_features)
        prob_in = param.new_zeros(module.in_features)
        return prob_out, prob_in


def accumulate_forward(module, inputs):
    if isinstance(module, MaskedConv2d):
        patches = F.unfold(
            inputs,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
        )
        # sum activations over all batches and patches
        return torch.sum(patches, dim=(0, 2))

    elif isinstance(module, MaskedLinear):
        # transformers can have multi-dimensional inputs
        # in that case sum over all but the last dimension
        inputs = torch.flatten(inputs, end_dim=-2)
        return torch.sum(inputs, dim=0)


def accumulate_backward(module, inputs):
    if isinstance(module, MaskedConv2d):
        return torch.sum(inputs, dim=(0, 2, 3))

    elif isinstance(module, MaskedLinear):
        # transformers can have multi-dimensional inputs
        # in that case sum over all but the last dimension
        inputs = torch.flatten(inputs, end_dim=-2)
        return torch.sum(inputs, dim=0)


@torch.no_grad()
def sample_gradient_masks(mask_param_pairs, probs_out, probs_in, subset_fraction):
    grad_masks = {}

    for _, param in mask_param_pairs:
        pid = id(param)

        # flatten the params tensor in case of conv layers
        # not the mask because it is all ones during the grow step
        p = torch.flatten(param, start_dim=1)

        num_nonzero = p.nonzero().size(0)
        sparsity = 1 - num_nonzero / p.numel()

        if sparsity <= 0.5:
            grad_mask = torch.ones_like(p, dtype=torch.bool)
            grad_masks[pid] = grad_mask.reshape_as(param)
            continue

        subset_size = int(num_nonzero * subset_fraction)
        subset = utils.sample(probs_out[pid], probs_in[pid], subset_size)
        to_nodes, from_nodes = subset.transpose(0, 1)

        grad_mask = torch.zeros_like(p, dtype=torch.bool)
        grad_mask[to_nodes, from_nodes] = True
        grad_masks[pid] = grad_mask.reshape_as(param)

    return grad_masks


def rigl_random(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device,
    subset_fraction: float,
    track_correlation=None
) -> Scores:

    model.zero_grad()
    model.train()

    # compute the masks for the gradient subsets, in the production system
    # this should be integrated into the gradient computation
    probs_out = {id(p): p.new_ones(p.size(0)) for _, p in mask_param_pairs}
    probs_in = {id(p): p.new_ones(
        math.prod(p.shape[1:])) for _, p in mask_param_pairs}
    grad_masks = sample_gradient_masks(
        mask_param_pairs, probs_out, probs_in, subset_fraction)

    # compute gradient
    for inputs, targets in data:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

    if track_correlation is not None:
        scores = {}
        heuristics = {}

        with torch.no_grad():
            for _, p in mask_param_pairs:
                pid = id(p)
                scores[pid] = torch.abs(p.grad)
                heuristics[pid] = probs_out[pid].view(
                    -1, 1) * probs_in[pid].view(1, -1)

            track_correlation(scores, heuristics)

    # calculate score |g|
    scores = {}
    with torch.no_grad():
        for _, p in mask_param_pairs:
            score = torch.abs(p.grad)
            is_sampled = grad_masks[id(p)]
            score[~is_sampled] = float("-inf")
            scores[id(p)] = score

    return scores


def rigl_synflow(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device,
    subset_fraction: float,
    track_correlation=None
) -> Scores:

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

    hooks = []
    probs_in = {}
    probs_out = {}
    # accumulate the input and output activity of each layer
    for module in all_masked_modules(model):
        prob_out, prob_in = accumulate_init(module)
        probs_out[id(module.weight)] = prob_out
        probs_in[id(module.weight)] = prob_in

        def forward_hook(module, inputs, outputs):
            val = accumulate_forward(module, inputs[0])
            probs_in[id(module.weight)] += val

        def backward_hook(module, grad_inputs, grad_outputs):
            val = accumulate_backward(module, grad_outputs[0])
            probs_out[id(module.weight)] += val

        hooks.append(module.register_forward_hook(forward_hook))
        hooks.append(module.register_full_backward_hook(backward_hook))

    inputs, _ = next(iter(data))
    input_dim = list(inputs.shape)
    input_dim[0] = 1  # set batch size to 1
    inputs = torch.ones(input_dim, dtype=torch.double, device=device)

    logits = model(inputs)
    logit_sum = torch.sum(logits)
    # scale the output of the model to be 1 so the gradients don't explode.
    logit_sum.div(logit_sum.detach()).backward()

    # compute the masks for the gradient subsets, in the production system
    # this should be integrated into the gradient computation
    grad_masks = sample_gradient_masks(
        mask_param_pairs, probs_out, probs_in, subset_fraction)

    # since this implementation still computes the gradients w.r.t. all the weights
    # we don't need to recompute the gradients

    for hook in hooks:
        hook.remove()

    model.float()

    # revert the model to nonlinearized parameters
    with torch.no_grad():
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    revert_model()
    model.train()
    model.zero_grad()

    # compute the gradients
    for inputs, targets in data:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

    if track_correlation is not None:
        scores = {}
        heuristics = {}

        with torch.no_grad():
            for _, p in mask_param_pairs:
                pid = id(p)
                scores[pid] = torch.abs(p.grad)
                heuristics[pid] = probs_out[pid].view(
                    -1, 1) * probs_in[pid].view(1, -1)

            track_correlation(scores, heuristics)

    scores = {}
    with torch.no_grad():
        for _, p in mask_param_pairs:
            score = torch.abs(p.grad)
            is_sampled = grad_masks[id(p)]
            score[~is_sampled] = float("-inf")
            scores[id(p)] = score

    return scores


def rigl_grabo(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device,
    subset_fraction: float,
    track_correlation=None
) -> Scores:

    model.zero_grad()
    model.train()

    hooks = []
    probs_in = {}
    probs_out = {}
    # accumulate the input and output activity of each layer
    for module in all_masked_modules(model):
        prob_out, prob_in = accumulate_init(module)
        probs_out[id(module.weight)] = prob_out
        probs_in[id(module.weight)] = prob_in

        def forward_hook(module, inputs, outputs):
            val = accumulate_forward(module, inputs[0].abs())
            probs_in[id(module.weight)] += val

        def backward_hook(module, grad_inputs, grad_outputs):
            val = accumulate_backward(module, grad_outputs[0].abs())
            probs_out[id(module.weight)] += val

        hooks.append(module.register_forward_hook(forward_hook))
        hooks.append(module.register_full_backward_hook(backward_hook))

    # compute edge sampling probabilities
    for inputs, targets in data:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

    # compute the masks for the gradient subsets, in the production system
    # this should be integrated into the gradient computation
    grad_masks = sample_gradient_masks(
        mask_param_pairs, probs_out, probs_in, subset_fraction)

    # since this implementation still computes the gradients w.r.t. all the weights
    # we don't need to recompute the gradients

    for hook in hooks:
        hook.remove()

    if track_correlation is not None:
        scores = {}
        heuristics = {}

        with torch.no_grad():
            for _, p in mask_param_pairs:
                pid = id(p)
                scores[pid] = torch.abs(p.grad)
                heuristics[pid] = probs_out[pid].view(
                    -1, 1) * probs_in[pid].view(1, -1)

            track_correlation(scores, heuristics)

    scores = {}
    with torch.no_grad():
        for _, p in mask_param_pairs:
            score = torch.abs(p.grad)
            is_sampled = grad_masks[id(p)]
            score[~is_sampled] = float("-inf")
            scores[id(p)] = score

    return scores


def rigl_ams(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device,
    subset_fraction: float,
    track_correlation=None
) -> Scores:

    model.zero_grad()
    model.train()

    hooks = []
    probs_in = {}
    probs_out = {}
    # accumulate the input and output activity of each layer
    for module in all_masked_modules(model):
        prob_out, prob_in = accumulate_init(module)
        probs_out[id(module.weight)] = prob_out
        probs_in[id(module.weight)] = prob_in

        module.sign_hash = utils.FourWayIndependent(1).to(module.weight.device)

        def forward_hook(module, inputs, outputs):
            device = inputs[0].device

            if isinstance(module, MaskedConv2d):
                # extract all the patches the "MLP" would be applied to
                patches = F.unfold(
                    inputs[0],
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                )
                # reshape to (batch_size * num_patches, C * K * K)
                patches = patches.permute(0, 2, 1).flatten(end_dim=1)

                batch_size = patches.size(0)
                batch_range = torch.arange(batch_size, device=device)
                signs = module.sign_hash(batch_range.unsqueeze(-1))
                signs = (signs % 2) * 2 - 1

                val = torch.sum(patches * signs, dim=0)
                probs_in[id(module.weight)] += val

            elif isinstance(module, MaskedLinear):
                inputs = torch.flatten(inputs[0], end_dim=-2)
                batch_size = inputs.size(0)
                batch_range = torch.arange(batch_size, device=device)
                signs = module.sign_hash(batch_range.unsqueeze(-1))
                signs = (signs % 2) * 2 - 1

                val = torch.sum(inputs * signs, dim=0)
                probs_in[id(module.weight)] += val

        def backward_hook(module, grad_inputs, grad_outputs):
            device = grad_outputs[0].device

            if isinstance(module, MaskedConv2d):
                # reshape to (batch_size, H, W, C)
                grad_outputs = grad_outputs[0].permute(0, 2, 3, 1)
                # reshape to (batch_size * H * W, C)
                grad_outputs = grad_outputs.reshape(-1, module.out_channels)

                batch_size = grad_outputs.size(0)
                batch_range = torch.arange(batch_size, device=device)
                signs = module.sign_hash(batch_range.unsqueeze(-1))
                signs = (signs % 2) * 2 - 1

                val = torch.sum(grad_outputs[0] * signs, dim=0)
                probs_out[id(module.weight)] += val

            elif isinstance(module, MaskedLinear):
                grad_outputs = torch.flatten(grad_outputs[0], end_dim=-2)
                batch_size = grad_outputs.size(0)
                batch_range = torch.arange(batch_size, device=device)
                signs = module.sign_hash(batch_range.unsqueeze(-1))
                signs = (signs % 2) * 2 - 1

                val = torch.sum(grad_outputs * signs, dim=0)
                probs_out[id(module.weight)] += val

        hooks.append(module.register_forward_hook(forward_hook))
        hooks.append(module.register_full_backward_hook(backward_hook))

    # compute edge sampling probabilities
    for inputs, targets in data:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

    with torch.no_grad():
        for key in probs_out.keys():
            probs_out[key].abs_()

        for key in probs_in.keys():
            probs_in[key].abs_()

    # compute the masks for the gradient subsets, in the production system
    # this should be integrated into the gradient computation
    grad_masks = sample_gradient_masks(
        mask_param_pairs, probs_out, probs_in, subset_fraction)

    # since this implementation still computes the gradients w.r.t. all the weights
    # we don't need to recompute the gradients

    for hook in hooks:
        hook.remove()

    if track_correlation is not None:
        scores = {}
        heuristics = {}

        with torch.no_grad():
            for _, p in mask_param_pairs:
                pid = id(p)
                scores[pid] = torch.abs(p.grad)
                heuristics[pid] = probs_out[pid].view(
                    -1, 1) * probs_in[pid].view(1, -1)

            track_correlation(scores, heuristics)

    scores = {}
    with torch.no_grad():
        for _, p in mask_param_pairs:
            score = torch.abs(p.grad)
            is_sampled = grad_masks[id(p)]
            score[~is_sampled] = float("-inf")
            scores[id(p)] = score

    return scores


score_fn_by_name = {
    "random": random,
    "rigl": rigl,
    "rigl-random": rigl_random,
    "rigl-synflow": rigl_synflow,
    "rigl-grabo": rigl_grabo,
    "rigl-ams": rigl_ams,
}


def update_masks_globally(mask_param_pairs: MaskParamIterable, scores: Scores, sparsity: float) -> None:

    all_scores = torch.cat([v.ravel() for v in scores.values()])

    num_nonzero = 0
    for mask, _ in mask_param_pairs:
        num_nonzero += mask.sum().item()

    num_grow_limit = torch.isfinite(all_scores).sum().item()
    limit_num_nonzero = num_nonzero + num_grow_limit
    target_num_nonzero = all_scores.numel() * (1 - sparsity)
    target_num_nonzero = int(min(target_num_nonzero, limit_num_nonzero))

    grown_sets = {}

    # if the target sparsity is already reached, do nothing
    if target_num_nonzero <= num_nonzero:
        return grown_sets

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

        grown_sets[id(param)] = prev_mask != mask
        start = end

    return grown_sets


def update_masks_locally(mask_param_pairs: MaskParamIterable, scores: Scores, sparsities: dict[int, float]) -> None:

    grown_sets = {}

    for mask, param in mask_param_pairs:
        pid = id(param)
        score = scores[pid].ravel()
        sparsity = sparsities[pid]

        num_nonzero = mask.sum().item()
        num_grow_limit = torch.isfinite(score).sum().item()
        limit_num_nonzero = num_nonzero + num_grow_limit
        target_num_nonzero = mask.numel() * (1 - sparsity)
        target_num_nonzero = int(min(target_num_nonzero, limit_num_nonzero))

        # if the target sparsity is already reached, do nothing
        if target_num_nonzero <= num_nonzero:
            continue

        prev_mask = mask.clone()

        # set the topk largest scores to one, i.e., keep the topk parameters
        _, indices = score.topk(target_num_nonzero, largest=True)
        mask.zero_()
        mask.view(-1)[indices] = 1.0

        grown_sets[id(param)] = prev_mask != mask

    return grown_sets


def compute_grow_score(
    model: nn.Module,
    mask_param_pairs: MaskParamIterable,
    method: GrowingName,
    subset_fraction: float,
    loss_fn: LossFn,
    data: DataLoader,
    device: torch.device,
    track_correlation=None,
) -> Scores:

    masks = {}
    # save the original masks and set all masks to one so that the gradients are computed for all parameters
    with torch.no_grad():
        for mask, param in mask_param_pairs:
            param.mul_(mask)  # apply the mask to the parameters
            masks[id(param)] = mask.clone()
            mask.copy_(torch.ones_like(mask))

    score_fn = score_fn_by_name[method]
    scores = score_fn(
        model,
        mask_param_pairs,
        loss_fn,
        data,
        device,
        subset_fraction,
        track_correlation,
    )

    with torch.no_grad():
        # revert the masks to what they were before
        for mask, param in mask_param_pairs:
            mask.copy_(masks[id(param)])

        # set active connections to maximum value such that only inactive connections are grown
        for mask, params in mask_param_pairs:
            scores[id(params)][mask == 1.0] = float("inf")

    return scores


def disable_pruned_connections(scores: Scores, pruned_sets) -> None:
    # ignore those parameters that were pruned in this iteration
    for key in scores.keys():
        if key not in pruned_sets:
            continue
        # set their score to -inf
        scores[key][pruned_sets[key]] = float("-inf")


def get_grow_sparsity(grow_rate, sparsity, sparsity_by_layer, scope):
    if scope == "global":
        return 1 - (1 - sparsity) * (1 + grow_rate)

    elif scope == "local":
        sparsities = {}

        for k, v in sparsity_by_layer.items():
            sparsities[k] = 1 - (1 - v) * (1 + grow_rate)

        return sparsities


@torch.no_grad()
def reset_momentum(optimizer, grown_sets):
    if isinstance(optimizer, torch.optim.SGD):
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) not in grown_sets:
                    continue

                if p not in optimizer.state:
                    continue

                grew = grown_sets[id(p)]
                buffer = optimizer.state[p]["momentum_buffer"]
                buffer[grew] = 0.0

    else:
        raise NotImplementedError


def apply_growing(
    scores: Scores,
    mask_param_pairs: MaskParamIterable,
    sparsity: Union[float, dict[int, float]],
    scope: Scope,
) -> dict[int, torch.Tensor]:

    # apply the mask such that all grown connections are initialized at zero
    apply_masks(mask_param_pairs)

    if scope == "global":
        assert type(sparsity) == float
        assert sparsity >= 0.0 and sparsity < 1.0
        return update_masks_globally(mask_param_pairs, scores, sparsity)

    elif scope == "local":
        for s in sparsity.values():
            assert s >= 0.0 and s < 1.0
        return update_masks_locally(mask_param_pairs, scores, sparsity)
