# Based on: https://github.com/A-Klass/torch_topkast

import os
import time
import math
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from config import CommonOptions, new_result_directory
from datasets import get_dataloader, num_classes_by_dataset, image_size_by_dataset
from models import get_model, get_num_parameters
from disk import StateSaver, SparsityStatsSaver, MetricSaver
from utils import SparsityAssignment

import utils

from layers import MaskedLinear, MaskedConv2d, get_num_features, get_active_param_scaler


class Options(CommonOptions):
    sparsity_assignment: SparsityAssignment = "erdos-renyi"
    grow_subset: float = 1.0  # relative to the sparsity (or density)
    prune_grow_delta: int = 1000  # in number of steps
    num_epochs_explore: int = 20  # typically 10% of epochs



def sparsify(model: nn.Module, sparsity: float, subset_size: float, method: SparsityAssignment):
    """
    assign each layer's sparsity based on the erdos-renyi model
    that is, proportional to the sum of in and out features of a layer
    or uniform across all layers
    """

    scale = get_active_param_scaler(model, sparsity, method)

    # collect but do not apply the modifications
    # this is done to avoid mutating the model while iterating over it
    modifications = []
    sparsity_by_layer = {}
    for parent in model.modules():
        for name, m in parent.named_children():
            if not isinstance(m, (nn.Linear, nn.Conv2d)):
                continue

            in_f, out_f = get_num_features(m.weight)

            if method == "erdos-renyi":
                layer_sparsity = 1 - scale * (in_f + out_f) / (in_f * out_f)

            elif method == "scaled":
                layer_sparsity = 1 - scale * out_f / (in_f * out_f)

            elif method == "uniform":
                layer_sparsity = 1 - scale

            if isinstance(m, nn.Linear):

                if layer_sparsity < 0 or layer_sparsity > 1:
                    print(
                        f"WARNING: Layer sparsity must be between 0 and 1 but got {layer_sparsity}, clipping has been applied. This could happen if the sparsity is low and the parameters are unevenly distributed over the layers.")
                    layer_sparsity = min(max(layer_sparsity, 0.0), 1.0)

                sparse_module = MaskedLinear(
                    m.in_features,
                    m.out_features,
                    torch.is_tensor(m.bias),
                    m.weight.device,
                    m.weight.dtype
                )

                sparsity_by_layer[id(sparse_module.weight)] = layer_sparsity

                sparse_module.register_buffer("all_weight", sparse_module.weight.detach().clone().requires_grad_(False))
                sparse_module.p_forward = layer_sparsity
                sparse_module.p_backward = max(1 - ((1 - layer_sparsity) * (1 + subset_size)), 0)
                update_active_param_set(sparse_module)
                modifications.append((parent, name, sparse_module))

            if isinstance(m, nn.Conv2d):

                if layer_sparsity < 0 or layer_sparsity > 1:
                    print(
                        f"WARNING: Layer sparsity must be between 0 and 1 but got {layer_sparsity}, clipping has been applied. This could happen if the sparsity is low and the parameters are unevenly distributed over the layers.")
                    layer_sparsity = min(max(layer_sparsity, 0.0), 1.0)

                sparse_module = MaskedConv2d(
                    in_channels=m.in_channels,
                    out_channels=m.out_channels,
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=torch.is_tensor(m.bias),
                    padding_mode=m.padding_mode,
                    device=m.weight.device,
                    dtype=m.weight.dtype
                )

                sparsity_by_layer[id(sparse_module.weight)] = layer_sparsity
                
                sparse_module.register_buffer("all_weight", sparse_module.weight.detach().clone().requires_grad_(False))
                sparse_module.p_forward = layer_sparsity
                sparse_module.p_backward = max(1 - ((1 - layer_sparsity) * (1 + subset_size)), 0)
                update_active_param_set(sparse_module)
                modifications.append((parent, name, sparse_module))

    # apply the modifications to the model
    for parent, name, masked_module in modifications:
        setattr(parent, name, masked_module)

    return sparsity_by_layer

                
# Masking operations
def compute_mask(matrix, sparsity: float):
    """
    Get the indices of `matrix` values that belong to
    the `p` biggest absolute values in this matrix
    (as in: top 1 % of the layer, by weight norm).
    In the paper this refers to D or D+M, respectively.
    
    Args:
        matrix (torch.Tensor): weight matrix
        p(float): self.p_forward; p-quantile
        
    Returns:
        mask as torch.Tensor tuple containing indices of
        matrix's biggest values
    """
    threshold = torch.quantile(torch.abs(matrix), sparsity)
    return torch.abs(matrix) >= threshold


def compute_just_bwd(mask_fwd, mask_bwd):
    """
    Compute set difference between forward set (A) and backward set (B).
    Supposed to be called within update_active_param_set() which 
    sets the indices by computing the mask for self.idx_fwd and 
    self.idx_bwd, thus creating self.idx_fwd and self.idx_bwd
    
    Input:
        The mask from compute_mask(matrix, K)
        
    Returns:
        torch.Tensor containing indices of B\A
    """
    
    return (mask_bwd.long() - mask_fwd.long()) == 1


# Update step for active set
def update_active_param_set(layer) -> None:
    """
    Updates the dense (complete) weight tensor with 
    newly learned weights from B.
    Computes the masks to get the subsets of active parameters
    (sets A, B, and B\A) in terms of indices. 
    """
    # when not calling for first time, then update 
    # all parameters affected in the backward pass
    if hasattr(layer, "mask_fwd"):
        mask_fwd = layer.mask_fwd
        layer.all_weight[mask_fwd] = layer.weight.detach()[mask_fwd]       
        
    mask_fwd = compute_mask(layer.all_weight, layer.p_forward)
    mask_bwd = compute_mask(layer.all_weight, layer.p_backward)
    mask_justbwd = compute_just_bwd(mask_fwd, mask_bwd)

    layer.mask.copy_(mask_bwd)

    with torch.no_grad():
        layer.weight[mask_justbwd] = 0

    layer.mask_fwd = mask_fwd
    layer.mask_bwd = mask_bwd
    layer.mask_justbwd = mask_justbwd      


def _update_sparse_layers(model):
    """
    Goes through all sparse TopKast layers
    and updates the active parameter set, accordingly.
    
    This is supposed to be called by `_burn_in()` and 
    `_update_periodically()`
    """
    for layer in model.modules():
        if isinstance(layer, (MaskedLinear, MaskedConv2d)):
            update_active_param_set(layer)


def reset_justbwd_weights(model) -> None:
    """
    Wrapper function for resetting B\A to zeros, since optim.step makes 
    these parameters nonzero.

    Updates weight matrix for B\A and resets the corresponding weights in 
    the active_fwd_weights.
    """
    for layer in model.modules():
        if isinstance(layer, (MaskedLinear, MaskedConv2d)):
            all_weight = layer.all_weight
            weight = layer.weight
            mask_justbwd = layer.mask_justbwd

            all_weight[mask_justbwd] += weight.detach()[mask_justbwd]

            with torch.no_grad():
                weight[mask_justbwd] = 0


def weight_decay_loss(model):

    loss = 0
    for layer in model.modules():

        if isinstance(layer, (MaskedLinear, MaskedConv2d)):
            loss += torch.linalg.norm(layer.weight[layer.mask_fwd])

            layer.weight.data[layer.mask_justbwd] = layer.all_weight[layer.mask_justbwd]
            loss += torch.linalg.norm(layer.weight[layer.mask_justbwd]) / (1 - layer.p_forward)

            layer.weight.data[layer.mask_justbwd] = 0.

        else:
            for name in layer._parameters.keys():
                if name != 'weight': continue
                loss += torch.linalg.norm(layer._parameters[name])

    return loss


def main():
    config = Options(underscores_to_dashes=True).parse_args()

    # Only set the torch and numpy random seed because the python random package
    # still need to return random numbers between runs for file naming.
    if config.seed is not None:
        torch.manual_seed(config.seed)
        numpy.random.seed(config.seed)

    result_dir = new_result_directory(config.result_dir)
    config_path = os.path.join(result_dir, "config.json")
    config.save(config_path)
    print(f"Storing artifacts in: {result_dir}")

    device = config.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloader(
        config.dataset,
        config.dataset_dir,
        config.train_batch_size,
        config.eval_batch_size,
        config.workers,
        config.download,
        device=device,
    )

    num_classes = num_classes_by_dataset[config.dataset]
    image_size = image_size_by_dataset[config.dataset]
    model = get_model(config.model, image_size, num_classes)
    
    sparsify(model, config.sparsity, config.grow_subset, config.sparsity_assignment)
    model = model.to(device)

    num_params = get_num_parameters(model)
    print(f"Number of model parameters: {num_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        config.learning_rate,
        momentum=config.momentum,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=config.gamma
    )

    # set epoch_idx to -1 because the state_fn requires it to be defined
    epoch_idx = -1
    step_idx = 0

    def state_fn():
        return {
            "epoch": epoch_idx,
            "step": step_idx,
            "wall_time": time.time(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

    state_saver = StateSaver(
        result_dir, state_fn, config.checkpoints, config.save_initial, config.save_best
    )

    metrics_path = os.path.join(result_dir, "metrics.csv")
    metric_fieldnames = [
        "epoch",
        "step",
        "wall_time",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
    ]
    metric_saver = MetricSaver(metrics_path, metric_fieldnames)

    sparsity_path = os.path.join(result_dir, "sparsity.csv")
    sparsity_saver = SparsityStatsSaver(sparsity_path, model, epoch_idx=-1, step_idx=-1)

    for epoch_idx in range(config.epochs):
        print(f"Epoch: {epoch_idx + 1}/{config.epochs}")

        mean_loss = utils.MeanMetric()
        mean_accuracy = utils.MeanMetric()

        with tqdm(total=len(train_loader), unit="batches") as pbar:
            for step_idx, (inputs, targets) in enumerate(train_loader, start=step_idx):
                # Prune grow step
                is_update_step = (step_idx + 1) % config.prune_grow_delta == 0
                is_exploring = epoch_idx <= config.num_epochs_explore

                if is_exploring or is_update_step:

                    _update_sparse_layers(model)
                    sparsity_saver.step(epoch_idx, step_idx)

                # Training step
                model.train()
                
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                loss = loss_fn(logits, targets) + config.weight_decay * weight_decay_loss(model)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                reset_justbwd_weights(model)

                with torch.no_grad():
                    batch_size = inputs.size(0)
                    accuracy = utils.calc_accuracy(logits, targets)
                    mean_loss.update(loss.cpu(), weight=batch_size)
                    mean_accuracy.update(accuracy.cpu(), weight=batch_size)

                    pbar.update(1)
                    pbar.set_postfix(
                        loss=mean_loss.compute().item(),
                        acc=mean_accuracy.compute().item() * 100,
                    )

        train_metrics = (mean_loss.compute().item(), mean_accuracy.compute().item())

        test_metrics = utils.evaluate(model, loss_fn, test_loader, device)

        metric_saver.write(
            {
                "epoch": epoch_idx,
                "step": step_idx,
                "wall_time": time.time(),
                "train_loss": train_metrics[0],
                "train_acc": train_metrics[1],
                "test_loss": test_metrics[0],
                "test_acc": test_metrics[1],
            }
        )

        scheduler.step()
        state_saver.step(test_metrics[0])


if __name__ == "__main__":
    main()
