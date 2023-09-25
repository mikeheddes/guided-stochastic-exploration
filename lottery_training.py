from typing import List
import os
import time
import copy
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from config import CommonOptions, new_result_directory
from datasets import get_dataloader, num_classes_by_dataset, image_size_by_dataset
from models import get_model, get_num_parameters
from disk import StateSaver, SparsityStatsSaver, MetricSaver
from layers import all_named_mask_param_pairs, all_mask_param_pairs, sparsify
import prune
import utils


class Options(CommonOptions):
    # the progressive levels of sparsity to prune
    sparsities: List[float] = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98]


@torch.no_grad()
def lottery_process(trained_model: nn.Module, init_model: nn.Module, sparsity: float, scope, device) -> nn.Module:
    """Prune the trained model and assign the initial (untrained) weights to the model but keep the masks"""

    mask_param_pairs = all_mask_param_pairs(trained_model)
    scores = prune.compute_prune_score(
        trained_model,
        mask_param_pairs,
        "magnitude",
        None,
        None,
        device,
    )

    prune.apply_pruning(scores, mask_param_pairs, sparsity, scope)

    masks = {}
    for name, mask, _ in all_named_mask_param_pairs(trained_model):
        # clone because otherwise the values will be overwritten by loading the
        # parameters of the initial model in the next step
        masks[name] = mask.clone()

    # copy all the parameters from the initial model to the trained model
    trained_model.load_state_dict(init_model.state_dict())

    # copy the masks back to the trained model
    for name, mask, _ in all_named_mask_param_pairs(trained_model):
        mask.copy_(masks[name])

    return trained_model


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
        device=device
    )

    num_classes = num_classes_by_dataset[config.dataset]
    image_size = image_size_by_dataset[config.dataset]
    # the first model is trained without sparsity (dense)
    model = get_model(config.model, image_size, num_classes)
    sparsify(model, 0.0, "uniform")
    model = model.to(device)

    # keep copy of initial weights
    model_init = copy.deepcopy(model)

    num_params = get_num_parameters(model)
    print(f"Number of model parameters: {num_params:,}")

    metrics_path = os.path.join(result_dir, "metrics.csv")
    metric_fieldnames = ["epoch", "step", "wall_time", "sparsity",
                         "train_loss", "train_acc", "test_loss", "test_acc"]
    metric_saver = MetricSaver(metrics_path, metric_fieldnames)

    sparsity_path = os.path.join(result_dir, "sparsity.csv")
    sparsity_saver = SparsityStatsSaver(
        sparsity_path, model, epoch_idx=-1, step_idx=-1)

    for sparsity in config.sparsities:
        # set epoch_idx to -1 because the state_fn requires it to be defined
        epoch_idx = -1

        lottery_process(model, model_init, sparsity, config.scope, device)
        sparsity_saver.step(epoch_idx=-1, step_idx=-1)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma
        )

        def state_fn():
            return {
                "epoch": epoch_idx,
                "wall_time": time.time(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

        state_saver = StateSaver(
            os.path.join(result_dir, f"sparsity-{sparsity}"),
            state_fn,
            config.checkpoints,
            config.save_initial,
            config.save_best
        )

        for epoch_idx in range(config.epochs):
            print(f"Epoch: {epoch_idx + 1}/{config.epochs}")

            train_metrics = utils.train(
                model, optimizer, loss_fn, train_loader, device)

            test_metrics = utils.evaluate(model, loss_fn, test_loader, device)

            metric_saver.write({
                "epoch": epoch_idx,
                "step": (epoch_idx + 1) * len(train_loader),
                "wall_time": time.time(),
                "sparsity": sparsity,
                "train_loss": train_metrics[0],
                "train_acc": train_metrics[1],
                "test_loss": test_metrics[0],
                "test_acc": test_metrics[1],
            })

            scheduler.step()
            state_saver.step(test_metrics[0])


if __name__ == "__main__":
    main()
