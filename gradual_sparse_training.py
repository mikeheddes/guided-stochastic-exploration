import os
import math
import time
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from config import CommonOptions, new_result_directory
from datasets import get_dataloader, num_classes_by_dataset, image_size_by_dataset
from models import get_model, get_num_parameters
from disk import StateSaver, SparsityStatsSaver, MetricSaver
from layers import all_mask_param_pairs, sparsify
from utils import PruningName
import prune
import utils


class Options(CommonOptions):
    prune: PruningName
    prune_delta: int = 1000  # in number of steps
    prune_finish: int = 120  # in epochs


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
    sparsify(model, 0.0, "uniform")
    model = model.to(device)
    mask_param_pairs = all_mask_param_pairs(model)

    num_params = get_num_parameters(model)
    print(f"Number of model parameters: {num_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=config.gamma
    )

    # set epoch_idx to -1 because the state_fn requires it to be defined
    epoch_idx = -1
    step_idx = 0
    final_prune_step = config.prune_finish * len(train_loader)

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

    total_prune_grow_samples = config.prune_grow_samples * num_classes
    prune_grow_batches = math.ceil(total_prune_grow_samples / config.train_batch_size)

    for epoch_idx in range(config.epochs):
        print(f"Epoch: {epoch_idx + 1}/{config.epochs}")

        model.train()

        mean_loss = utils.MeanMetric()
        mean_accuracy = utils.MeanMetric()

        with tqdm(total=len(train_loader), unit="batches") as pbar:
            for step_idx, (inputs, targets) in enumerate(train_loader, start=step_idx):
                # Prune step
                is_update_step = (step_idx + 1) % config.prune_delta == 0
                is_before_final_step = step_idx <= final_prune_step
                if is_update_step and is_before_final_step:
                    prune_rate = utils.cosine_annealing(step_idx, final_prune_step, 1.0)
                    sparsity = config.sparsity * (1 - prune_rate)

                    scores = prune.compute_prune_score(
                        model,
                        mask_param_pairs,
                        config.prune,
                        loss_fn,
                        utils.take(train_loader, prune_grow_batches),
                        device,
                    )

                    prune.apply_pruning(scores, mask_param_pairs, sparsity, config.scope)

                    sparsity_saver.step(epoch_idx, step_idx)

                # Training step
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                loss = loss_fn(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
