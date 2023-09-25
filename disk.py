from typing import List, Optional, Callable, Dict, Any
import os
import csv
import time
import random
import string
import torch
import torch.nn as nn

import layers


def random_string(len: int = 7) -> str:
    chars = string.ascii_letters + string.digits
    rand_chars = random.choices(chars, k=len)
    rand_str = "".join(rand_chars)
    return rand_str


class StateSaver:
    def __init__(
        self,
        root: str,
        state_fn: Callable[[], object],
        checkpoints: Optional[List[int]] = [],
        initial: Optional[bool] = True,
        best: Optional[bool] = True,
    ) -> None:

        self.epoch_idx = -1
        self.best_loss = float("inf")
        self.best_model_path = None

        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.state_fn = state_fn
        self.checkpoints = checkpoints
        self.checkpoint_paths = []
        self.save_initial_model = initial
        self.save_best_model = best

        if self.save_initial_model:
            state = self.state_fn()
            filename = f"initial-state-{random_string()}.pt"
            path = os.path.join(self.root, filename)
            torch.save(state, path)

    def step(self, loss: float) -> None:
        self.epoch_idx = self.epoch_idx + 1
        state = None

        if loss < self.best_loss and self.save_best_model:
            if state == None:
                state = self.state_fn()

            self.best_loss = loss
            filename = f"best-state-loss={loss:.7g}-{random_string()}.pt"
            path = os.path.join(self.root, filename)
            torch.save(state, path)

            # remove previous best model
            if self.best_model_path != None:
                os.remove(self.best_model_path)

            self.best_model_path = path

        if self.epoch_idx in self.checkpoints:
            if state == None:
                state = self.state_fn()

            filename = f"checkpoint-state-epoch={self.epoch_idx}-{random_string()}.pt"
            path = os.path.join(self.root, filename)
            torch.save(state, path)
            self.checkpoint_paths.append(path)


def is_empty_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) == 0


class MetricSaver:
    def __init__(self, path: str, fieldnames: List[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames

        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames)

        if is_empty_file(path):
            self.writer.writeheader()

    def write(self, metrics: Dict[str, Any]) -> None:
        self.writer.writerow(metrics)

    def close(self) -> None:
        self.file.close()


class SparsityStatsSaver:
    def __init__(self, path: str, model: nn.Module, epoch_idx: int, step_idx: int) -> None:
        self.model = model

        metrics = self.metrics(epoch_idx, step_idx)
        fieldnames = list(metrics.keys())

        self.metric_writer = MetricSaver(path, fieldnames)
        self.metric_writer.write(metrics)

    def metrics(self, epoch_idx: int, step_idx: int) -> Dict[str, Any]:
        # Track time in training progress
        sparsities = {
            "epoch": epoch_idx,
            "step": step_idx,
            "wall_time": time.time(),
        }

        for name, module in layers.all_named_masked_modules(self.model):
            sparsities[f"layer_active_{name}"] = int(module.mask.sum().item())
            sparsities[f"layer_total_{name}"] = module.weight.numel()
            sparsities[f"layer_sparsity_{name}"] = module.sparsity

        mask_param_pairs = layers.all_mask_param_pairs(self.model)
        active_params, total_params = layers.stats(mask_param_pairs)
        sparsities["model_active"] = active_params
        sparsities["model_total"] = total_params
        sparsities["model_sparsity"] = 1 - active_params / total_params

        return sparsities

    def step(self, epoch_idx: int, step_idx: int) -> None:
        metrics = self.metrics(epoch_idx, step_idx)
        self.metric_writer.write(metrics)


class CorrelationSaver:
    def __init__(self, root: str, model: nn.Module, every: int = 1) -> None:
        self.root = root
        self.model = model
        self.every = every
        self.step_count = -1

    def step(self, epoch_idx: int, step_idx: int, values, heuristic) -> None:
        self.step_count += 1

        # Only save every N steps
        if self.step_count % self.every != 0:
            return

        values_by_name = {}
        heuristic_by_name = {}
    
        for name, module in layers.all_named_masked_modules(self.model):
            pid = id(module.weight)

            if pid not in values:
                continue

            values_by_name[name] = values[pid]
            heuristic_by_name[name] = heuristic[pid]


        path = os.path.join(self.root, f"correlation-step={step_idx}-{random_string()}.pt")
        torch.save({
            "epoch": epoch_idx,
            "step": step_idx,
            "wall_time": time.time(),
            "values": values_by_name,
            "heuristic": heuristic_by_name,
        }, path)
