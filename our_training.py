import os
import math
import torch
from torch import Tensor

from training import Config as BaseConfig, Runner as BaseRunner
from models import get_model
from disk import SparsityStatsSaver, CorrelationSaver
from layers import all_mask_param_pairs, sparsify
from utils import Scope, PruningName, GrowingName, SparsityAssignment
import utils
import prune
import grow


# TODO: when enabling channels last, all tensors should be initialized with channels last


class Config(BaseConfig):
    prune: PruningName
    grow: GrowingName
    sparsity: float
    # the scope of the pruning/growing process
    scope: Scope = "global"
    # how many samples of train data to use per class for pruning/growing
    prune_grow_samples: int = 10
    grow_subset: float = 1.0  # relative to the sparsity (or density)
    sparsity_assignment: SparsityAssignment = "erdos-renyi"
    prune_grow_rate: float = 0.2  # fraction of active parameters to prune/grow
    prune_grow_delta: int = 1000  # in number of steps
    prune_grow_finish: int = 120  # in epochs
    track_correlation_freq: int = -1  # in prune-grow steps, -1 to disable


class Runner(BaseRunner):
    config: Config

    def __init__(self, config: Config):
        super().__init__(config)

        self.mask_param_pairs = all_mask_param_pairs(self.model)

        self.final_prune_grow_step = self.config.prune_grow_finish * \
            len(self.train_loader)

        sparsity_path = os.path.join(self.artifact_dir, "sparsity.csv")
        self.sparsity_saver = SparsityStatsSaver(
            sparsity_path, 
            self.model, 
            epoch_idx=-1, 
            step_idx=-1
        )

        if self.config.track_correlation_freq > 0:
            self.correlation_saver = CorrelationSaver(
                self.artifact_dir, 
                self.model, 
                self.config.track_correlation_freq
            )

            def report_correlation(values, heuristics):
                self.correlation_saver.step(
                    self.epoch_idx, 
                    self.step_idx, 
                    values, 
                    heuristics
                )
        else:
            report_correlation = None

        self.report_correlation = report_correlation

        total_prune_grow_samples = self.config.prune_grow_samples * self.num_classes
        self.prune_grow_batches = math.ceil(
            total_prune_grow_samples / self.config.train_batch_size)

    def init_model(self):
        model = get_model(
            self.config.model,
            self.image_size,
            self.num_classes,
            self.config.model_width
        )

        self.sparsity_by_layer = sparsify(
            model,
            self.config.sparsity,
            self.config.sparsity_assignment
        )

        model = model.to(
            device=self.device, 
            memory_format=self.memory_format,
        )

        if self.config.compile:
            model = torch.compile(model)

        return model

    def train_step(self, batch_idx: int, inputs: Tensor, targets: Tensor):

        is_update_step = (
            self.step_idx + 1) % self.config.prune_grow_delta == 0
        is_before_final_step = self.step_idx <= self.final_prune_grow_step
        is_none = self.config.prune == "none"

        if is_update_step and is_before_final_step and (not is_none):
            print("Pruning and growing model...")

            samples = utils.take(self.train_loader, self.prune_grow_batches)

            grow_scores = grow.compute_grow_score(
                self.model,
                self.mask_param_pairs,
                self.config.grow,
                self.config.grow_subset,
                self.loss_fn,
                samples,
                self.device,
                self.report_correlation,
            )

            prune_scores = prune.compute_prune_score(
                self.model,
                self.mask_param_pairs,
                self.config.prune,
                self.loss_fn,
                samples,
                self.device,
            )

            grow_rate = utils.cosine_annealing(
                self.step_idx,
                self.final_prune_grow_step,
                self.config.prune_grow_rate
            )
            grow_sparsity = grow.get_grow_sparsity(
                grow_rate,
                self.config.sparsity,
                self.sparsity_by_layer,
                self.config.scope
            )

            # update the masks to add those with the highest grow scores
            grown_sets = grow.apply_growing(
                grow_scores,
                self.mask_param_pairs,
                grow_sparsity,
                self.config.scope
            )

            # ensure that the just grown connections will not be pruned in this iteration
            prune.disable_grown_connections(prune_scores, grown_sets)
            prune_sparsity = self.config.sparsity if self.config.scope == "global" else self.sparsity_by_layer

            # update the masks to remove those with the lowest prune scores
            pruned_sets = prune.apply_pruning(
                prune_scores,
                self.mask_param_pairs,
                prune_sparsity,
                self.config.scope
            )

            # ensure that the momentum buffers are reset for the newly grown connections
            grow.reset_momentum(self.optimizer, grown_sets)

            self.sparsity_saver.step(self.epoch_idx, self.step_idx)

        return super().train_step(batch_idx, inputs, targets)


if __name__ == "__main__":
    config = Config(underscores_to_dashes=True).parse_args()
    runner = Runner(config)
    runner.train()
