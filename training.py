from typing import List, Literal, Optional
import os
import time
import random
import warnings
import numpy
import torch
from torch import Tensor
from contextlib import suppress
from functools import partial
import torchvision
from tap import Tap

from datasets import DatasetOptions, get_dataloader, num_classes_by_dataset, image_size_by_dataset
from models import ModelOptions, get_model
from disk import StateSaver, MetricSaver
import utils


class Config(Tap):
    dataset: DatasetOptions
    model: ModelOptions
    model_width: float = 1  # width multiplier of the selected model
    train_batch_size: int = 128
    eval_batch_size: int = 512
    epochs: int = 200
    warmup: int = 5  # epochs of learning rate warm up
    # epochs at which to reduce the learning rate
    milestones: List[int] = [80, 120]
    # factor by which to reduce the learning rate
    gamma: float = 0.1
    learning_rate: float = 0.1  # initial learning rate. CNNs: 0.1, ViT: 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    label_smoothing: float = 0.0
    device: str = "cuda:0"
    compile: bool = False
    amp: bool = False
    channels_last: bool = False
    image_backend: Optional[Literal["PIL", "accimage"]] = None
    seed: Optional[int] = None
    workers: int = 0
    download: bool = False
    result_dir: str = "runs"
    dataset_dir: str = "data"
    checkpoints: List[int] = []
    save_initial: bool = True
    save_best: bool = True
    print_freq: int = 50  # in batches


class Runner:

    # metrics to write to disk
    fieldnames = [
        "epoch",
        "step",
        "train_loss",
        "train_accuracy",
        "train_start_time",
        "train_duration",
        "test_loss",
        "test_accuracy",
        "test_start_time",
        "test_duration",
    ]

    def __init__(self, config: Config):
        self.config = config

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set before seeding because seeding disables benchmark
            torch.backends.cudnn.benchmark = True

        if self.config.seed is not None:
            self.seed(self.config.seed)

        if self.config.image_backend is not None:
            torchvision.set_image_backend(self.config.image_backend)

        self.artifact_dir = self.create_artifact_dir(self.config.result_dir)
        print(f"Storing artifacts in: {self.artifact_dir}")

        config_path = os.path.join(self.artifact_dir, "config.json")
        self.config.save(config_path)

        self.device = self.resolve_device()
        print(f"Using device: {self.device}")

        if self.config.channels_last:
            self.memory_format = torch.channels_last
        else:
            self.memory_format = torch.contiguous_format

        self.train_loader, self.test_loader = self.init_dataloaders()

        self.num_classes = num_classes_by_dataset[config.dataset]
        self.image_size = image_size_by_dataset[config.dataset]

        self.model = self.init_model()
        num_params = utils.num_parameters(self.model)
        print(f"Number of model parameters: {num_params:,}")

        self.autocast, self.scaler = self.init_amp()

        metrics_path = os.path.join(self.artifact_dir, "metrics.csv")
        self.metric_saver = MetricSaver(metrics_path, self.fieldnames)

        # set epoch_idx to -1 because the state_fn requires it to be defined
        self.epoch_idx = -1
        self.step_idx = -1

        self.loss_fn = self.init_loss()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

        def state_fn():
            return {
                "epoch": self.epoch_idx + 1,
                "step": self.step_idx + 1,
                "wall_time": time.time(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

        self.state_saver = StateSaver(
            self.artifact_dir,
            state_fn,
            self.config.checkpoints,
            self.config.save_initial,
            self.config.save_best
        )

    def seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        numpy.random.seed(seed)

        if torch.cuda.is_available():
            torch.backend.cudnn.deterministic = True
            torch.backend.cudnn.benchmark = False

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    def create_artifact_dir(self, root: str):
        random_id = utils.random_id()

        artifact_dir = os.path.join(root, random_id)
        artifact_dir = os.path.abspath(artifact_dir)
        os.makedirs(artifact_dir)

        return artifact_dir

    def resolve_device(self):
        device = torch.device(self.config.device)

        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        elif device.type == "mps" and not torch.backends.mps.is_available():
            device = torch.device("cpu")

        return device

    def init_dataloaders(self):
        return get_dataloader(
            self.config.dataset,
            self.config.dataset_dir,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.config.workers,
            self.config.download,
            self.device,
        )

    def init_model(self):
        model = get_model(
            self.config.model,
            self.image_size,
            self.num_classes,
            self.config.model_width
        )
        
        model = model.to(
            device=self.device,
            memory_format=self.memory_format,
        )

        if self.config.compile:
            model = torch.compile(model)

        return model

    def init_amp(self):
        if self.config.amp:
            autocast = partial(torch.autocast, self.device.type)
        else:
            autocast = suppress

        if self.config.amp and self.device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        return autocast, scaler

    def init_loss(self):
        return torch.nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

    def init_optimizer(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

        return optimizer

    def init_scheduler(self):
        # Duration is specified in epochs
        # but the scheduler is updated every step.
        # Thus convert epoch to step number.
        num_batches = len(self.train_loader)
        schedulers = []

        # Disable learning rate warmup when 0 or negative
        if self.config.warmup > 0:
            warmup_steps = self.config.warmup * num_batches
            scheduler1 = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            schedulers.append(scheduler1)

        milestones = [x * num_batches for x in self.config.milestones]
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=self.config.gamma
        )
        schedulers.append(scheduler2)

        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers)
        return scheduler

    def train(self):
        for epoch_idx in range(self.config.epochs):

            self.epoch_idx = epoch_idx
            print(f"Epoch: {epoch_idx + 1}/{self.config.epochs}")

            train_metrics = self.train_one_epoch()
            test_metrics = self.evaluate()

            metrics = {
                "epoch": self.epoch_idx + 1,
                "step": self.step_idx + 1,
            }

            metrics = metrics | train_metrics | test_metrics
            self.metric_saver.write(metrics)

            self.state_saver.step(test_metrics["test_loss"])

            train_loss = metrics["train_loss"]
            test_loss = metrics["test_loss"]
            train_accuracy = metrics["train_accuracy"] * 100
            test_accuracy = metrics["test_accuracy"] * 100
            print(
                f"train loss: {train_loss:.4g} \t test loss: {test_loss:.4g} \t train accuracy: {train_accuracy:.4g} \t test accuracy: {test_accuracy:.4g}")

    def train_one_epoch(self):
        self.model.train()

        mean_loss = utils.MeanMetric()
        mean_accuracy = utils.MeanMetric()

        start_time = time.time()
        batch_start_time = time.time()

        total_batches = len(self.train_loader)

        # Don't print if there would only be one
        enable_print = total_batches > self.config.print_freq

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):

            self.step_idx += 1

            inputs = inputs.to(
                device=self.device,
                memory_format=self.memory_format,
                non_blocking=True
            )
            targets = targets.to(device=self.device, non_blocking=True)

            loss, accuracy = self.train_step(batch_idx, inputs, targets)

            self.scheduler.step()

            with torch.no_grad():
                batch_size = inputs.size(0)
                mean_loss.update(loss.cpu(), weight=batch_size)
                mean_accuracy.update(accuracy.cpu(), weight=batch_size)

                is_print_step = (batch_idx % self.config.print_freq) == 0
                if is_print_step and enable_print:
                    duration = time.time() - batch_start_time
                    batch_start_time += duration

                    # duration of one batch
                    divisor = 1 if batch_idx == 0 else self.config.print_freq
                    avg_duration = duration / divisor

                    loss = mean_loss.compute().item()
                    accuracy = mean_accuracy.compute().item() * 100

                    lrl = [param_group['lr']
                           for param_group in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)

                    print(
                        f"[{batch_idx}/{total_batches}] \t loss: {loss:.4g} \t accuracy: {accuracy:.4g} \t time: {avg_duration:.3g}s \t lr: {lr:.3g}")

        duration = time.time() - start_time

        return {
            "train_loss": mean_loss.compute().item(),
            "train_accuracy": mean_accuracy.compute().item(),
            "train_start_time": start_time,
            "train_duration": duration,
        }

    def train_step(self, batch_idx: int, inputs: Tensor, targets: Tensor):

        self.optimizer.zero_grad(set_to_none=True)

        with self.autocast():
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            accuracy = utils.calc_accuracy(logits, targets)

        return loss.detach(), accuracy

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        mean_loss = utils.MeanMetric()
        mean_accuracy = utils.MeanMetric()

        start_time = time.time()
        batch_start_time = time.time()

        total_batches = len(self.test_loader)

        # Don't print if there would only be one
        enable_print = total_batches > self.config.print_freq

        for batch_idx, (inputs, targets) in enumerate(self.test_loader):

            inputs = inputs.to(
                device=self.device,
                memory_format=self.memory_format,
                non_blocking=True
            )
            targets = targets.to(device=self.device, non_blocking=True)

            loss, accuracy = self.evaluate_step(batch_idx, inputs, targets)

            batch_size = inputs.size(0)
            mean_loss.update(loss.cpu(), weight=batch_size)
            mean_accuracy.update(accuracy.cpu(), weight=batch_size)

            is_print_step = (batch_idx % self.config.print_freq) == 0
            if is_print_step and enable_print:
                duration = time.time() - batch_start_time
                batch_start_time += duration

                loss = mean_loss.compute().item()
                accuracy = mean_accuracy.compute().item() * 100

                print(
                    f"[{batch_idx}/{total_batches}] \t loss: {loss:.4g} \t accuracy: {accuracy:.4g} \t time: {duration:.3g}s")

        duration = time.time() - start_time

        return {
            "test_loss": mean_loss.compute().item(),
            "test_accuracy": mean_accuracy.compute().item(),
            "test_start_time": start_time,
            "test_duration": duration,
        }

    @torch.no_grad()
    def evaluate_step(self, batch_idx: int, inputs: Tensor, targets: Tensor):

        with self.autocast():
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)

        accuracy = utils.calc_accuracy(logits, targets)
        return loss, accuracy


if __name__ == "__main__":
    config = Config(underscores_to_dashes=True).parse_args()
    runner = Runner(config)
    runner.train()
