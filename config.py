import os
from typing import List, Literal, Optional
import string
import random
from datetime import datetime
from tap import Tap

from datasets import DatasetOptions
from models import ModelOptions
from utils import Scope


class CommonOptions(Tap):
    dataset: DatasetOptions
    model: ModelOptions
    sparsity: float = 0.98
    # the scope of the pruning/growing process
    scope: Scope = "global"
    train_batch_size: int = 128
    eval_batch_size: int = 512
    # how many samples of train data to use per class for pruning/growing
    prune_grow_samples: int = 10
    epochs: int = 200
    # epochs at which to reduce the learning rate
    milestones: List[int] = [80, 120]
    # factor by which to reduce the learning rate
    gamma: float = 0.1
    learning_rate: float = 0.1  # initial learning rate. CNNs: 0.1, ViT: 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
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


def get_random_id(len=7) -> str:
    """Create a random id containing the current date and time"""
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d-%H_%M_%S")

    chars = string.ascii_letters + string.digits
    rand_chars = random.choices(chars, k=len)
    rand_str = "".join(rand_chars)

    return f"{time_str}-{rand_str}"


def new_result_directory(base_dir: str) -> str:
    """Creates a new directory and returns its path"""
    random_id = get_random_id()

    result_dir = os.path.join(base_dir, random_id)
    result_dir = os.path.abspath(result_dir)
    os.makedirs(result_dir)

    return result_dir
