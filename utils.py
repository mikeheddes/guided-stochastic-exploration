from typing import Dict, Iterable, Tuple, Callable, Literal, Optional
import math
import random
import numpy
import torch
from torch import Tensor
from tqdm import tqdm
import string
from datetime import datetime
from itertools import islice
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from contextlib import suppress
from torch.utils.data import DataLoader

try:
    import cupy
except ImportError:
    pass


from models.simple_vit import PosembSincos2d, PosembSincos2dABS

PruningName = Literal["none", "random", "magnitude", "snip", "grasp", "synflow"]
GrowingName = Literal["none", "random", "rigl", "rigl-random", "rigl-synflow", "rigl-grabo", "rigl-ams"]
Scope = Literal["local", "global"]
SparsityAssignment = Literal["uniform", "erdos-renyi", "scaled"]
SparseFormat = Literal["masked", "sparse", "mixed"]
Scores = Dict[int, Tensor]
LossFn = Callable[[Tensor, Tensor], Tensor]
MaskParamIterable = Iterable[Tuple[Tensor, Tensor]]


class MeanMetric:
    value: Tensor
    weight: Tensor

    def __init__(self, device: torch.device = None) -> None:
        self.value = torch.tensor(0.0, dtype=torch.float, device=device)
        self.weight = torch.tensor(0.0, dtype=torch.float, device=device)

    def update(self, value: Tensor, weight: Tensor = 1.0) -> None:
        self.value += value * weight
        self.weight += weight

    def compute(self) -> Tensor:
        return self.value / self.weight
    

def random_id(len=7) -> str:
    """Create a random id containing the current date and time"""
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d-%H_%M_%S")

    chars = string.ascii_letters + string.digits
    rand_chars = random.choices(chars, k=len)
    rand_str = "".join(rand_chars)

    return f"{time_str}-{rand_str}"


def num_parameters(model: nn.Module) -> int:
    total_params = 0

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += param.numel()

    return total_params


def calc_accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    batch_size = logits.size(0)
    predictions = logits.argmax(-1)
    return torch.sum(predictions == targets, dtype=torch.float) / batch_size


def cosine_annealing(t, t_end, start_value) -> float:
    if t >= t_end:
        return 0.0

    return start_value / 2 * (1 + math.cos(t * math.pi / t_end))


def divmod(x, y, /):
    return x // y, x % y


def setdiff1d(input: Tensor, other: Tensor, assume_unique: bool=False) -> Tensor:
    device = input.device

    if device.type == "cuda":            
        op = cupy.setdiff1d
        asarray = cupy.asarray
    elif device.type == "cpu":
        op = numpy.setdiff1d
        asarray = numpy.asarray

    input = asarray(input)
    other = asarray(other)

    diff = op(input, other, assume_unique=assume_unique)
    
    if diff.shape == (0,):
        return torch.empty(0, device=device)
    else:
        return torch.as_tensor(diff, device=device)


class ptdict(dict):
    '''dict that returns the key without storing it if key not found'''
    def __missing__(self, key):
        return key


def generate_uniform_wo_replacement(N, exclude = None):
    '''sampling without replacement, returns a generator does not sample from exclude'''
    # See: https://folk.ntnu.no/staal/programming/algorithms/wrsampling/
    # Ref: Ernvall, Jarmo, and Olli Nevalainen. 1982. “An Algorithm for Unbiased Random Sampling.” The Computer Journal 25 (1): 45–47.

    end = N
    remap = ptdict()

    if exclude is not None:
        for x in sorted(exclude, reverse=True):
            remap[x] = remap[end-1]
            end -= 1

    for _ in range(end):
        j = random.randrange(end)
        k = remap[j]
        remap[j] = remap[end-1]
        end -= 1
        yield k


def sample_uniform_wo_replacement(N, n, exclude=None):
    generator = generate_uniform_wo_replacement(N, exclude=exclude)
    return list(islice(generator, n))


def sample(probs_1: torch.Tensor, probs_2: torch.Tensor, n: int) -> torch.Tensor:
    """This function samples pairs from the joint probability distribution induced by the probabilities in probs_1 and probs_2 and returns the indices of the sampled pairs"""
    # Sample n indices without replacement from probs_1 and probs_2
    indices_1 = torch.multinomial(probs_1, num_samples=n, replacement=True)
    indices_2 = torch.multinomial(probs_2, num_samples=n, replacement=True)
    # Combine the indices into pairs and return as a tensor
    pairs = torch.stack((indices_1, indices_2), dim=1)
    return pairs


def sample_wo_replacement(prob_out, prob_in, num_samples, active_edges, eps=1e-6):
    """
    Sample edges without replacement according to the in and out probability distributions
    without represending the squared probability matrix
    """
    dtype = active_edges.dtype
    device = active_edges.device

    # store the nonzero (to, from) pairs as a set
    active_set = set(tuple(x) for x in active_edges.tolist())
    subset = set()

    num_params = prob_out.numel() * prob_in.numel()
    sparsity = 1.0 - len(active_set) / num_params

    # make sure we don't sample more than the number of available parameters
    num_samples = min(num_samples, num_params - len(active_set))

    # if there are no available parameters, return an empty tensor
    if num_samples == 0:
        return torch.zeros((0, 2), dtype=dtype, device=device)

    # if there are more available parameters than active ones, use efficient sampling
    if sparsity > 0.5:
        num_iters = 0
        while len(subset) < num_samples:

            # if we are stuck in an infinite loop, raise an error (this should never happen)
            num_iters += 1
            if num_iters > 1000:
                raise RuntimeError("Infinite loop detected")

            # the eps ensures that the probability is never zero and thus the multinomial
            # never gets stuck in an infinite while loop
            to_sample = num_samples - len(subset)
            to_node = torch.multinomial(prob_out + eps, to_sample, replacement=True)
            from_node = torch.multinomial(prob_in + eps, to_sample, replacement=True)

            for to_n, from_n in zip(to_node.tolist(), from_node.tolist()):
                if (to_n, from_n) in active_set:
                    continue

                subset.add((to_n, from_n))

    # else represent the full probability matrix and sample from that
    # in this case it only takes at most twice as much compute so it is still efficient
    else:
        prob_matrix = prob_out.view(-1, 1) * prob_in.view(1, -1)
        prob_matrix[active_edges[:, 0], active_edges[:, 1]] = 0.0
        prob_matrix = prob_matrix.view(-1)

        indices = torch.multinomial(prob_matrix, num_samples, replacement=False)
        num_columns = prob_in.size(0)

        for index in indices.tolist():
            to_n, from_n = divmod(index, num_columns)
            subset.add((to_n, from_n))

    return torch.tensor(list(subset), dtype=dtype, device=device)


# http://en.wikipedia.org/wiki/Mersenne_prime
mersenne_prime = (1 << 61) - 1


class FourWayIndependent(torch.nn.Module):
    params: Tensor

    def __init__(self, *size: int) -> None:
        super().__init__()
        self.size = size

        params = torch.empty(4, 1, math.prod(size), dtype=torch.long)
        self.register_buffer("params", params)
        self.reset_params()

    def reset_params(self) -> None:
        # Create parameters for a random bijective permutation function
        # https://en.wikipedia.org/wiki/Universal_hashing
        abc = torch.randint(1, mersenne_prime, (3, 1, math.prod(self.size)))
        d = torch.randint(0, mersenne_prime, (1, 1, math.prod(self.size)))
        torch.cat((abc, d), dim=0, out=self.params)

    def forward(self, input: Tensor) -> Tensor:
        assert input.dim() == 2
        assert input.size(1) == 1

        a, b, c, d = self.params
        output = (a * input**3 + b * input**2 + c * input + d) % mersenne_prime
        return output.view(-1, *self.size)


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None,
    scaler: Optional[GradScaler] = None,
    amp_autocast = suppress,
) -> Tuple[float, float]:
    model.train()

    mean_loss = MeanMetric()
    mean_accuracy = MeanMetric()

    with tqdm(total=len(data_loader), unit="batches") as pbar:
        for inputs, targets in data_loader:
            
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                logits = model(inputs)
                loss = loss_fn(logits, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                batch_size = inputs.size(0)
                accuracy = calc_accuracy(logits, targets)
                mean_loss.update(loss.cpu(), weight=batch_size)
                mean_accuracy.update(accuracy.cpu(), weight=batch_size)

                pbar.update(1)
                pbar.set_postfix(
                    loss=mean_loss.compute().item(),
                    acc=mean_accuracy.compute().item() * 100,
                )

    return mean_loss.compute().item(), mean_accuracy.compute().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None,
    amp_autocast = suppress,
) -> Tuple[float, float]:
    model.eval()

    mean_loss = MeanMetric()
    mean_accuracy = MeanMetric()

    with tqdm(total=len(data_loader), unit="batches") as pbar:
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with amp_autocast():
                logits = model(inputs)
                loss = loss_fn(logits, targets)

            accuracy = calc_accuracy(logits, targets)

            batch_size = inputs.size(0)
            mean_loss.update(loss.cpu(), weight=batch_size)
            mean_accuracy.update(accuracy.cpu(), weight=batch_size)

            pbar.update(1)
            pbar.set_postfix(
                loss=mean_loss.compute().item(),
                acc=mean_accuracy.compute().item() * 100,
            )

    return mean_loss.compute().item(), mean_accuracy.compute().item()


def take(iterable, n):
    """Return first n items of the iterable as a list"""
    iterator = iter(iterable)
    return [next(iterator) for _ in range(n)]


def prepare_synflow_model(model):
    mods = []
    unmods = []

    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, (nn.LayerNorm, nn.GELU, nn.Softmax)):
                mods.append((parent, name, nn.Identity()))
                unmods.append((parent, name, child))

            if isinstance(child, PosembSincos2d):
                mods.append((parent, name, PosembSincos2dABS()))
                unmods.append((parent, name, child))

    # apply the identity function to all the layer norm and gelu layers
    for mod in mods:
        setattr(*mod)

    def revert():
        for mod in unmods:
            setattr(*mod)

    return revert
