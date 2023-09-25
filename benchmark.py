import math
import torch
import torch.utils.benchmark as benchmark
import pandas as pd
from tqdm import tqdm

from disk import random_string


def experiment(sparsity, dims, batch_size, device, repeats=10):

    w = torch.randn(dims, dims, device=device)
    m = torch.empty_like(w).bernoulli_(1 - sparsity)
    w_dense = w * m
    w_coo = w_dense.to_sparse_coo()
    w_csr = w_dense.to_sparse_csr()

    x = torch.randn(dims, batch_size, device=device)

    dense_time = benchmark.Timer(
        stmt='torch.mm(w, x)',
        setup='import torch',
        globals={'w': w_dense, 'x': x}).timeit(repeats)

    yield {
        'device': device.type,
        'dimensions': dims,
        'sparsity': sparsity,
        'method': 'dense',
        'time': dense_time.mean,
    }

    coo_time = benchmark.Timer(
        stmt='torch.sparse.mm(w, x)',
        setup='import torch',
        globals={'w': w_coo, 'x': x}).timeit(repeats)

    yield {
        'device': device.type,
        'dimensions': dims,
        'sparsity': sparsity,
        'method': 'coo',
        'time': coo_time.mean,
    }

    csr_time = benchmark.Timer(
        stmt='torch.sparse.mm(w, x)',
        setup='import torch',
        globals={'w': w_csr, 'x': x}).timeit(repeats)

    yield {
        'device': device.type,
        'dimensions': dims,
        'sparsity': sparsity,
        'method': 'csr',
        'time': csr_time.mean,
    }


def get_machine_name():
    """returns the name of the machine"""
    import socket
    return socket.gethostname()


def get_device_name(device):
    """returns the name of the device"""
    import platform

    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)

    # return cpu name and manufacturer
    return platform.processor()


if __name__ == "__main__":

    sparsities = (1 - torch.logspace(math.log10(0.2),
                  math.log10(0.001), 10)).tolist()
    dims = [2048, 4096, 8192, 16384]

    devices = {torch.device("cpu")}
    if torch.cuda.is_available():
        devices.add(torch.device("cuda:0"))

    results = []

    for sparsity in tqdm(sparsities):
        for device in devices:
            for dim in dims:
                for result in experiment(sparsity, dim, 128, device):
                    result['machine'] = get_machine_name()
                    result['device_name'] = get_device_name(device)
                    results.append(result)

    rand_str = random_string()

    results = pd.DataFrame(results)
    results.to_csv(f'results/sparse-ops-timing-{rand_str}.csv', index=False)
