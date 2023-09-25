from typing import Optional, Union, Tuple, List
import torch
import math
from torch import Tensor
from torch import LongTensor
import torch.nn as nn
import torch.nn.functional as F

from utils import MaskParamIterable, SparsityAssignment, SparseFormat


def sample_erdos_renyi(in_features: int, out_features: int, num_edges: int) -> LongTensor:
    edges = torch.empty(2, 0, dtype=torch.long)

    while edges.size(1) < num_edges:

        remaining = num_edges - edges.size(1)
        from_nodes = torch.randint(0, in_features, (remaining,))
        to_nodes = torch.randint(0, out_features, (remaining,))

        edge_samples = torch.stack((to_nodes, from_nodes), dim=0)
        edges = torch.cat((edges, edge_samples), dim=1)
        edges = torch.unique(edges, sorted=True, dim=1)

    return edges


def init_sparse_params(in_features: int, out_features: int, sparsity: float, bias: bool = True) -> Tuple[Tensor, Optional[Tensor]]:

    num_edges = math.ceil(in_features * out_features * (1 - sparsity))
    edges = sample_erdos_renyi(in_features, out_features, num_edges)

    row_count = torch.bincount(edges[0] + 1, minlength=out_features + 1)
    crow_indices = torch.cumsum(row_count, dim=0)
    col_indices = edges[1]

    # row_count is the actual fan-in, taking into account the sparsity
    fan_in = row_count[1:]
    # set the bound such that the output has mean=0, std=1
    # clamp 1 prevents divide by zero
    bound = math.sqrt(3) / torch.sqrt(fan_in.clamp_min(1))

    weights = torch.empty(num_edges).uniform_(-1.0, 1.0)
    # scale the weights by the actual fan in of each output neuron
    weights.multiply_(bound[edges[0]])

    if bias:
        bias = torch.empty(out_features).uniform_(-1.0, 1.0)
        bias.multiply_(bound)
    else:
        bias = None

    weight = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        weights,
        size=(out_features, in_features)
    )

    return weight, bias


def register_input(module, args, output):
    module.input = args[0]


def register_grad(module, grad_input, grad_output):
    module.grad_output = grad_output[0]


# use as:
# forward_handle = module.register_forward_hook(register_input)
# backward_handle = module.register_full_backward_hook(register_grad)


class SparseLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, sparsity: float, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        weight, bias = init_sparse_params(in_features, out_features, sparsity, bias)

        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    @property
    def sparsity(self) -> float:
        return 1 - self.weight.values().size(0) / math.prod(self.weight.size())

    def forward(self, input: Tensor) -> Tensor:

        if self.bias is None:
            return torch.sparse.mm(self.weight, input.T).T
        else:
            return torch.sparse.addmm(self.bias.unsqueeze(1), self.weight, input.T).T

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, sparsity={:.3g}, bias={}'.format(
            self.in_features, self.out_features, self.sparsity, self.bias is not None
        )


class MaskedLinear(nn.Linear):
    mask: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)

        mask = torch.ones((out_features, in_features), **factory_kwargs)
        self.register_buffer('mask', mask)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.mask.sum().item() / self.mask.numel()

    def extra_repr(self) -> str:
        s = super().extra_repr()
        return s + ', sparsity={:.3f}'.format(self.sparsity)


class MaskedConv2d(nn.Conv2d):
    mask: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, **factory_kwargs)

        mask = torch.ones(self.weight.shape, **factory_kwargs)
        self.register_buffer('mask', mask)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight * self.mask, self.bias)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.mask.sum().item() / self.mask.numel()

    def extra_repr(self):
        s = super().extra_repr()
        s += ', sparsity={:.3f}'.format(self.sparsity)
        return s


@torch.no_grad()
def init_masked_params(sparsity: float, module: Union[MaskedLinear, MaskedConv2d]) -> None:
    out_features = module.weight.size(0)

    # if the weights are of an conv2d layer, its shape is (out_channels, in_channels, kernel_size, kernel_size)
    # we need to reshape the mask to (out_channels, in_channels * kernel_size * kernel_size)
    # so that the tensor can be interpreted as the weights of a feed forward layer
    as_mlp_weight = module.weight.view(out_features, -1)
    as_mlp_mask = module.mask.view(out_features, -1)

    # create a random bipartite graph with the target sparsity, uses the Erdos-Renyi G(n, p) model
    # note that for large n, and small p, the G(n, M) model is more efficient
    as_mlp_mask.bernoulli_(1.0 - sparsity)

    # calculate the actual fan in taking into account the sparsity
    fan_in = as_mlp_mask.sum(dim=1, keepdim=True)
    # set the bound such that the output has mean=0, std=1
    # clamp 1 prevents divide by zero
    bound = math.sqrt(3) / torch.sqrt(fan_in.clamp_min(1))

    as_mlp_weight.uniform_(-1.0, 1.0)
    # scale the weights by the actual fan in of each output neuron
    as_mlp_weight.multiply_(bound)
    # ensure that the inactive connections are initialized as zero
    as_mlp_weight.multiply_(as_mlp_mask)

    if module.bias is not None:
        module.bias.uniform_(-1.0, 1.0)
        module.bias.multiply_(bound.squeeze(1))


def get_num_features(weight: Tensor) -> Tuple[int, int]:
    """
    Compute the number of input and output features of a linear or convolution layer
    """
    in_features = math.prod(weight.shape[1:])
    out_features = weight.size(0)
    return in_features, out_features


def get_active_param_scaler(model: nn.Module, sparsity: float, method: SparsityAssignment) -> float:
    """
    calculate the scale factor for the number of active parameters
    that will be used to calculate the sparsity of each layer
    this ensures that the overal sparsity is the same as the target sparsity
    """

    linear_params = 0
    total_params = 0

    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        in_f, out_f = get_num_features(module.weight)
        total_params += in_f * out_f

        if method == "erdos-renyi":
            linear_params += in_f + out_f

        elif method == "scaled":
            linear_params += out_f

        elif method == "uniform":
            # if the method is uniform, then all layers have the same sparsity
            linear_params += in_f * out_f

    if linear_params == 0:
        raise ValueError("Model has no linear or conv2d layers")

    return (1 - sparsity) * total_params / linear_params


def sparsify(model: nn.Module, sparsity: float, method: SparsityAssignment, format: SparseFormat = "masked"):
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

                if format == "masked":

                    sparse_module = MaskedLinear(
                        m.in_features,
                        m.out_features,
                        torch.is_tensor(m.bias),
                        m.weight.device,
                        m.weight.dtype
                    )
                    sparsity_by_layer[id(sparse_module.weight)] = layer_sparsity
                    init_masked_params(layer_sparsity, sparse_module)
                
                elif format == "sparse":

                    sparse_module = SparseLinear(
                        m.in_features,
                        m.out_features,
                        layer_sparsity,
                        torch.is_tensor(m.bias),
                    )
                    sparsity_by_layer[id(sparse_module.weight)] = layer_sparsity

                elif format == "mixed":

                    if layer_sparsity > 0.9:

                        sparse_module = SparseLinear(
                            m.in_features,
                            m.out_features,
                            layer_sparsity,
                            torch.is_tensor(m.bias),
                        )
                        sparsity_by_layer[id(sparse_module.weight)] = layer_sparsity

                    else:

                        sparse_module = nn.Linear(
                            m.in_features,
                            m.out_features,
                            torch.is_tensor(m.bias),
                        )
                        sparsity_by_layer[id(sparse_module.weight)] = 0.0
                
                modifications.append((parent, name, sparse_module))

            if isinstance(m, nn.Conv2d):

                if layer_sparsity < 0 or layer_sparsity > 1:
                    print(
                        f"WARNING: Layer sparsity must be between 0 and 1 but got {layer_sparsity}, clipping has been applied. This could happen if the sparsity is low and the parameters are unevenly distributed over the layers.")
                    layer_sparsity = min(max(layer_sparsity, 0.0), 1.0)

                if format != "masked":
                    raise RuntimeError("Convolution layers are not yet available as sparse layers")

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
                init_masked_params(layer_sparsity, sparse_module)
                modifications.append((parent, name, sparse_module))

    # apply the modifications to the model
    for parent, name, masked_module in modifications:
        setattr(parent, name, masked_module)

    return sparsity_by_layer


class all_masked_modules:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def __iter__(self):
        for module in self.model.modules():
            if isinstance(module, (MaskedLinear, MaskedConv2d)):
                yield module


class all_named_masked_modules:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def __iter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (MaskedLinear, MaskedConv2d)):
                yield name, module


class all_named_mask_param_pairs:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def __iter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (MaskedLinear, MaskedConv2d)):
                yield name, module.mask, module.weight


class all_mask_param_pairs:
    def __init__(self, model: nn.Module):
        self.model = model

    def __iter__(self):
        for module in self.model.modules():
            if isinstance(module, (MaskedLinear, MaskedConv2d)):
                yield module.mask, module.weight


@torch.no_grad()
def apply_masks(mask_param_pairs: MaskParamIterable) -> None:
    for mask, param in mask_param_pairs:
        param.mul_(mask)


@torch.no_grad()
def stats(mask_param_pairs: MaskParamIterable) -> Tuple[int, int]:

    active_params = 0
    total_params = 0

    for mask, _ in mask_param_pairs:
        active_params += mask.sum().item()
        total_params += mask.numel()

    return int(active_params), int(total_params)
