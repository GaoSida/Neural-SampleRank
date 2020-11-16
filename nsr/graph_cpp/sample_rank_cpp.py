"""Wrap C++ SampleRank loss as a PyTorch autograd.Function
"""
from typing import List, Tuple, Dict

import torch
from torch import Tensor

import cpp_sampling


class SampleRankLossCpp(torch.autograd.Function):
    """SamplerRank loss with the heavy-lifting done in C++
    """
    @staticmethod
    def forward(ctx, config: dict, label_space_sizes: List[int],
                factor_dependencies: Dict[str, List[Tuple[int]]],
                sample_pool: List[Tuple[int]], ground_truth: List[int],
                factor_types: List[str], *args) -> Tensor:
        """Compute SampleRank loss in C++, and save the gradient
        Args:
            ctx: the required contex object
            config: A dictionary holding the configs
            factor_types: list of factor type names
            *args: the factor Tensors in that order
        Returns: the loss Tensor
        """
        results = cpp_sampling.sample_rank_loss(
            label_space_sizes, factor_dependencies, factor_types, list(args),
            sample_pool, ground_truth, config["train_num_samples"],
            config["gold_loss_enabled"], config["pair_loss_enabled"],
            config["train_init_with_oracle"]
        )
        ctx.save_for_backward(*results[1:])
        return results[0][0]

    @staticmethod
    def backward(ctx, grad_loss: Tensor) -> Tuple[Tensor]:
        """The args and returns are determined by the forward definition.
        """
        return tuple([None] * 6) + ctx.saved_variables
