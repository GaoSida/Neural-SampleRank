"""High order scoring factor for multiple tokens.
It can be used as transition factors (i.e. binary factor operates on bigrams),
or skip-chain factors (i.e. operates on non-adjacent token pairs).
The factor can touch 3 or more tokens/RVs as well.
"""
from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class HighOrderFactor(nn.Module):
    """High order factor for 2 or more tokens' dense representations.
    """
    def __init__(self, input_dim: int, order: int, ops: List[Callable],
                 output_dim: int, hidden_dim: int, dropout: float):
        """
        Args:
            input_dim: dimension of the representation for each token.
            order: the number of tokens this factor touches.
            ops: a list of callable reduce ops that takes stacked token
                hidden states as input, the output should be of input_dim.
            output_dim: size of the combinatorial label space.
                This is usually the product of the label space size.
            hidden_dim: size of the hidden state in the fc layer.
            dropout: the dropout rate.
        """
        super().__init__()
        self.feature_ops = ops
        # concatenation, then element-wise ops as defined above
        self.fc = nn.Linear(input_dim * (order + len(self.feature_ops)),
                            hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_states: List[Tensor]) -> Tensor:
        """Feature engineer then fully connected layers.
        Args:
            token_states: list of tensors with shape [batch_size, *, state_dim]
        Returns:
            confidence in the combined label space: [batch_size, *, output_dim]
        """
        features = token_states
        stacked_input = torch.stack(token_states)  # dim = 0
        for op in self.feature_ops:
            features.append(op(stacked_input))
        features = torch.cat(features, dim=-1)
        
        hidden = F.relu(self.fc(self.dropout(features)))
        # shape: [batch_size, *,  hidden_dim]
        output = self.fc_out(self.dropout(hidden))
        # shape: [batch_size, *, output_dim]
        return output
