"""Unary factor for single tokens, 
which could also be used as a decoder, a quick baseline for sequence tagging.
"""
from typing import Union

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class UnaryFactor(nn.Module):
    """An unary factor for each token's dense representation.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 dropout: float):
        """
        Args:
            input_dim: dimension of the representation for each token.
            output_dim: size of the label space.
            hidden_dim: size of the hidden state in the fc layer.
            dropout: the dropout rate.
        """
        super().__init__()
        if hidden_dim != 0:
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = None
            self.fc_out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_states: Union[Tensor, list]) -> Tensor:
        """Fully connected layers.
        Args:
            token_states: [batch_size, *, state_dim]
                batch_size is total number of tokens / unary factors.
                Also accepts list of len 1, consistent with HighOrderFactor
        Returns:
            label confidence: [batch_size, *, output_dim]
        """
        if type(token_states) == list:
            token_states = token_states[0]
        if self.fc is not None:
            hidden = F.relu(self.fc(self.dropout(token_states)))
            # shape: [batch_size, *,  hidden_dim]
            output = self.fc_out(self.dropout(hidden))
        else:
            output = self.fc_out(self.dropout(token_states))
        # shape: [batch_size, *, output_dim]
        return output
