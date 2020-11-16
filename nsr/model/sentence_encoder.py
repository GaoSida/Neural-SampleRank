"""Sentence encoder that takes token embedding sequence as input,
and output a contextual representation for each token.

Bi-directional LSTM / GRU; Transformer etc.
"""

import torch
from torch import Tensor
import torch.nn as nn
import flair


class BiRNN(nn.Module):
    """Bi-directional LSTM or GRU
    """ 
    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, cell_type: str,
                 embed_dropout: float = None, word_dropout: float = None,
                 locked_dropout: float = None):
        """
        Args:
            embed_dim: dimension of input token embeddings
            hidden_dim: dimensions of hidden states of RNN in each direction.
            num_layers: number of layers of BiRNN.
            dropout: the dropout rate. The input is assumed to be before
                any dropout.
            cell_type: "LSTM" or "GRU"
            embed_dropout: when None, follow the main dropout rate
            word_dropout, locked_dropout: when None, disabled
        """
        super().__init__()
        assert cell_type in {"LSTM", "GRU"}, \
            "Unknown cell type for BiRNN: {}".format(cell_type)
        
        rnn_cell = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_cell(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        
        if embed_dropout is not None:
            if embed_dropout > 0.0:
                self.embed_dropout = nn.Dropout(embed_dropout)
            else:
                self.embed_dropout = None
        else:
            self.embed_dropout = nn.Dropout(dropout)
        
        if word_dropout is not None and word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        else:
            self.word_dropout = None
        
        if locked_dropout is not None and locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
        else:
            self.locked_dropout = None
        
    def forward(self, embeddings: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            embeddings: token embeddings, batch first and padded.
                shape [batch_size, max_num_tokens, embed_dim]
            lengths: the length of each sentence.
                shape [batch_size, ], each length is as most max_num_tokens
        Returns:
            RNN output: [batch_size, max_num_tokens, 2 * hidden_dim]
        """
        if self.embed_dropout:
            embeddings = self.embed_dropout(embeddings)
        if self.word_dropout:
            embeddings = self.word_dropout(embeddings)
        if self.locked_dropout:
            embeddings = self.locked_dropout(embeddings)
        
        rnn_input = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, 
            batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(rnn_input)  # Ignore hidden states
        # PackedSequence, to be unpack then padded back
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            rnn_output, batch_first=True)
        return rnn_output
