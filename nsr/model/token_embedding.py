"""The token embedding layer.
We represent each token as a concatenation of pretrained word embedding and 
character embedding from 1-d convolution.
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vectors


class CharCNN(nn.Module):
    """1-D convolution on character sequence.
    One layer convolution, with one size of kernel.
    """
    def __init__(self, vocab_size: int, embed_dim: int, kernel_size: int,
                 num_kernels: int, dropout: float):
        """Initialize the 1-d convolution layer, and character embedding.
        
        Args:
            vocab_size: the number of characters in the vocabulary.
            embed_size: the size of embedding for each character.
            kernel_size: the character window size of the kernel filters.
            num_kernels: Numbe of kernels, i.e. size of output embedding.
            dropout: the dropout rate.
        """
        super().__init__()
        # TODO: Tweak layer initializations. We start with the defaults.
        self.embeddings = nn.Embedding(vocab_size, embed_dim,
                                       padding_idx=1)  # PyTorch convention
        self.conv1d = nn.Conv1d(embed_dim, num_kernels, kernel_size)
        self.fc = nn.Linear(num_kernels, num_kernels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_chars: Tensor) -> Tensor:
        """Compute character embeddings on character indices for each token.
        
        Args:
            token_chars: Each row is the characters of a token in the batch.
                shape [batch_size, max_token_len]
                batch_size is the token number of tokens in the batch of text.
        Returns:
            character embeddings: [batch_size, num_kernels]
        """
        char_embedding = self.dropout(self.embeddings(token_chars))
        # shape: [batch_size, max_token_len, embed_dim]
        char_embedding = char_embedding.permute(0, 2, 1)
        # shape: [batch_size, embed_dim, max_token_len], To work with 1d conv
        
        hidden = F.relu(self.conv1d(char_embedding))
        # shape: [batch_size, num_kernels, conved len]
        hidden, _ = torch.max(hidden, dim=2)  # global max pooling
        # shape: [batch_size, num_kernels]
        output = self.fc(self.dropout(hidden))  # Dropout after pooling
        
        return output  # The output embedding should have no dropouts


class TokenEmbedding(nn.Module):
    """The token embedding layer that assembles pretrained word embedding
    and character embedding from CNN for each token.
    """
    def __init__(self, pretrained: Tensor, char_cnn: CharCNN):
        """
        Args:
            pretrained: a Tensor of pretrained embedding e.g. Glove
                Shape: [vocab_size, embed_dim]
            char_cnn: the model to compute character embedding
        """
        super().__init__()
        # Initialize the embedding table, gradient fozen
        self.pretrained_embeddings = nn.Embedding.from_pretrained(pretrained)
        self.char_cnn = char_cnn
    
    def forward(self, tokens: Tensor, token_chars: Tensor) -> Tensor:
        """
        Args:
            tokens: the token indices. [batch_size, max_num_tokens]
                max_num_tokens is the max sentence length in the batch.
            token_chars: the character indicies for each token.
                Shape [batch_size * max_num_tokens, max_num_chars]
                max_num_chars is the max token length in the batch.
        Returns:
            full token embeddings: [batch_size, max_num_tokens, 
                word embedding dim + char embedding dim]
        """
        token_embeddings = self.pretrained_embeddings(tokens)
        # [batch_size, max_num_tokens, word embedding dim]
        char_embeddings = self.char_cnn(token_chars)
        # [batch_size * max_num_tokens, char embedding dim]
        char_embeddings = char_embeddings.view(
            token_embeddings.shape[0], token_embeddings.shape[1],
            char_embeddings.shape[1]
        )  # [batch_size, max_num_tokens, char embedding dim]
        full_embeddings = torch.cat([token_embeddings, char_embeddings],
                                    dim=2)
        return full_embeddings
