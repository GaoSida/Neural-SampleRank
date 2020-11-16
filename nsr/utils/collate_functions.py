"""Collate functions used in batching for PyTorch DataLoader
"""
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def generate_sentence_batch(batch: List[Dict]):
    """The collate function to be used on a batch of sentences.
    It prepares the batch to be used with PyTorch RNN.
    
    Returns:
        tokens: [batch_size, max_num_tokens]
        token_chars: [batch_size * max_num_tokens, max_num_chars]
        sentence_lengths: [batch_size, ]  i.e. num_tokens
        labels: [batch_size, max_num_tokens]
    """
    tokens = [entry["tokens"] for entry in batch]
    labels = [entry["labels"] for entry in batch]
    sentence_lengths = [len(sent) for sent in tokens]
    max_sent_length = max(sentence_lengths)
    token_chars = list()
    for entry in batch:
        token_chars.extend(entry["token_chars"])
        # Pad number of character sequences to max num tokens
        token_chars.extend([torch.tensor([1]) for _ in range(
            max_sent_length - len(entry["token_chars"]))])
    
    # Pad the token sequences
    tokens = pad_sequence(tokens, batch_first=True, padding_value=1)
    labels = pad_sequence(labels, batch_first=True, padding_value=1)
    # Pad the character sequences
    token_chars = pad_sequence(token_chars, batch_first=True, padding_value=1)
    sentence_lengths = torch.tensor(sentence_lengths)
    
    return tokens, token_chars, sentence_lengths, labels


def generate_document_batch(batch: List[Dict]):
    """The collate function to be used on a batch of documents. Besides NN
    model inputs, it also realigns FactorGraph dependencies in padded batch.
    
    Returns: (batch_size: total number of sentences in the batch)
        tokens: [batch_size, max_num_tokens]
        token_chars: [batch_size * max_num_tokens, max_num_chars]
        sentence_lengths: [batch_size, ]  i.e. num_tokens
        factor_dependencies: Dict[str, List[Tuple[int]]]
        labels: [batch_size, max_num_tokens]
    """
    tokens = [sent for entry in batch for sent in entry["tokens"]]
    labels = [sent for entry in batch for sent in entry["labels"]]
    sentence_lengths = [len(sent) for sent in tokens]
    max_sent_length = max(sentence_lengths)
    token_chars = list()
    for entry in batch:
        for sent in entry["token_chars"]:
            token_chars.extend(sent)
            # Pad number of character sequences to max num tokens
            token_chars.extend([torch.tensor([1]) for _ in 
                                range(max_sent_length - len(sent))])
    
    # Pad the token sequences
    tokens = pad_sequence(tokens, batch_first=True, padding_value=1)
    labels = pad_sequence(labels, batch_first=True, padding_value=1)
    # Pad the character sequences
    token_chars = pad_sequence(token_chars, batch_first=True, padding_value=1)
    sentence_lengths = torch.tensor(sentence_lengths)
    
    # Transpose the factor dependencies into the padded batch of sentences
    factor_dependencies = dict()
    for factor_type in batch[0]["graph"]:
        factor_dependencies[factor_type] = list()
        sent_idx_offset = 0
        for entry in batch:
            for factor in entry["graph"][factor_type]:
                factor_dependencies[factor_type].append(tuple(
                    (sent_idx_offset + sent_idx) * max_sent_length + token_idx
                    for sent_idx, token_idx in factor
                ))
            sent_idx_offset += len(entry["tokens"])
    
    # For Flair embeddings, return raw strings for each sentence as well
    strings = [" ".join(sentence) for entry in batch
               for sentence in entry["strings"]]
    
    return tokens, token_chars, sentence_lengths, factor_dependencies, \
        strings, labels
