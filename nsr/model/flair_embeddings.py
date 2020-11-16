"""Flair embeddings wrapper, the output format can be directly used with factor
models. The implementation refers to
https://github.com/flairNLP/flair/blob/master/resources/docs/EXPERIMENTS.md
for reccommended Flair settings for NER tasks;
and https://github.com/flairNLP/flair/blob/72f4ad6706d0d4efc8711f007eeb7e2bad5a69d6/flair/models/sequence_tagger_model.py#L502
for embedding sentences and pad the embeddings.
"""
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import flair
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, StackedEmbeddings, \
    PooledFlairEmbeddings


class FlairEmbeddings:
    """Embed sentences with the reccommended Flair embedding setting for each
    of these languages {"en", "de", "nl"}
    """
    def __init__(self, language: str, cache: "FlairEmbeddingCache" = None):
        """
        Args:
            language: one of {"en", "de", "nl", "es"}
        """
        super().__init__()
        assert language in {"en", "de", "nl", "es"}, \
            "Unknown language for Flair: {}".format(language)
        
        self.cache = None
        if cache is None:
            if language == "en":
                self.embeddings = StackedEmbeddings(embeddings=[
                    # GloVe embeddings
                    WordEmbeddings('glove'),
                    # contextual string embeddings, forward
                    PooledFlairEmbeddings('news-forward', pooling='min'),
                    # contextual string embeddings, backward
                    PooledFlairEmbeddings('news-backward', pooling='min')
                ])
            elif language == "de":
                self.embeddings = StackedEmbeddings(embeddings=[
                    WordEmbeddings('de'),
                    # Temporary hack: comment out Flair lines for Fasttext only
                    PooledFlairEmbeddings('german-forward'),
                    PooledFlairEmbeddings('german-backward')
                ])
            elif language == "nl":
                self.embeddings = StackedEmbeddings(embeddings=[
                    WordEmbeddings('nl'),
                    PooledFlairEmbeddings('dutch-forward', pooling='mean'),
                    PooledFlairEmbeddings('dutch-backward', pooling='mean')
                ])
            elif language == "es":
                self.embeddings = StackedEmbeddings(embeddings=[
                    WordEmbeddings('es')
                ])
            self.embedding_dim = self.embeddings.embedding_length
        else:
            self.cache = cache
            self.embedding_dim = cache.embedding_dim

    def __call__(self, sentences: List[str]) -> Tensor:
        """Return the Flair token embeddings for each sentence, the number of
        tokens is padded to the maximum length of the sentences.
        
        Args:
            sentences: each sentence string, space separated tokens.
                len(sentences) == batch_size
        Returns:
            embeddings: shape [batch_size, max_num_tokens, token_embedding_dim]
        """
        lengths: List[int] = [len(sentence.split()) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embedding_dim * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for i, sentence in enumerate(sentences):
            if self.cache is None:
                sent = Sentence(sentence)
                self.embeddings.embed([sent])
                all_embs += [emb for token in sent
                             for emb in token.get_each_embedding()]
                sent.clear_embeddings()
            else:
                all_embs.append(
                    self.cache[sentence].to(flair.device).flatten())
            
            nb_padding_tokens = longest_token_sequence_in_batch - lengths[i]
            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embedding_dim * nb_padding_tokens
                ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embedding_dim,
            ]
        )
        return sentence_tensor
    