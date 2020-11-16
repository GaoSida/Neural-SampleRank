"""Generate an embedding cache, also support queries at runtime.
"""
import os
import pickle
from typing import Dict
import logging

import torch
from torch import Tensor
from torch.utils.data import Dataset
from flair.data import Sentence

from nsr.model.flair_embeddings import FlairEmbeddings

logger = logging.getLogger(__name__)


class FlairEmbeddingCache:
    """Wrapper and utilities for cached Flair embeddings.
    """
    cache: Dict[str, Tensor]  # Sentence to the flattened token embeddings
    
    def __init__(self):
        self.cache = dict()
        self.embedding_dim = -1  # Set after loading cache
        self.cutoff_dim = None  # Set while loading cache
    
    def compute_cache(self, datasets: Dict[str, Dataset], language: str,
                      cache_path: str):
        """
        Args:
            The datasets are CoNLL document datasets. (i.e. doc mode = True)
        """
        flair_embeddings = FlairEmbeddings(language)
        for name, dataset in datasets.items():
            logger.info("Computing Flair embedding for dataset %s", name)
            for i, entry in enumerate(dataset):
                for tokens in entry["strings"]:
                    s = " ".join(tokens)
                    sentence = Sentence(s)
                    flair_embeddings.embeddings.embed([sentence])
                    all_embeddings = [emb for token in sentence
                                      for emb in token.get_each_embedding()]
                    self.cache[s] = torch.cat(all_embeddings).view(
                        [len(tokens),
                         flair_embeddings.embeddings.embedding_length]
                    ).to("cpu")
                    sentence.clear_embeddings()
                if (i + 1) % 5 == 0:
                    logger.info("Flair embedding computation progress: %s/%s",
                                i + 1, len(dataset))
        
        torch.save(self.cache, cache_path)
        
    def load_cache(self, cache_path: str, cutoff_dim: int = None):
        logger.info("Loading Flair cache %s", cache_path)
        self.cache = torch.load(cache_path)
        self.cutoff_dim = cutoff_dim
        if cutoff_dim is None:
            self.embedding_dim = self.cache[
                list(self.cache.keys())[0]].shape[1]
        else:
            self.embedding_dim = cutoff_dim
    
    def __getitem__(self, s: str) -> Tensor:
        if self.cutoff_dim is None:
            return self.cache[s]
        else:
            return self.cache[s][:, :self.cutoff_dim]
