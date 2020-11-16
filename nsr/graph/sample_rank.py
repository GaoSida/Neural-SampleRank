"""The abstract sample rank inference engine that operates on any factor graph.
Include a training criteria and an inference routine.

Note: both inference and training factor graph contains documents in a batch.
The graph for each document is not connected, thus maximizing score in a batch
is equivalent to maximizing score for each doc separately.
"""
import random
import logging
from typing import List, Tuple, Callable

from torch import Tensor
import torch.nn as nn

from nsr.graph.gibbs_sampling import gibbs_sample_step
from nsr.graph.factor_graph import FactorGraph
from nsr.graph.margin_metric import MarginMetric

logger = logging.getLogger(__name__)


class SampleRankLoss(nn.Module):
    """SampleRank loss.
    TODO: Support memorization of sample state from previous epoch. (Now we
    always start from a random initialization.)
    """
    def __init__(self, num_samples: int,
                 margin_metric: Callable[[List[int]], MarginMetric],
                 gold_loss_enabled: bool = True,
                 pair_loss_enabled: bool = True,
                 init_with_oracle: bool = False):
        """
        Args:
            num_samples: number of times to cycle through the sample pool.
            margin_metric: a constructor of class derived from MarginMetric.
            gold_loss_enabled: whether to include max margin loss compared to 
                ground truth.
            pair_loss_enabled: whether to include max margin loss compared
                between two consecutive samples.
            init_with_orcale: whether to use ground truth (with a certain match
                rate) as the initial sample.
        """
        super().__init__()
        self.num_samples = num_samples
        self.margin_metric = margin_metric
        self.gold_loss_enabled = gold_loss_enabled
        self.pair_loss_enabled = pair_loss_enabled
        self.init_with_oracle = init_with_oracle
    
    @staticmethod
    def max_margin_loss(margin: float, better_label_score: Tensor,
                        worse_label_score: Tensor) -> Tensor:
        """
        Args:
            margin: any number type. Should be positive.
            better_label_score, worse_label_score: score of two labels,
                better or worse is w.r.t the ground truth label.
        """
        return max(0, margin - (better_label_score - worse_label_score))
    
    def forward(self, factor_graph: FactorGraph, sample_pool: List[Tuple[int]],
                ground_truth: List[int]) -> Tensor:
        """Take Gibbs samples with the factor graph, and construct loss.
        Args:
            factor_graph: A FactorGraph object to learn.
            sample_pool: The pool of label (RV) nodes to sampling from.
            ground_truth: target label values for each label node. 
        Returns: scalar loss.
        """
        metric = self.margin_metric(ground_truth)
        gold_score = factor_graph.compute_score_on_labels(ground_truth)
        gold_metric = metric.compute_metric(ground_truth)
        
        # Keep track of states of the previous sample
        if self.init_with_oracle:
            prev_score = factor_graph.random_init_labels(ground_truth,
                                                         random.random())
        else:
            prev_score = factor_graph.random_init_labels()
        prev_sample = factor_graph.get_label_values()
        prev_metric = metric.compute_metric(prev_sample)
        
        loss_terms = list()
        logger.debug("Gold: %s", gold_metric)
        for sample_idx in range(self.num_samples):
            logger.debug("Train Sample %s metric: %s", sample_idx, prev_metric)
            random.shuffle(sample_pool)  # Shuffle the sample pool then cycle
            for block in sample_pool:
                gibbs_sample_step(factor_graph, block)
                current_metric = metric.incremental_compute_metric(
                    factor_graph.label_nodes, prev_metric, prev_sample, block)
                
                if self.gold_loss_enabled and current_metric < gold_metric:
                    loss_terms.append(self.max_margin_loss(
                        gold_metric - current_metric, gold_score,
                        factor_graph.current_score))
                
                if self.pair_loss_enabled:
                    if current_metric > prev_metric:  # Current label is better
                        loss_terms.append(self.max_margin_loss(
                            current_metric - prev_metric,
                            factor_graph.current_score, prev_score))
                    elif current_metric < prev_metric:  # prev label is better
                        loss_terms.append(self.max_margin_loss(
                            prev_metric - current_metric,
                            prev_score, factor_graph.current_score))
                
                prev_score = factor_graph.current_score
                prev_metric = current_metric
                for i in block:
                    prev_sample[i] = factor_graph.label_nodes[i].current_value
        
        return sum(loss_terms) / len(loss_terms)
