"""Standard Gibbs sampling algorithm on a Factor Graph.
- Compute conditional scores on a factor given a set of variables (i.e. labels)
- Take one Gibbs sampling step
- Gibbs sampling inference with annealing
"""
import random
import logging
from typing import List, Tuple, Callable

import numpy as np
import torch
from torch import Tensor

from nsr.graph.factor_graph import FactorNode, FactorGraph
from nsr.graph.margin_metric import MarginMetric

logger = logging.getLogger(__name__)


def compute_conditional_scores(factor_node: FactorNode,
                               label_ids: List[int]) -> Tensor:
    """Compute the conditional score table of the given label ids on a factor,
    conditioned on everything else.
    Args:
        label_ids: we are interested in these labels' distribution conditioned
        on every other label in the graph.
        This list of labels should have a non-empty intersection with
        self.label_nodes.
    Returns:
        a Tenor that has len(label_ids) dimensions, corresponding to each
        label's conditional score (i.e. in the same order):
        - The dimension is 1 if that label is not in self.label_nodes
        - Otherwise, the dimension is size of label space.
    """
    label_id_map = {label_ids[i]: i for i in range(len(label_ids))}
    # Slice the scores based on the intersection of label nodes to be sampled
    # and the label nodes that this factor touches
    score_slice_index = list()
    # The output dimension index for each node in the intersection label set.
    # This order may not agree with self.score_table
    output_dim_index = list()
    output_shape = [1] * len(label_ids)
    for node in factor_node.label_nodes:
        if node.node_id in label_id_map:
            score_slice_index.append(slice(None))
            output_dim_index.append(label_id_map[node.node_id])
            output_shape[label_id_map[node.node_id]] = \
                node.label_space_size
        else:
            score_slice_index.append(node.current_value)
    conditioned_scores = factor_node.score_table[tuple(score_slice_index)]
    # Number of dimensions is the size of intersection label set
    # The order of dimensions is according to self.label_nodes, and
    # we need to permute according to the order in labeld_ids
    if output_dim_index:
        conditioned_scores = conditioned_scores.permute(
            np.argsort(output_dim_index).tolist())
    # else: intersection label set is empty, conditional_scores is scalar
    # If the permute call is effective, then the tensor is no longer
    # contiguous, hence we need to use reshape here (i.e. may copy).
    return conditioned_scores.reshape(output_shape)


def gibbs_sample_step(factor_graph: FactorGraph, label_ids: Tuple[int],
                      temp: float = 1.0) -> Tensor:
    """Take one gibbs sample step in the block of label nodes.
    The size of the block can be 1, i.e. the standard Gibbs sampling.
    Args:
        label nodes: the label ID block to be sampled.
        temp: the temperature used in computing the distribution.
    Returns: Un-normalized conditional score after applying temperature.
    """
    factor_nodes_set = set()
    for idx in label_ids:
        factor_nodes_set.update(factor_graph.label_nodes[idx].factor_nodes)
    conditional_scores = list()
    for factor in factor_nodes_set:
        conditional_scores.append(compute_conditional_scores(factor, 
                                                             label_ids))
    conditional_scores = sum(conditional_scores) / temp
    return_val = conditional_scores
    # len(conditional_scores.shape) == len(label_ids)
    # Size of each dimension is label space size of each label node
    
    conditional_scores = conditional_scores.flatten()
    # Minus the max to avoid overflow when normalizing
    conditional_scores = torch.exp(conditional_scores -
                                   torch.max(conditional_scores))
    conditional_scores = conditional_scores / torch.sum(conditional_scores)
    conditional_scores = conditional_scores.to("cpu").detach().numpy()
    next_sample = np.random.choice(range(conditional_scores.shape[0]),
                                   p=conditional_scores)
    
    # Parse the next_sample index into the index of each label
    for idx in label_ids[::-1]:
        node = factor_graph.label_nodes[idx]
        node.current_value = next_sample % node.label_space_size
        next_sample = next_sample // node.label_space_size
    # Update the scores
    for factor in factor_nodes_set:
        factor_graph.current_score -= factor.current_score
        factor.update_score()
        factor_graph.current_score += factor.current_score

    return return_val


def gibbs_sampling_inference(factor_graph: FactorGraph,
                             sample_pool: List[Tuple[int]],
                             num_samples: int, init_temp: float,
                             anneal_rate: float, min_temp: float,
                             metric: Callable[[List[int]],
                                              MarginMetric] = None,
                             ground_truth: List[int] = None) -> List[int]:
    """Gibbs sampling for inference, not specific to SampleRank.
    TODO: see if the inference routine can support multi-processing.
    Args:
        factor_graph: A FactorGraph object to optimize.
        sample_pool: The pool of label (RV) nodes to sampling from.
            A list of RV blocks, the size of the block can be 1.
        num_samples: number of times to cycle through the sample pool.
        init_temp: the initial temperature in sampling.
        anneal_rate: multiplied to the temperature after each sample.
        min_temp: stop annealing at this temperature.
        metric, ground_truth: for debug logging. See nsr.graph.sample_rank
    Returns:
        A list of label values for the graph.
    """
    factor_graph.random_init_labels()
    max_score = factor_graph.current_score.item()
    best_label = [node.current_value for node in factor_graph.label_nodes]
    
    is_debugging = (logging.getLogger().level == logging.DEBUG) and \
        (metric is not None) and (ground_truth is not None)
    if is_debugging:
        if type(ground_truth) == Tensor:
            ground_truth = ground_truth.flatten().tolist()
        metric_tracker = metric(ground_truth)
        logger.debug("Inference metric: %s",
                     metric_tracker.compute_metric(best_label))
    
    temperature = init_temp
    for sample_idx in range(num_samples):
        random.shuffle(sample_pool)  # Shuffle the sample pool then cycle it
        for block in sample_pool:
            gibbs_sample_step(factor_graph, block, temperature)
            current_score = factor_graph.current_score.item()
            if current_score > max_score:
                max_score = current_score
                best_label = [node.current_value
                              for node in factor_graph.label_nodes]
        if is_debugging:
            logger.debug("Inf Sample %s metric %s", sample_idx + 1,
                         metric_tracker.compute_metric(best_label))
        # Take an annealing step
        temperature = max(min_temp, temperature * anneal_rate)

    return best_label
