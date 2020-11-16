"""Factor Graph model:
    A bipartite graph with FactorNodes and LabelNodes.
The FactorGraph module is agnostic of the inference method.
"""
import random
from typing import List, Dict, Tuple

from torch import Tensor
import torch.nn as nn


class LabelNode:
    """LabelNode, i.e. random variable nodes, in the graph.
    One or more factor nodes touch on a label node for scoring.
    """
    node_id: int  # ID used to identify edges
    label_space_size: int  # The label space for this node: 0, 1, ... size - 1
    current_value: int
    factor_nodes: List["FactorNode"]  # Forward reference

    def __init__(self, node_id: int, label_space_size: int):
        self.node_id = node_id
        self.label_space_size = label_space_size
        self.current_value = 0
        self.factor_nodes = list()


class FactorNode:
    """FactorNode in the graph, i.e. the potentials / factors
    which touches multiple LabelNodes for scoring.
    """
    label_nodes: List[LabelNode]
    score_table: Tensor  # len(score_table.shape) == len(label_nodes)
    # score_table is indexable by label_nodes values
    current_score: Tensor
    
    def __init__(self, label_nodes: List[LabelNode]):
        self.label_nodes = label_nodes
        self.score_table_shape = [node.label_space_size
                                  for node in label_nodes]
        self.score_table = None
        self.current_score = None
    
    def set_score_table(self, flat_table: Tensor):
        """Set the score_table with a flattened table after reshaping.
        """
        self.score_table = flat_table.view(self.score_table_shape)
    
    def update_score(self) -> Tensor:
        """Update the factor node score based on the label node values.
        Returns: the updated score.
        """
        self.current_score = self.score_table[
            tuple(node.current_value for node in self.label_nodes)]
        return self.current_score


class FactorGraph(nn.Module):
    """Collection of nodes in the FactorGraph.
    """
    label_nodes: List[LabelNode]  # Node id is the index in this list
    
    factor_models: Dict[str, nn.Module]  # factor node type to scoring model
    factor_nodes: Dict[str, List[FactorNode]]  # factor nodes under each type
    
    current_score: Tensor  # Sum of factor scores based on the labels
    
    def __init__(self, label_space_sizes: List[int],
                 factor_dependencies: Dict[str, List[Tuple[int]]],
                 factor_models: Dict[str, nn.Module]):
        """Build the factor graph
        Args:
            label_space_sizes: for each label in that order.
            factor_dependencies: key - factor type;
                Each type has a list of label node dependencies,
                where the label node ids are stored in a tuple.
            factor_models: the Pytorch model to compute the factor scores.
        """
        super().__init__()
        self.factor_dependencies = factor_dependencies
        # Register the models to this nn.Module
        for factor_type, model in factor_models.items():
            self.__setattr__(factor_type + "_factor_model", model)
        self.factor_models = factor_models
        
        # Create the label nodes
        self.label_nodes = list()
        for i, n in enumerate(label_space_sizes):
            self.label_nodes.append(LabelNode(i, n))
        
        # Create the factor nodes
        self.factor_nodes = dict()
        self.factor_label_indices = dict()  # to index label states
        for factor_type, dependencies in factor_dependencies.items():
            self.factor_nodes[factor_type] = list()
            if len(dependencies) == 0:  # These type of factor does not exist
                self.factor_label_indices[factor_type] = []
                continue
            self.factor_label_indices[factor_type] = \
                [[] for _ in range(len(dependencies[0]))]
            for nodes in dependencies:
                new_factor_node = FactorNode([self.label_nodes[i]
                                              for i in nodes])
                self.factor_nodes[factor_type].append(new_factor_node)
                for i, idx in enumerate(nodes):
                    self.label_nodes[idx].factor_nodes.append(new_factor_node)
                    self.factor_label_indices[factor_type][i].append(idx)
        
        self.current_score = None

    def forward(self, input_states: Tensor, force_cpu: bool = True) -> None:
        """Evaluate all the factor Tensors update the nodes
        Args:
            input_states: states for each label node.
                shape: [len(self.label_nodes), state_dim]
            force_cpu: whether we force score tables to stay on CPU.
        """
        for factor_type, model in self.factor_models.items():
            if not self.factor_label_indices[factor_type]:
                continue
            states = [input_states[indices]
                      for indices in self.factor_label_indices[factor_type]]
            all_factors = model(states)
            # Each row is the factor scores of one factor node
            for i, node in enumerate(self.factor_nodes[factor_type]):
                # Push score tensors to CPU can be much faster for sampling
                if force_cpu:
                    node.set_score_table(all_factors[i].to("cpu"))
                else:
                    node.set_score_table(all_factors[i])
    
    def set_label_values(self, label_vals: List[int]) -> Tensor:
        """Set the values of each label, and update the score.
        Args:
            label_vals: must have len(label_vals) == len(self.label_nodes)
        Returns: current score after updating label values.
        """
        for i, val in enumerate(label_vals):
            self.label_nodes[i].current_value = val
        
        factor_scores = list()
        for _, factors in self.factor_nodes.items():
            for factor in factors:
                factor.update_score()
                factor_scores.append(factor.current_score)
        self.current_score = sum(factor_scores)
        return self.current_score
    
    def get_label_values(self) -> List[int]:
        return [node.current_value for node in self.label_nodes]
            
    def random_init_labels(self, ground_truth: List[int] = None,
                           p: float = 0.0) -> Tensor:
        """Initializing the labels uniformly random, and initialize the score.
        Returns: current score on the initialized labels.
        Args:
            ground_truth: the ground truth label values.
            p: the probability to initialize with ground truth.
        """
        label_vals = list()
        for node in self.label_nodes:
            if ground_truth is None or random.random() > p:
                label_vals.append(random.randint(0, node.label_space_size - 1))
            else:
                label_vals.append(ground_truth[node.node_id])
        return self.set_label_values(label_vals)
    
    def compute_score_on_labels(self, label_values: List[int]) -> Tensor:
        """Given the values of labels, return the score of graph.
        This method does not change the label state of the graph.
        """
        factor_scores = list()
        for _, factors in self.factor_nodes.items():
            for factor in factors:
                factor_label_values = [label_values[node.node_id]
                                       for node in factor.label_nodes]
                factor_scores.append(factor.score_table[
                    tuple(factor_label_values)])
        return sum(factor_scores)
