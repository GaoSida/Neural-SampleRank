"""Wrap the cpp implementations with an interface with the rest of model.
"""
from typing import Dict, List, Tuple

from torch import Tensor
import torch.nn as nn


class FactorGraphCpp(nn.Module):
    """FactorGraph interface to work with cpp implementations.
    It manages the models for the factors and the inference of their scores.
    """
    def __init__(self, factor_dependencies: Dict[str, List[Tuple[int]]],
                 factor_models: Dict[str, nn.Module]):
        """See nsr.graph.factor_graph.FactorGraph.
        """
        super().__init__()
        self.factor_dependencies = factor_dependencies
        # Register the models to this nn.Module
        for factor_type, model in factor_models.items():
            self.__setattr__(factor_type + "_factor_model", model)
        self.factor_models = factor_models
        
        self.factor_label_indices = dict()  # to index label states
        for factor_type, dependencies in factor_dependencies.items():
            if len(dependencies) == 0:  # These type of factor does not exist
                self.factor_label_indices[factor_type] = []
                continue
            self.factor_label_indices[factor_type] = \
                [[] for _ in range(len(dependencies[0]))]
            for nodes in dependencies:
                for i, idx in enumerate(nodes):
                    self.factor_label_indices[factor_type][i].append(idx)
    
    def forward(self, input_states: Tensor) -> Dict[str, Tensor]:
        """Evaluate all the factor Tensors.
        Args:
            input_states: states for each label node.
                shape: [num label_nodes, state_dim]
        Returns:
            keys are factor types, to the values of all factors.
        """
        factor_values = dict()
        for factor_type, model in self.factor_models.items():
            if not self.factor_label_indices[factor_type]:
                continue
            states = [input_states[indices]
                      for indices in self.factor_label_indices[factor_type]]
            factor_values[factor_type] = model(states)
        return factor_values
