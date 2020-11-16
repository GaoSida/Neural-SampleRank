"""Metrics used as margin in SampleRank loss, including negative Hamming 
distance etc. The metrics support local incremental updates, which is more
efficient when used with Gibbs sampling.
"""
import abc
from typing import List

from nsr.graph.factor_graph import LabelNode


class MarginMetric(abc.ABC):
    """Abstract margin metric. Efficient for computing metrics of a stream of
    prediction samples against the same ground truth.
    Metric can be of any number type, annotated as float for brevity.
    """
    def __init__(self, ground_truth: List[int]):
        super().__init__()
        self.ground_truth = ground_truth
        self.pad_idx = 1  # PyTorch convention: label is 1 for padded tokens
    
    @abc.abstractmethod
    def compute_metric(self, label_sample: List[int]) -> float:
        """Compute the metric of the label_sample compared to ground truth
        """
        pass
    
    @abc.abstractmethod
    def incremental_compute_metric(self, sample: List[LabelNode],
                                   prev_metric: float, prev_sample: List[int],
                                   diff_indices: List[int]) -> float:
        """Incrementally computes the metric of a label sample, leveraging the
        locality of label changes in the two consecutive samples.
        Args:
            sample: current labels, should be a reference into FactorGraph.
            prev_metric, prev_sample: metric and label of previous sample.
            diff_indices: label indices where sample and prev_sample differ.
        Returns: metric value of (current) sample.
        """
        pass


class NegHammingDistance(MarginMetric):
    """Negative Hamming distance. The higher the better.
    """
    def compute_metric(self, label_sample: List[int]) -> int:
        return - sum([(p != t and t != 1)
                      for p, t in zip(label_sample, self.ground_truth)])

    def incremental_compute_metric(self, sample: List[LabelNode],
                                   prev_metric: int, prev_sample: List[int],
                                   diff_indices: List[int]) -> int:
        # Assume the sampled block does not have padded tokens
        current_metric = prev_metric
        for i in diff_indices:
            # assert self.ground_truth[i] != 1  # Not <pad>
            prev_correct = (prev_sample[i] == self.ground_truth[i])
            current_correct = (sample[i].current_value == self.ground_truth[i])
            if prev_correct != current_correct:
                if prev_correct:
                    current_metric -= 1
                else:
                    current_metric += 1
        return current_metric
