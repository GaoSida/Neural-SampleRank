"""A helper for running inference callable multiple times, and ensemble the
predictions with a simple majority vote.
"""

from typing import Callable
from collections import Counter


def majority_vote_ensemble(eval_func: Callable, num_runs: int):
    """
    Args:
        eval_func: call without argument to get a prediction or
            a list of predictions.
        num_runs: how many times to run the eval_func to get the predictions
    Returns:
        a prediction or a list of predictions after majority vote.
    """
    if num_runs == 1:
        return eval_func()
    
    def _vote(prediction_list):
        results = list()
        for i in range(len(prediction_list[0])):
            votes = Counter([pred[i] for pred in prediction_list])
            results.append(votes.most_common(1)[0][0])
        return results
    
    all_predictions = [eval_func() for _ in range(num_runs)]
    if not isinstance(all_predictions[0][0], list):
        # eval func gives single prediction
        return _vote(all_predictions)
    else:
        # eval func gives a list of predictions
        results = list()
        for i in range(len(all_predictions[0])):
            results.append(_vote([pred_list[i]
                                  for pred_list in all_predictions]))
        return results
