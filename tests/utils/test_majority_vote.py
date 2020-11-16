from nsr.utils.majority_vote import majority_vote_ensemble


class DummyPredictor:
    def __init__(self, predictions: list):
        """Initialize with a list of predictions
        """
        self.current_idx = -1
        self.predictions = predictions
    
    def __call__(self):
        """Each time the object is called, return one predictions
        """
        self.current_idx += 1
        return self.predictions[self.current_idx]


def test_majority_vote_ensemble():
    predictor = DummyPredictor([
        [1, 2, 3, 4, 5],
        [2, 4, 3, 1, 5],
        [2, 2, 4, 1, 3]
    ])
    assert majority_vote_ensemble(predictor, 3) == [2, 2, 3, 1, 5]

    predictor = DummyPredictor([
        [[1, 2, 3], [1, 2], [1, 2, 3, 4]],
        [[3, 1, 2], [1, 2], [3, 4, 3, 1]],
        [[3, 2, 2], [1, 3], [1, 4, 2, 1]]
    ])
    assert majority_vote_ensemble(predictor, 3) == [
        [3, 2, 2], [1, 2], [1, 4, 3, 1]
    ]
