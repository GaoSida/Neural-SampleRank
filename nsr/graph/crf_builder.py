"""Build CRFs, in dependency list format, during data loading time. 
The model structure, and interface, should be very task-specific.
"""
import itertools
from typing import List, Dict, Tuple


class SeqTagCRFBuilder:
    def __init__(self, skip_chain_enabled: bool):
        """Configure the CRF structure for sequence tagging tasks. 
        """
        self.skip_chain_enabled = skip_chain_enabled

    def __call__(self, doc: List[List[str]]) \
            -> Dict[str, List[Tuple[Tuple[int]]]]:
        """Build a skip chain model for NER task.
        Args:
            doc: list of sentences, and each sentence is a list of tokens
        Returns:
            Map factor type names to a list of factors of that type.
            Each factor is a list of label nodes, each node is represented as
            the sentence index then token index.
        """
        graph = {}
        # One unary factor for each token.
        # One transition binary factor for each adjacent token pairs.
        graph["unary"] = list()
        graph["transition"] = list()
        for sent_idx, sent in enumerate(doc):
            for token_idx in range(len(sent)):
                graph["unary"].append(((sent_idx, token_idx),))
                if token_idx < len(sent) - 1:
                    graph["transition"].append((
                        (sent_idx, token_idx), (sent_idx, token_idx + 1)
                    ))
        
        # One skip-chain binary factor for repeated capitalized token pairs
        if self.skip_chain_enabled:
            graph["skip"] = list()
            token_positions = dict()  # Token to list of token positions
            for sent_idx, sent in enumerate(doc):
                for token_idx, token in enumerate(sent):
                    if token[0].isupper():
                        if token in token_positions:
                            token_positions[token].append(
                                (sent_idx, token_idx))
                        else:
                            token_positions[token] = [
                                (sent_idx, token_idx)]
            for token, positions in token_positions.items():
                # Add a factor for each pair
                for combo in itertools.combinations(range(len(positions)), 2):
                    graph["skip"].append((positions[combo[0]],
                                          positions[combo[1]]))
        return graph
