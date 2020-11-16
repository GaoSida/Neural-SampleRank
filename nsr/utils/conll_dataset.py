"""
Load CoNLL data with PyTorch Datasets and Iterator utilities
"""
import tempfile
import logging
import subprocess
from collections import Counter
from typing import Dict, List, Tuple, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

logger = logging.getLogger(__name__)


def count_conll_vocab(conll_file: str) -> Tuple[Counter]:
    """Construct the counters from a CoNLL data file.
    Args:
        conll_file: each line's first field is the token, last is label.
    Returns:
        char_counter, label_counter
    """
    char_counter = Counter()
    label_counter = Counter()
    with open(conll_file) as fin:
        for line in fin:
            line = line.strip().split()
            if len(line) > 0:
                token = line[0]
                char_counter.update(token)  # Count characters in string
                label = line[-1]
                label_counter.update([label])  # Full string to count
    return char_counter, label_counter


class ConllDataset(Dataset):
    """CoNLL format dataset like CoNLL 2003. The dataset is kept as a 
    collection of sentences or a collection of documents.
    """
    def __init__(self, conll_file: str, token_vocab: Vocab, char_vocab: Vocab,
                 label_vocab: Vocab, doc_mode: bool = False, graph_builder:
                     Callable[[List[List[str]]],
                              Dict[str, List[Tuple[Tuple[int]]]]] = None):
        """
        Args:
            conll_file: path to the CoNLL data file.
            token_vocab, char_vocab, label_vocab: Vocabularies for mapping.
            doc_mode: if True, each dataset item will be document.
            graph_builder: Given the document string, build graph as a
                dependency map. See nsr.graph.factor_graph.crf_builder.
        """
        doc_start_token = "-DOCSTART-"
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        # The dataset is small so we load everything directly in memory
        self.dataset = list()
        
        def _empty_buffer():
            return {"tokens": list(), "token_chars": list(),
                    "labels": list(), "strings": list()}
        
        def _append_sent(doc_buffer, sent_buffer):
            if len(sent_buffer["tokens"]) == 0:
                return
            if doc_mode:
                for key in doc_buffer:
                    doc_buffer[key].append(sent_buffer[key])
            else:
                self.dataset.append(sent_buffer)
        
        def _append_doc(doc_buffer):
            if len(doc_buffer["tokens"]) == 0:
                return
            if graph_builder:
                doc_buffer["graph"] = graph_builder(doc_buffer["strings"])
            self.dataset.append(doc_buffer)
        
        with open(conll_file) as fin:
            document_buffer = _empty_buffer()
            sentence_buffer = _empty_buffer()
            for line in fin:
                line = line.strip().split()
                if len(line) == 0:
                    _append_sent(document_buffer, sentence_buffer)
                    sentence_buffer = _empty_buffer()
                else:
                    token, label = line[0], line[-1]
                    if token == doc_start_token:
                        _append_doc(document_buffer)
                        document_buffer = _empty_buffer()
                    # Lowercase the token to query token embedding
                    sentence_buffer["tokens"].append(
                        token_vocab.stoi[token.lower()])
                    # Keep the original case for char embedding
                    sentence_buffer["token_chars"].append(
                        [char_vocab.stoi[c] for c in token])
                    sentence_buffer["labels"].append(label_vocab.stoi[label])
                    sentence_buffer["strings"].append(token)

            # When there is no trailing empty new lines in the end
            if len(sentence_buffer["tokens"]) > 0:
                _append_sent(document_buffer, sentence_buffer)
            # Clear doc buffer as we won't have doc_start_token in the end
            if len(document_buffer["tokens"]) > 0:
                _append_doc(document_buffer)
        
        # Convert from list to tensor
        for entry in self.dataset:
            if doc_mode:
                entry["tokens"] = [torch.tensor(s) for s in entry["tokens"]]
                entry["token_chars"] = [[torch.tensor(t) for t in s]
                                        for s in entry["token_chars"]]
                entry["labels"] = [torch.tensor(s) for s in entry["labels"]]
            else:
                entry["tokens"] = torch.tensor(entry["tokens"])
                entry["token_chars"] = [torch.tensor(t)
                                        for t in entry["token_chars"]]
                entry["labels"] = torch.tensor(entry["labels"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def write_prediction(self, tokens: Tensor, labels: Tensor,
                         predictions: Tensor) -> str:
        """Write predictions in the format that's compatible with conlleval.pl:
            The final two items should contain the correct tag and the 
            guessed tag in that order.
        Args:
            tokens: token indices for the data. [batch_size, max_num_tokens]
            labels: label indices for each token, in the same shape.
            predictions: predictedlabel indices for each token, same shape.
        Returns:
            a string that is the subsection of file for this batch.
            It's up to the caller to assemble the strings.
        """
        
        def _convert_biose_to_bio(tag):
            if tag.startswith("E"):
                return "I" + tag[1:]
            elif tag.startswith("S"):
                return "B" + tag[1:]
            else:
                return tag
        
        tokens, labels, predictions = \
            tokens.tolist(), labels.tolist(), predictions.tolist()
        
        batch_size = len(tokens)
        max_num_tokens = len(tokens[0])
        result_str = ""
        for i in range(batch_size):
            for j in range(max_num_tokens):
                if tokens[i][j] != 1:  # Not padding
                    # We may get <UNK> but it's okay for evaluation
                    result_str += self.token_vocab.itos[tokens[i][j]] + " " + \
                        _convert_biose_to_bio(
                            self.label_vocab.itos[labels[i][j]]) + " " + \
                        _convert_biose_to_bio(
                            self.label_vocab.itos[predictions[i][j]]) + "\n"
            result_str += "\n"
        return result_str


def compute_f1(prediction_dump: str, conlleval_file: str,
               dump_file: str = "") -> float:
    """Call out to conlleval.pl to get the F-1 score.
    Args:
        prediction_dump: the string content of CoNLL format file.
        conlleval_file: path to the perl evaluation script.
        dump_file: If empty, use tmp file.
    Return:
        the F1 score computed by conlleval.pl
    """
    if dump_file:
        fout = open(dump_file, "w")
        filename = dump_file
    else:
        fout = tempfile.NamedTemporaryFile("w")
        filename = fout.name
            
    fout.write(prediction_dump)
    fout.flush()
    output = subprocess.check_output(
        "perl {} < {}".format(conlleval_file, filename), shell=True)
    fout.close()
    
    output = output.decode("utf-8")
    lines = str(output).split("\n")
    logger.info("Summary: %s", lines[0])
    logger.info("F metrics: %s", lines[1])
    
    # F-1 score is in the last chunk
    return eval(lines[1].split()[-1])
