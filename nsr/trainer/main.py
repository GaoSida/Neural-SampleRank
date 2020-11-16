"""Entry point to run models.
"""
import os
import pprint
import logging
import argparse
from collections import Counter

import yaml
import torch
from torchtext.vocab import Vocab, GloVe, FastText
from torch.utils.data import ConcatDataset

import nsr.utils.conll_dataset as conll
import nsr.utils.collate_functions as collate
from nsr.trainer.unary_trainer import UnaryTrainer
from nsr.trainer.sample_rank_trainer import SampleRankTrainer
from nsr.trainer.linear_chain_trainer import LinearChainCRFTrainer
from nsr.graph.crf_builder import SeqTagCRFBuilder
from nsr.flair.embedding_cache import FlairEmbeddingCache

logger = logging.getLogger(__name__)


def main(config: dict):
    logger.info(pprint.pformat(config))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Setting up the vocabs
    if config["language"] == "en":
        logger.info("Loading glove embeddings...")
        glove = GloVe(name="6B", dim=config["glove_dim"], 
                      cache=config["glove_cache"],
                      unk_init=torch.Tensor.normal_)
        token_vocab = Vocab(Counter(list(glove.stoi.keys())), vectors=glove)
    else:
        logger.info("Loading fasttext {} embedding".format(config["language"]))
        fasttext = FastText(language=config["language"],
                            cache=config["fasttext_cache"],
                            unk_init=torch.Tensor.normal_)
        if not config["flair_enabled"]:
            token_vocab = Vocab(Counter(list(fasttext.stoi.keys())),
                                vectors=fasttext)
        else:
            token_vocab = Vocab(Counter(list(fasttext.stoi.keys())))
            del fasttext
    
    # Setting up the datasets
    logger.info("Loading datasets...")
    char_counter, label_counter = conll.count_conll_vocab(
        config["conll"]["train"])
    char_vocab = Vocab(char_counter, min_freq=2)
    label_vocab = Vocab(label_counter)
    
    doc_mode = False
    graph_builder = None
    if config["decoder"] in {"factor_graph", "linear_chain"}:
        doc_mode = True
        graph_builder = SeqTagCRFBuilder(config["skip_chain_enabled"])
    
    dataset_params = (token_vocab, char_vocab, label_vocab,
                      doc_mode, graph_builder)
    datasets = {
        "train": conll.ConllDataset(config["conll"]["train"], *dataset_params),
        "dev": conll.ConllDataset(config["conll"]["testa"], *dataset_params),
        "test": conll.ConllDataset(config["conll"]["testb"], *dataset_params)
    }
    logger.info("Train set size %s; dev size %s; test size %s",
                len(datasets["train"]), len(datasets["dev"]),
                len(datasets["test"]))
    
    if config["flair_compute_cache"]:
        FlairEmbeddingCache().compute_cache(datasets, config["language"],
                                            config["flair_cache_path"])
        return
    
    if config["max_train_size"] is not None:
        datasets["train"] = datasets["train"][:config["max_train_size"]]
    if config["train_with_dev"]:
        datasets["train"] = ConcatDataset([datasets["train"], datasets["dev"]])
    
    # Prepare the artifact directory
    artifact_root = os.path.join(config["artifact_dir"], config["exp_id"])
    if not os.path.exists(artifact_root):
        os.makedirs(artifact_root)
    else:
        logging.warning("Artifact root exists. Overwriting old artifacts!")
    
    # Create trainer and enter the main training loop
    if config["decoder"] == "unary":
        trainer = UnaryTrainer(config, token_vocab.vectors,
                               len(char_vocab), len(label_vocab), device)
        trainer.train(datasets, collate.generate_sentence_batch)
    elif config["decoder"] == "factor_graph":
        trainer = SampleRankTrainer(config, token_vocab.vectors,
                                    len(char_vocab), len(label_vocab), device)
        if config["load_test"] or config["continue_from_checkpoint"]:
            model_path = os.path.join(config["artifact_dir"],
                                      config["model_load_id"])
            logger.info("Loading model from %s", model_path)
            trainer.model.load_state_dict(torch.load(model_path))
        
        if config["load_test"]:
            for i in range(config["repeat_eval_runs"]):
                logger.info("Evaluation Run #{}".format(i + 1))
                trainer.eval(datasets["dev"], collate.generate_document_batch,
                             os.path.join(
                                 artifact_root,
                                 "dev_predictions-{}.txt".format(i + 1)
                            ))
            
            if not config["ignore_test_set"]:
                trainer.eval(datasets["test"], collate.generate_document_batch,
                             os.path.join(artifact_root,
                                          "test_predictions.txt"))
        else:
            trainer.train(datasets, collate.generate_document_batch)
    elif config["decoder"] == "linear_chain":
        trainer = LinearChainCRFTrainer(
            config, token_vocab.vectors,
            len(char_vocab), len(label_vocab), device)
        
        if config["load_test"] or config["continue_from_checkpoint"]:
            model_path = os.path.join(config["artifact_dir"],
                                      config["model_load_id"])
            logger.info("Loading model from %s", model_path)
            trainer.model.load_state_dict(torch.load(model_path))
            
        if config["load_test"]:
            trainer.eval(datasets["dev"], collate.generate_document_batch,
                         os.path.join(artifact_root, "dev_predictions.txt"))
            trainer.eval(datasets["test"], collate.generate_document_batch,
                         os.path.join(artifact_root, "test_predictions.txt"))
        else:
            trainer.train(datasets, collate.generate_document_batch)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config) as fin:
        config = yaml.safe_load(fin)
    
    logging_level = logging.INFO
    if config["logging_level"] == "debug":
        logging_level = logging.DEBUG
    logging.basicConfig(
        filename=os.path.join("local", "logs", config["exp_id"] + ".log"),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level)
    console = logging.StreamHandler()
    console.setLevel(logging_level)
    logging.getLogger("").addHandler(console)
    
    main(config)
