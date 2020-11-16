"""Assemble the Factor Graph model with neural scoring factors.
Train the model with Sample Rank loss and inference with Gibbs Sampling.
"""
import os
import time
import copy
import logging
import uuid
from typing import Dict, List, Tuple, Callable, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from nsr.model.token_embedding import CharCNN, TokenEmbedding
from nsr.model.flair_embeddings import FlairEmbeddings
from nsr.model.sentence_encoder import BiRNN
from nsr.model.unary_factor import UnaryFactor
from nsr.model.high_order_factor import HighOrderFactor
from nsr.graph.factor_graph import FactorGraph
from nsr.graph.gibbs_sampling import gibbs_sampling_inference
from nsr.graph.sample_rank import SampleRankLoss
from nsr.graph.margin_metric import MarginMetric, NegHammingDistance
import nsr.utils.conll_dataset as conll
from nsr.utils.majority_vote import majority_vote_ensemble
import cpp_sampling
from cpp_sampling import gibbs_sampling_inference_multithreading as \
    gibbs_sampling_inference_mth
from nsr.graph_cpp.factor_graph_cpp import FactorGraphCpp
from nsr.graph_cpp.sample_rank_cpp import SampleRankLossCpp
from nsr.flair.embedding_cache import FlairEmbeddingCache

logger = logging.getLogger(__name__)


class SkipChainCRFModel(nn.Module):
    """Neural factors for skip-chain CRF model for sequence labeling.
    """
    def __init__(self, config: dict, pretrained_embedding: Tensor,
                 char_vocab_size: int, label_vocab_size: int):
        """Assemble the factor model according to the config.
        Args:
            config: model configs and hyperparameters.
            pretrained_embedding: token embedding.
            char_vocab_size: size of the character vocabulary.
            label_vocab_size: size of the label space, i.e. output dim.
        """
        super().__init__()
        self.is_debugging = (config["logging_level"] == "debug")
        self.debug_root = os.path.join(config["artifact_dir"],
                                       config["exp_id"], "debug")
        os.makedirs(self.debug_root, exist_ok=True)
        self.cpp_sampling = config["cpp_sampling"]
        self.flair_enabled = config["flair_enabled"]
        self.label_vocab_size = label_vocab_size
        
        if self.flair_enabled:
            flair_cache = None
            if config["flair_cache_enabled"]:
                flair_cache = FlairEmbeddingCache()
                flair_cache.load_cache(config["flair_cache_path"],
                                       config["flair_cutoff_dim"])
            self.flair_embedding = FlairEmbeddings(config["language"],
                                                   flair_cache)
            total_embedding_dim = self.flair_embedding.embedding_dim
            
            self.flair_with_char_cnn = config["flair_with_char_cnn"]
            if self.flair_with_char_cnn:
                self.char_cnn = CharCNN(
                    char_vocab_size, config["char_embed_dim"],
                    config["char_kernel_size"], config["char_num_kernels"],
                    config["dropout"]
                )
                total_embedding_dim += config["char_num_kernels"]
        else:
            char_cnn = CharCNN(
                char_vocab_size, config["char_embed_dim"],
                config["char_kernel_size"], config["char_num_kernels"],
                config["dropout"]
            )
            self.token_embedding = TokenEmbedding(pretrained_embedding,
                                                  char_cnn)
            total_embedding_dim = config["glove_dim"] + \
                config["char_num_kernels"]
        
        self.bi_rnn = BiRNN(
            total_embedding_dim,
            config["rnn_hidden_dim"], config["rnn_num_layers"],
            config["dropout"], config["rnn_type"],
            config["embed_dropout"], config["word_dropout"],
            config["locked_dropout"]
        )
        
        # Factor model keys work with nsr.graph.crf_builder
        self.factor_model_dict = dict()
        self.state_dim = 2 * config["rnn_hidden_dim"]
        self.unary_factor = UnaryFactor(
            self.state_dim, label_vocab_size,
            config["unary_hidden_dim"], config["dropout"]
        )
        self.factor_model_dict["unary"] = self.unary_factor
        feature_ops = [lambda x: torch.sum(x, dim=0),
                       lambda x: torch.max(x, dim=0)[0]]
        self.transition_factor = HighOrderFactor(
            self.state_dim, 2, feature_ops, label_vocab_size ** 2,
            config["binary_hidden_dim"], config["dropout"]
        )
        self.factor_model_dict["transition"] = self.transition_factor
        if config["skip_chain_enabled"]:
            self.skip_chain_factor = HighOrderFactor(
                self.state_dim, 2, feature_ops, label_vocab_size ** 2,
                config["binary_hidden_dim"], config["dropout"]
            )
            self.factor_model_dict["skip"] = self.skip_chain_factor
        
    def forward(self, tokens: Tensor, token_chars: Tensor, lengths: Tensor,
                dependencies: Dict[str, List[Tuple[int]]], strings: List[str])\
            -> Union[FactorGraph, Tuple[FactorGraphCpp, Dict[str, Tensor]]]:
        """
        Args: (batch_size is number of sentences in this batch of documents)
            tokens: the token indices. [batch_size, max_num_tokens]
            token_chars: the character indices for each token.
                Shape [batch_size * max_num_tokens, max_num_chars]
            lengths: length of each sentence in batch. shape [batch_size, ]
            dependencies: See nsr.graph.factor_graph
            strings: str sentences, len(strings) == batch_size
        Returns:
            FactorGraph object after inference, OR
            the FactorGraphCpp object (structure only) with factor values
        """
        if self.flair_enabled:
            embeddings = self.flair_embedding(strings)
            if self.is_debugging:
                debug_id = uuid.uuid4()
                logger.debug("Flair embedding requires grad: %s, saved as %s",
                             embeddings.requires_grad, debug_id)
                torch.save(embeddings, os.path.join(
                    self.debug_root, "{}-flair.pt".format(debug_id)))
            if self.flair_with_char_cnn:
                char_embeddings = self.char_cnn(token_chars)
                char_embeddings = char_embeddings.view(
                    embeddings.shape[0], embeddings.shape[1],
                    char_embeddings.shape[1]
                )
                embeddings = torch.cat([embeddings, char_embeddings], dim=2)
        else:
            embeddings = self.token_embedding(tokens, token_chars)
        # shape: [batch_size, max_num_tokens, token_embedding_dim]
        rnn_output = self.bi_rnn(embeddings, lengths)
        if self.is_debugging:
            logger.debug("RNN output shape %s, saved as %s",
                         rnn_output.shape, debug_id)
            torch.save(rnn_output, os.path.join(self.debug_root,
                                                "{}-rnn.pt".format(debug_id)))
        # shape: [batch_size, max_num_tokens, 2 * rnn_hidden_dim]
        states = rnn_output.view(-1, self.state_dim)
        # shape: [batch_size * max_num_tokens, 2 * rnn_hidden_dim]
        if self.cpp_sampling:
            factor_graph = FactorGraphCpp(dependencies, self.factor_model_dict)
            return factor_graph, factor_graph(states)
        else:
            factor_graph = FactorGraph([self.label_vocab_size]
                                       * states.shape[0], dependencies,
                                       self.factor_model_dict)
            factor_graph(states)
            return factor_graph


class SampleRankTrainer:
    """Train a neural FactorGraph model with SampleRank, then evaluate with
    standard Gibbs sampling.
    """
    def __init__(self, config: dict, pretrained_embedding: Tensor,
                 char_vocab_size: int, label_vocab_size: int,
                 device: torch.device):
        """See SkipChainCRFModel for the semantics of arguments.
        """
        self.config = config
        self.device = device
        self.model = SkipChainCRFModel(
            config, pretrained_embedding, char_vocab_size, label_vocab_size
        ).to(self.device)
        logger.info("Model: \n" + str(self.model))
        
        self.label_vocab_size = label_vocab_size
    
    def get_sample_pool(self, factor_graph: FactorGraph) -> List[Tuple[int]]:
        """Get the pool of label node blocks to sample from.
        """
        sample_pool = copy.deepcopy(factor_graph.factor_dependencies["unary"])
        if self.config["block_sampling"]:
            sample_pool += factor_graph.factor_dependencies["skip"]
        return sample_pool
    
    def eval(self, dataset: Dataset, collate_fn: Callable, 
             dump_file: str = "") -> float:
        """Evaluate the model on one dataset.
        Args:
            datasets: one dataset
            collate_fn: batch preprocessing to be passed into DataLoader
            dump_file: path to dump predictions.
        Returns: F1 score
        """
        config = self.config
        device = self.device
        num_proceses = config["inf_num_processes"]
        if config["dump_history"]:
            dump_root = os.path.join(config["artifact_dir"], config["exp_id"])
        else:
            dump_root = ""
        
        data = DataLoader(dataset[:config["max_eval_size"]],
                          batch_size=config["batch_size"],
                          collate_fn=collate_fn)
        
        self.model.eval()  # Switch to evaluation mode
        prediction_dump = ""
        start = time.perf_counter()
        total_num_tokens = 0
        sampling_input_cache = list()
        prediction_dump_cache = list()
        num_batches = 0
        for tokens, token_chars, lengths, dependencies, strings, labels \
                in data:
            tokens, token_chars, lengths = \
                tokens.to(device), token_chars.to(device), lengths.to(device)
            total_num_tokens += torch.sum(lengths).item()
            num_batches += 1
            
            if config["cpp_sampling"]:
                factor_graph, factor_values = self.model(
                    tokens, token_chars, lengths, dependencies, strings)
                # Move Tensors to CPU before stepping into Cpp
                for factor_type in factor_values:
                    factor_values[factor_type] = \
                        factor_values[factor_type].to("cpu")
                sample_pool = self.get_sample_pool(factor_graph)
                
                if config["logging_level"] == "debug":
                    # The evaluation is expensive so we add extra condition
                    for factor_type in factor_values:
                        score_table = factor_values[factor_type]
                        logger.debug("%s factor shape %s, row 0 %s",
                                     factor_type, score_table.shape,
                                     score_table[0])
                
                if num_proceses == 1:
                    predictions = majority_vote_ensemble(
                        lambda: cpp_sampling.gibbs_sampling_inference(
                            [self.label_vocab_size] * token_chars.shape[0],
                            factor_values, dependencies, sample_pool,
                            config["inf_num_samples"], config["init_temp"],
                            config["anneal_rate"], config["min_temp"],
                            config["logging_level"] == "debug",
                            labels.flatten().tolist(), dump_root
                        ), num_runs=config["num_ensemble_runs"])
                    prediction_dump += dataset.write_prediction(
                        tokens, labels,
                        torch.tensor(predictions).view(labels.shape))
                else:
                    sampling_input_cache.append((
                        [self.label_vocab_size] * token_chars.shape[0],
                        factor_values, dependencies, sample_pool
                    ))
                    prediction_dump_cache.append((tokens, labels))
                    if len(sampling_input_cache) == num_proceses or \
                            num_batches == len(data):
                        predictions_list = majority_vote_ensemble(
                            lambda: gibbs_sampling_inference_mth(
                                [inputs[0] for inputs in sampling_input_cache],
                                [inputs[1] for inputs in sampling_input_cache],
                                [inputs[2] for inputs in sampling_input_cache],
                                [inputs[3] for inputs in sampling_input_cache],
                                config["inf_num_samples"], config["init_temp"],
                                config["anneal_rate"], config["min_temp"],
                                dump_root
                            ), num_runs=config["num_ensemble_runs"])
                        for (toks, lbls), predictions in zip(
                                prediction_dump_cache, predictions_list):
                            prediction_dump += dataset.write_prediction(
                                toks, lbls,
                                torch.tensor(predictions).view(lbls.shape))
                        sampling_input_cache = list()
                        prediction_dump_cache = list()
            else:
                factor_graph = self.model(tokens, token_chars, lengths,
                                          dependencies, strings)
                sample_pool = self.get_sample_pool(factor_graph)
                predictions = majority_vote_ensemble(
                    lambda: gibbs_sampling_inference(
                        factor_graph, sample_pool, config["inf_num_samples"],
                        config["init_temp"], config["anneal_rate"],
                        config["min_temp"], NegHammingDistance, labels
                    ), num_runs=config["num_ensemble_runs"])  # flat prediction
                prediction_dump += dataset.write_prediction(
                    tokens, labels,
                    torch.tensor(predictions).view(labels.shape))
        
            if num_batches % config["eval_print_freq"] == 0:
                logger.info("Eval progress %s/%s", num_batches, len(data))
        
        logger.info("Eval speed %.2f tokens per sec",
                    total_num_tokens / (time.perf_counter() - start))
        return conll.compute_f1(prediction_dump, config["conlleval_path"],
                                dump_file=dump_file)
            
    def train(self, datasets: Dict[str, Dataset], collate_fn: Callable):
        """Train the model and evaluate after every epoch.
        Args:
            datasets: keys {"train", "dev", "test"}
            collate_fn: batch preprocessing to be passed into DataLoader
        """
        config = self.config
        device = self.device
        artifact_root = os.path.join(config["artifact_dir"], config["exp_id"])

        if config["cpp_sampling"]:
            criterion = SampleRankLossCpp
        else:
            criterion = SampleRankLoss(
                config["train_num_samples"], NegHammingDistance,
                config["gold_loss_enabled"], config["pair_loss_enabled"],
                config["train_init_with_oracle"]
            )
        optimizer = optim.Adam(self.model.parameters(),
                               lr=config["learning_rate"])
        if config["lr_scheduler"]:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=5)
        
        best_dev_f1 = 0
        best_dev_f1_epoch = 0
        best_test_f1 = 0
        best_test_f1_epoch = 0
        best_model_path = os.path.join(artifact_root, "best_model.pth")
        last_model_path = os.path.join(artifact_root, "last_model.pth")
        for epoch in range(1, 1 + config["num_epochs"]):
            logging.info("Start training epoch %s", epoch)
            self.model.train()  # Turn on training mode (e.g. dropout)
            
            data = DataLoader(datasets["train"],
                              batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
            
            start = time.perf_counter()
            num_batches = 0
            total_num_tokens = 0
            accumulated_loss = 0.0
            for tokens, token_chars, lengths, dependencies, strings, labels \
                    in data:
                optimizer.zero_grad()
                
                num_batches += 1
                num_batch_tokens = torch.sum(lengths).item()
                total_num_tokens += num_batch_tokens
                logger.debug("Num batch tokens: %s", num_batch_tokens)
                logger.debug("Num sentences: %s", tokens.shape[0])
                logger.debug("Max sent length: %s", tokens.shape[1])
                
                tokens, token_chars, lengths = tokens.to(device), \
                    token_chars.to(device), lengths.to(device)
                
                if config["cpp_sampling"]:
                    factor_graph, factor_values = self.model(
                        tokens, token_chars, lengths, dependencies, strings)
                    factor_types = list()
                    factor_score_tables = list()
                    for factor_type in factor_values:
                        factor_types.append(factor_type)
                        factor_values[factor_type] = \
                            factor_values[factor_type].to("cpu")
                        factor_score_tables.append(factor_values[factor_type])
                    sample_pool = self.get_sample_pool(factor_graph)
                    
                    loss = criterion.apply(
                        config, [self.label_vocab_size] * token_chars.shape[0],
                        dependencies, sample_pool, labels.view(-1).tolist(),
                        factor_types, *factor_score_tables
                    )
                else:
                    factor_graph = self.model(tokens, token_chars, lengths,
                                              dependencies, strings)
                    sample_pool = self.get_sample_pool(factor_graph)
                    loss = criterion(factor_graph, sample_pool,
                                     labels.view(-1).tolist())
                
                accumulated_loss += loss.item() * num_batch_tokens
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         config["gradient_clip"])
                optimizer.step()
                
                if num_batches % config["print_freq"] == 0:
                    logger.info(
                        "Epoch %s, Progress: %s, Loss: %.3f, "
                        "speed: %.2f tokens per sec", epoch,
                        "{}/{}".format(num_batches, len(data)),
                        accumulated_loss / total_num_tokens,
                        total_num_tokens / (time.perf_counter() - start)
                    )

            if config["train_with_dev"]:
                # As a reference, should not rely on this result too much
                # Save all models
                test_f1 = self.eval(
                    datasets["test"], collate_fn,
                    os.path.join(artifact_root,
                                 "test_predictions_{}.txt".format(epoch)))
                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
                    best_test_f1_epoch = epoch
                    torch.save(self.model.state_dict(), best_model_path)
                logger.info("Best test F1 {} from epoch {}".format(
                    best_test_f1, best_test_f1_epoch))
            else:
                logger.info("Evaluating on dev set...")
                dev_f1 = self.eval(datasets["dev"], collate_fn)
                if dev_f1 > best_dev_f1:
                    logger.info("Saving model - best so far...")
                    best_dev_f1 = dev_f1
                    best_dev_f1_epoch = epoch
                    torch.save(self.model.state_dict(), best_model_path)
                    self.eval(datasets["test"], collate_fn)
            if config["save_all_checkpoints"]:
                torch.save(self.model.state_dict(),
                           os.path.join(artifact_root, 
                                        "model_epoch-{}.pth".format(epoch)))
            else:
                torch.save(self.model.state_dict(), last_model_path)
            
            if config["lr_scheduler"]:
                train_loss = accumulated_loss / total_num_tokens
                scheduler.step(train_loss)
                logger.info("Train loss this epoch: %.4f, new lr %s",
                            train_loss, optimizer.param_groups[0]['lr'])
        
        if not config["train_with_dev"]:
            # Re-evaluate the best model and dump predictions
            self.model.load_state_dict(torch.load(best_model_path))
            logger.info("Evaluating best model on dev from epoch %s ...",
                        best_dev_f1_epoch)
            self.eval(datasets["dev"], collate_fn,
                      os.path.join(artifact_root, "dev_predictions_best.txt"))
            logger.info("Evaluating best model on test...")
            self.eval(datasets["test"], collate_fn,
                      os.path.join(artifact_root, "test_predictions_best.txt"))
        
        logger.info("Best test F1 seen %s from epoch %s",
                    best_test_f1, best_test_f1_epoch)
