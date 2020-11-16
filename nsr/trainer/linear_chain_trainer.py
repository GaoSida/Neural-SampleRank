"""Assemble the linear chain CRF model, then train and evaluation.
"""
import os
import time
import logging
from typing import List, Callable, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from nsr.model.token_embedding import CharCNN, TokenEmbedding
from nsr.model.flair_embeddings import FlairEmbeddings
from nsr.model.sentence_encoder import BiRNN
from nsr.model.unary_factor import UnaryFactor
from nsr.flair.embedding_cache import FlairEmbeddingCache
from nsr.graph.linear_chain import NegLogLikelihoodFwdLoss
import nsr.utils.conll_dataset as conll

logger = logging.getLogger(__name__)


class LinearChainCRFModel(nn.Module):
    """Linear Chain CRF Model.
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
        self.flair_enabled = config["flair_enabled"]
        self.label_vocab_size = label_vocab_size
        # For <START> and <STOP> tag for the CRF
        self.start_tag_index = label_vocab_size
        self.stop_tag_index = label_vocab_size + 1
        
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
        
        self.emission = UnaryFactor(
            2 * config["rnn_hidden_dim"], label_vocab_size + 2,
            config["decoder_hidden_dim"], config["dropout"]
        )  # Regular label space size + <START> and <STOP>
        
        self.transitions = torch.nn.Parameter(
            torch.randn(label_vocab_size + 2, label_vocab_size + 2)
        )
        self.transitions.detach()[self.start_tag_index, :] = -10000
        self.transitions.detach()[:, self.stop_tag_index] = -10000

    def forward(self, tokens: Tensor, token_chars: Tensor, lengths: Tensor,
                strings: List[str]) -> Tensor:
        """
        Args:
            tokens: the token indices. [batch_size, max_num_tokens]
            token_chars: the character indices for each token.
                Shape [batch_size * max_num_tokens, max_num_chars]
            lengths: length of each sentence in batch. shape [batch_size, ]
            strings: str sentences, len(strings) == batch_size
        Returns:
            emission_scores: [batch_size, max_num_tokens, num_labels]
            transition_scores
        """
        if self.flair_enabled:
            embeddings = self.flair_embedding(strings)
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
        # shape: [batch_size, max_num_tokens, 2 * rnn_hidden_dim]
        return self.emission(rnn_output), self.transitions
        # shape: [batch_size, max_num_tokens, label_vocab_size + 2]


class LinearChainCRFTrainer:
    """Train the linear chain CRF with neg log likelihood loss (forward algo)
    and evaluate with Viterbi decoding.
    """
    def __init__(self, config: dict, pretrained_embedding: Tensor,
                 char_vocab_size: int, label_vocab_size: int,
                 device: torch.device):
        """See LinearChainCRFModel for the semantics of arguments.
        """
        self.config = config
        self.device = device
        self.model = LinearChainCRFModel(
            config, pretrained_embedding, char_vocab_size, label_vocab_size
        ).to(self.device)
        logger.info("Model: \n" + str(self.model))
        
        self.label_vocab_size = label_vocab_size
        
        self.criterion = NegLogLikelihoodFwdLoss(label_vocab_size, device)

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

        data = DataLoader(dataset, batch_size=config["batch_size"],
                          collate_fn=collate_fn)
        
        self.model.eval()  # Switch to evaluation mode
        prediction_dump = ""
        start = time.perf_counter()
        total_num_tokens = 0
        num_batches = 0
        for tokens, token_chars, lengths, _, strings, labels in data:
            tokens, token_chars, lengths = \
                tokens.to(device), token_chars.to(device), lengths.to(device)
            total_num_tokens += torch.sum(lengths).item()
            num_batches += 1
            
            emissions, transitions = \
                self.model(tokens, token_chars, lengths, strings)
            
            predictions = self.criterion.viterbi(emissions, transitions,
                                                 lengths)
            
            prediction_dump += dataset.write_prediction(
                tokens, labels, predictions)

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
            
            data = DataLoader(
                datasets["train"], batch_size=config["batch_size"],
                shuffle=True, collate_fn=collate_fn
            )
            
            start = time.perf_counter()
            num_batches = 0
            total_num_tokens = 0
            accumulated_loss = 0.0
            for tokens, token_chars, lengths, _, strings, labels in data:
                optimizer.zero_grad()
                
                num_batches += 1
                num_batch_tokens = torch.sum(lengths).item()
                total_num_tokens += num_batch_tokens
                
                tokens, token_chars, lengths, labels = \
                    tokens.to(device), token_chars.to(device), \
                    lengths.to(device), labels.to(device)

                emissions, transitions = \
                    self.model(tokens, token_chars, lengths, strings)
                # shape: [batch_size, max_num_tokens, label_vocab_size]
                loss = self.criterion(emissions, transitions, lengths, labels)
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
                    test_f1 = self.eval(datasets["test"], collate_fn)
                    if test_f1 > best_test_f1:
                        best_test_f1 = test_f1
                        best_test_f1_epoch = epoch
            torch.save(self.model.state_dict(), last_model_path)
            
            if config["lr_scheduler"]:
                train_loss = accumulated_loss / total_num_tokens
                scheduler.step(train_loss)
                logger.info("Train loss this epoch: %.4f, new lr %s",
                            train_loss, optimizer.param_groups[0]['lr'])
        
        if not config["train_with_dev"]:
            # Re-evaluate the best model and dump predictions
            self.model.load_state_dict(torch.load(best_model_path))
            logger.info("Evaluating best model on dev from epoch %s...",
                        best_dev_f1_epoch)
            self.eval(datasets["dev"], collate_fn,
                      os.path.join(artifact_root, "dev_predictions_best.txt"))
            logger.info("Evaluating best model on test...")
            self.eval(datasets["test"], collate_fn,
                      os.path.join(artifact_root, "test_predictions_best.txt"))
        
        logger.info("Best test F1 seen %s from epoch %s",
                    best_test_f1, best_test_f1_epoch)
