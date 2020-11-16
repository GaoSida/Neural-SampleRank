"""Trainer for unary decoding model for sequence tagging.
"""
import os
import time
import logging
from typing import Dict, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from nsr.model.token_embedding import CharCNN, TokenEmbedding
from nsr.model.sentence_encoder import BiRNN
from nsr.model.unary_factor import UnaryFactor
import nsr.utils.conll_dataset as conll

logger = logging.getLogger(__name__)


class UnaryDecoderModel(nn.Module):
    """A sequence labeling model with unary (i.e. token-wise) decoding.
    """
    def __init__(self, config: dict, pretrained_embedding: Tensor,
                 char_vocab_size: int, label_vocab_size: int):
        """Assemble the model according to the config.
        Args:
            config: model configs and hyperparameters.
            pretrained_embedding: token embedding.
            char_vocab_size: size of the character vocabulary.
            label_vocab_size: size of the label space, i.e. output dim.
        """
        super().__init__()
        char_cnn = CharCNN(
            char_vocab_size, config["char_embed_dim"],
            config["char_kernel_size"], config["char_num_kernels"],
            config["dropout"]
        )
        self.token_embedding = TokenEmbedding(pretrained_embedding, char_cnn)
        self.bi_rnn = BiRNN(
            config["glove_dim"] + config["char_num_kernels"],
            config["rnn_hidden_dim"], config["rnn_num_layers"],
            config["dropout"], config["rnn_type"]
        )
        self.decoder = UnaryFactor(
            2 * config["rnn_hidden_dim"], label_vocab_size,
            config["decoder_hidden_dim"], config["dropout"]
        )

    def forward(self, tokens: Tensor, token_chars: Tensor, lengths: Tensor):
        """
        Args:
            tokens: the token indices. [batch_size, max_num_tokens]
            token_chars: the character indices for each token.
                Shape [batch_size * max_num_tokens, max_num_chars]
            lengths: length of each sentence in batch. shape [batch_size, ] 
        Returns:
            LL predictions: [batch_size, max_num_tokens, num_labels]
        """
        embeddings = self.token_embedding(tokens, token_chars)
        # shape: [batch_size, max_num_tokens, token_embedding_dim]
        rnn_output = self.bi_rnn(embeddings, lengths)
        # shape: [batch_size, max_num_tokens, 2 * rnn_hidden_dim]
        logits_prediction = self.decoder(rnn_output)
        # shape: [batch_size, max_num_tokens, label_vocab_size]
        return logits_prediction
    

class UnaryTrainer:
    """Train and evaluate a unary model.
    """
    def __init__(self, config: dict, pretrained_embedding: Tensor,
                 char_vocab_size: int, label_vocab_size: int,
                 device: torch.device):
        """See UnaryModel for the semantics of arguments.
        """
        self.config = config
        self.device = device
        self.model = UnaryDecoderModel(
            config, pretrained_embedding, char_vocab_size, label_vocab_size
        ).to(self.device)
        self.label_vocab_size = label_vocab_size
        logger.info("Model: \n" + str(self.model))

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
        for tokens, token_chars, lengths, labels in data:
            tokens, token_chars, lengths, labels = \
                    tokens.to(device), token_chars.to(device), \
                    lengths.to(device), labels.to(device)
            total_num_tokens += torch.sum(lengths).item()
            
            predictions = self.model(tokens, token_chars, lengths)
            # shape: [batch_size, max_num_tokens, label_vocab_size]
            _, predictions = torch.max(predictions, dim=2)
            # shape: [batch_size, max_num_tokens]
            
            # The way to compute F1 is CoNLL-specific as of now
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

        # Ignore the <pad> label, i.e. index 1
        criterion = nn.CrossEntropyLoss(ignore_index=1).to(device)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=config["learning_rate"])
        
        best_dev_f1 = 0
        best_model_path = os.path.join(artifact_root, "best_model.pth")
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
            for tokens, token_chars, lengths, labels in data:
                optimizer.zero_grad()
                
                num_batches += 1
                num_batch_tokens = torch.sum(lengths).item()
                total_num_tokens += num_batch_tokens
                
                tokens, token_chars, lengths, labels = \
                    tokens.to(device), token_chars.to(device), \
                    lengths.to(device), labels.to(device)

                predictions = self.model(tokens, token_chars, lengths)
                # shape: [batch_size, max_num_tokens, label_vocab_size]
                loss = criterion(predictions.view(-1, self.label_vocab_size),
                                 labels.view(-1))
                accumulated_loss += loss.item() * num_batch_tokens
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         config["gradient_clip"])
                optimizer.step()
                
                if num_batches % config["print_freq"] == 0:
                    logger.info(
                        "Progress: %s, Loss: %.3f, speed: %.2f tokens per sec",
                        "{}/{}".format(num_batches, len(data)),
                        accumulated_loss / total_num_tokens,
                        total_num_tokens / (time.perf_counter() - start)
                    )

            logger.info("Evaluating on dev set...")
            dev_f1 = self.eval(datasets["dev"], collate_fn)
            if dev_f1 > best_dev_f1:
                logger.info("Saving model - best so far...")
                best_dev_f1 = dev_f1
                torch.save(self.model.state_dict(), best_model_path)
        
        # Re-evaluate the best model and dump predictions
        self.model.load_state_dict(torch.load(best_model_path))
        logger.info("Evaluating best model on dev...")
        self.eval(datasets["dev"], collate_fn,
                  os.path.join(artifact_root, "dev_predictions_best.txt"))
        logger.info("Evaluating best model on test...")
        self.eval(datasets["test"], collate_fn,
                  os.path.join(artifact_root, "test_predictions_best.txt"))
