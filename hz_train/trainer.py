from transformers import Trainer
from .model import EmbeddingModel, EmbeddingModel4Qwen, EmbeddingModel4Qwen2
from typing import Optional, List
import torch
import os


class HzTrainer(Trainer):
    def compute_loss(
        self,
        model: EmbeddingModel | EmbeddingModel4Qwen | EmbeddingModel4Qwen2,
        inputs: dict[str, List[str]],
        **kwargs,
    ):
        """
        Calculate the constractive loss for the model, which is defined as:
            For each query, minimize the margin between the cosine similarity of the query and positive example and the cosine similarity of the query and negative example.
            The loss is calculated as:
                # ! view the margin as probability, then calculate the cross entropy loss(BCE loss)
                -\sum_{query} \sum_{neg} \log( sigmoid( similarity(query, pos/query) - similarity(query, neg)))
         
        Args:
            model: The model to train
            inputs: The inputs to the model, in this case, the query, positive and negative examples
            kwargs: Additional keyword arguments
            
        """
        query = inputs["query"]
        pos = inputs["pos"]
        neg = inputs["neg"]

        text_embeddings = model(query, max_len=self.args.query_max_len)

        if self.args.embedding_model_name == "qwen2":#isinstance(model, EmbeddingModel4Qwen2) or
            text_pos_embeddings = model(
                pos, max_len=self.args.passage_max_len, is_query=False
            )
            text_neg_embeddings = model(
                neg, max_len=self.args.passage_max_len, is_query=False
            )
        else:
            text_pos_embeddings = model(
                pos,
                max_len=self.args.passage_max_len,
            )
            text_neg_embeddings = model(
                neg,
                max_len=self.args.passage_max_len,
            )

        # [batch_size, embedding_dim] -> [batch_size]
        sim_pos_vector = torch.cosine_similarity(
            text_embeddings, text_pos_embeddings, dim=-1
        )
        sim_pos_vector = sim_pos_vector / self.args.temperature
        
        # [batch_size, embedding_dim] -> [batch_size, batch_size]
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_neg_matrix = sim_neg_matrix / self.args.temperature
        sim_diff_matrix = sim_pos_vector.unsqueeze(1) - sim_neg_matrix
        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).mean()
        return loss

    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)
