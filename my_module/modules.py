import torch.nn as nn
from fairseq.models.lstm import Embedding

from my_module.utils import load_pretrained_embedding_from_file

class FeatureProjection(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, embed_dim)
        self.embedding_dim = embed_dim
        self.padding_idx = -1

    def forward(self, x):
        return self.linear(x)


class SpatialEncoding(nn.Module):
    """
    Encodes bounding box coordinates and relative sizes
    as vector of dimensionality `args.encoder_embed_dim`.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(5, embed_dim)

    def forward(self, x):
        return self.linear(x)


def build_answer_question_embed(
    share_token_embeddings,
    question_dict,
    answer_dict,
    embed_dim,
    token_embed_path=None,
):
    """
    Build pretrained embeddings from file. Used in decoder and my_eval
    """
    if share_token_embeddings:
        assert question_dict == answer_dict

        if token_embed_path is not None:
            token_embed = load_pretrained_embedding_from_file(
                embed_path=token_embed_path,
                dictionary=question_dict,
                embed_dim=embed_dim,
            )
        else:
            token_embed = None
        question_embed = answer_embed = token_embed
    else:
        if token_embed_path is not None:
            question_embed = load_pretrained_embedding_from_file(
                embed_path=token_embed_path,
                dictionary=question_dict,
                embed_dim=embed_dim,
            )
            answer_embed = load_pretrained_embedding_from_file(
                embed_path=token_embed_path,
                dictionary=answer_dict,
                embed_dim=embed_dim,
            )
        else:
            question_embed = answer_embed = None
    
    if question_embed is None:
        question_embed = Embedding(
            num_embeddings=len(question_dict),
            embedding_dim=embed_dim,
            padding_idx=question_dict.pad(),
        )

    if answer_embed is None:
        if share_token_embeddings:
            answer_embed = question_embed
        else:
            answer_embed = Embedding(
                num_embeddings=len(answer_dict),
                embedding_dim=embed_dim,
                padding_idx=answer_dict.pad(),
            )

    return answer_embed, question_embed


def build_target_embed(
    target_dict,
    embed_dim,
    token_embed_path=None,
):
    if token_embed_path is not None:
        target_embed = load_pretrained_embedding_from_file(
            embed_path=token_embed_path,
            dictionary=target_dict,
            embed_dim=embed_dim,
        )
    else:
        target_embed = None
    
    if target_embed is None:
     
        target_embed = Embedding(
            num_embeddings=len(target_dict),
            embedding_dim=embed_dim,
            padding_idx=target_dict.pad(),
        )

    return target_embed