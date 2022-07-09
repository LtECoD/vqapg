import os
import torch
import logging

from my_module.dataset import GridFeaturesDataset

from fairseq.data.indexed_dataset import IndexedRawTextDataset


logger = logging.getLogger(__name__)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .ge(lengths.unsqueeze(1)))


from fairseq.models.lstm import Embedding
from fairseq import utils


def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
    embed_dict = utils.parse_embedding(embed_path)
    utils.print_embed_overlap(embed_dict, dictionary)
    return utils.load_embedding(embed_dict, dictionary, embed_tokens)


def reparameterize(mus, log_vars):
    std = torch.exp(0.5*log_vars)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mus)


def load_vqa_datasets(
    split,
    args,
    question_dict,
    answer_dict
):
    """load feature, question, image three datasets"""
    prefix = os.path.join(args.data, '{}'.format(split))
    question_dataset = IndexedRawTextDataset(
        prefix+'.'+args.question, question_dict, append_eos=False)
    answer_dataset = IndexedRawTextDataset(
        prefix+'.'+args.answer, answer_dict, append_eos=False)

    with open(prefix+'.'+args.image, 'r') as f:
        image_ids = f.readlines()
        image_ids = [image_id.strip() for image_id in image_ids]
    
    feature_dataset = GridFeaturesDataset(
        features_dir=prefix+f'-features-{args.feature}',
        image_ids=image_ids,
        grid_shape=tuple(args.grid_shape),
    )

    logger.info("loaded {} examples from: {}".format(len(question_dataset), prefix+'.'+args.question))
    logger.info("loaded {} examples from: {}".format(len(answer_dataset), prefix+'.'+args.answer))
    logger.info("loaded {} examples from: {}".format(len(feature_dataset), prefix+'.'+args.image))

    return feature_dataset, question_dataset, answer_dataset