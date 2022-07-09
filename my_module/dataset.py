# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset, BaseWrapperDataset


logger = logging.getLogger(__name__)


class GridFeaturesDataset(FairseqDataset):
    def __init__(self, features_dir, image_ids, grid_shape=(10, 10)):
        self.features_dir = features_dir
        self.image_ids = image_ids
        # self.image_features = {}
        self.grid_shape = grid_shape

        self.sizes = np.ones((len(image_ids)), dtype=np.int) * np.prod(self.grid_shape)
        
    def read_data(self, image_id):
        # if image_id not in self.image_features:
        features_file = os.path.join(self.features_dir, f'{image_id}.npy')
        features = torch.as_tensor(np.load(features_file))
        return features
            # self.image_features[image_id] = torch.as_tensor(features)
        # return self.image_features[image_id]


    def size(self, index):
        return self.sizes(index)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        return self.read_data(self.image_ids[index])


class JVAQGDataset(FairseqDataset):
    def __init__(
        self,
        feature,
        question, question_dict,
        answer, answer_dict,
        shuffle=True,
        append_eos_to_question=True,
        append_eos_to_answer=True,
    ):

        self.feature = feature

        self.question = question
        self.question_sizes = question.sizes
        self.question_dict = question_dict

        self.answer = answer
        self.answer_sizes = answer.sizes
        self.answer_dict = answer_dict

        self.shuffle = shuffle
        self.append_eos_to_question = append_eos_to_question
        self.append_eos_to_answer = append_eos_to_answer


    def __getitem__(self, index):
        image_item = self.feature.image_ids[index]
        feature_item = self.feature[index]

        question_item = self.question[index]
        answer_item = self.answer[index]
     
        if self.append_eos_to_question:
            eos = self.question_dict.eos()
            if self.question[index][-1] != eos:
                question_item = torch.cat([self.question[index], torch.LongTensor([eos])])

        if self.append_eos_to_answer:
            eos = self.answer_dict.eos()
            if self.answer[index][-1] != eos:
                answer_item = torch.cat([self.answer[index], torch.LongTensor([eos])])

        example = {
            'id': index,
            'image_id': image_item,
            'feature': feature_item,
            'question': question_item,
            'answer': answer_item,
        }
        return example

    def __len__(self):
        return len(self.feature)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])
        image_id = [s['image_id'] for s in samples]        # list of strs

        features = [s['feature'] for s in samples]
        features = torch.stack(features, dim=0)                   # bsz x 100 x 2048

        questions = data_utils.collate_tokens(
            values=[s['question'] for s in samples],
            pad_idx=self.question_dict.pad(),
            eos_idx=self.question_dict.eos(),
            move_eos_to_beginning=False
        )
        answers = data_utils.collate_tokens(
            values=[s['answer'] for s in samples],
            pad_idx=self.answer_dict.pad(),
            eos_idx=self.answer_dict.eos(),
            move_eos_to_beginning=False
        )

        question_lengths = torch.LongTensor([
            s['question'].ne(self.question_dict.pad()).long().sum() for s in samples])
        answer_lengths = torch.LongTensor([
            s['answer'].ne(self.answer_dict.pad()).long().sum() for s in samples])

        question_tokens = question_lengths.sum().item()
        answer_tokens = answer_lengths.sum().item()
        ntokens = question_tokens + answer_tokens

        prev_question_tokens = data_utils.collate_tokens(
            values=[s['question'] for s in samples],
            pad_idx=self.question_dict.pad(),
            eos_idx=self.question_dict.eos(),
            move_eos_to_beginning=True,
        )
        prev_answer_tokens = data_utils.collate_tokens(
            values=[s['answer'] for s in samples],
            pad_idx=self.answer_dict.pad(),
            eos_idx=self.answer_dict.eos(),
            move_eos_to_beginning=True,
        )

        batch = {
            'id': id,
            'image_id': image_id,
            'ntokens': ntokens,
            'question_tokens': question_tokens,
            'answer_tokens': answer_tokens,
            'batch_size': len(samples),
            'features': features,
            'questions': questions,
            'answers': answers,
            'question_lengths': question_lengths,
            'answer_lengths': answer_lengths,
            'prev_question_tokens': prev_question_tokens,
            'prev_answer_tokens': prev_answer_tokens,
        }

        return batch

    def size(self, index):
        return self.feature.sizes[index], max(self.question.sizes[index], self.answer.sizes[index])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.question_sizes[index] + self.answer_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        # sort by answer length, then question length
        indices = indices[
            np.argsort(self.answer_sizes[indices], kind='mergesort')
        ]
        return indices[np.argsort(self.question_sizes[indices], kind='mergesort')]


class VAQGS2SDataset(FairseqDataset):
    def __init__(
        self,
        feature,
        target,
        target_dict,
        shuffle=True,
        append_eos_to_target=True,
    ):
        self.feature = feature

        self.target = target
        self.target_sizes = target.sizes
        self.target_dict = target_dict

        self.shuffle = shuffle
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        image_item = self.feature.image_ids[index]
        feature_item = self.feature[index]
        target_item = self.target[index]
     
        if self.append_eos_to_target:
            eos = self.target_dict.eos()
            if self.target[index][-1] != eos:
                target_item = torch.cat([self.target[index], torch.LongTensor([eos])])

        example = {
            'id': index,
            'image_id': image_item,
            'feature': feature_item,
            'target': target_item,
        }
        return example

    def __len__(self):
        return len(self.feature)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])
        image_id = [s['image_id'] for s in samples]        # list of strs

        features = [s['feature'] for s in samples]
        features = torch.stack(features, dim=0)                   # bsz x 100 x 2048

        targets = data_utils.collate_tokens(
            values=[s['target'] for s in samples],
            pad_idx=self.target_dict.pad(),
            eos_idx=self.target_dict.eos(),
            move_eos_to_beginning=False
        )
    
        target_lengths = torch.LongTensor([
            s['target'].ne(self.target_dict.pad()).long().sum() for s in samples])
        ntokens = target_lengths.sum().item()

        prev_tokens = data_utils.collate_tokens(
            values=[s['target'] for s in samples],
            pad_idx=self.target_dict.pad(),
            eos_idx=self.target_dict.eos(),
            move_eos_to_beginning=True,
        )

        batch = {
            'id': id,
            'image_id': image_id,
            'ntokens': ntokens,
            'batch_size': len(samples),
            'features': features,
            'prev_tokens': prev_tokens,
            'targets': targets,
            'target_lengths': target_lengths,
        }

        return batch

    def size(self, index):
        return self.feature.sizes[index], self.target.sizes[index]

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.target_sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        indices = indices[
            np.argsort(self.target_sizes[indices], kind='mergesort')
        ]
        return indices


class VAQGPipeDataset(FairseqDataset):
    def __init__(
        self,
        feature,
        preliminary,
        target,
        target_dict,
        pre_dict,
        shuffle=True,
        append_eos_to_target=True,
    ):
        self.feature = feature
        self.preliminary = preliminary
        self.pre_sizes = preliminary.sizes if preliminary is not None else None
        self.pre_dict = pre_dict

        self.target = target
        self.target_sizes = target.sizes
        self.target_dict = target_dict

        self.shuffle = shuffle
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        image_item = self.feature.image_ids[index]
        feature_item = self.feature[index]
        if self.preliminary is not None:
            pre_item = self.preliminary[index]
        else:
            pre_item = None
        target_item = self.target[index]
        
        if self.append_eos_to_target:
            eos = self.target_dict.eos()
            if self.target[index][-1] != eos:
                target_item = torch.cat([self.target[index], torch.LongTensor([eos])])

        example = {
            'id': index,
            'image_id': image_item,
            'feature': feature_item,
            'preliminary': pre_item,
            'target': target_item,
        }
        return example

    def __len__(self):
        return len(self.feature)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])
        image_id = [s['image_id'] for s in samples]        # list of strs

        features = [s['feature'] for s in samples]
        features = torch.stack(features, dim=0)                   # bsz x 100 x 2048

        if samples[0]['preliminary'] is not None:
            preliminaries = data_utils.collate_tokens(
                values=[s['preliminary'] for s in samples],
                pad_idx=self.pre_dict.pad(),
                eos_idx=self.pre_dict.eos(),
                move_eos_to_beginning=False
            )
            preliminary_lengths = torch.LongTensor([
                s['preliminary'].ne(self.pre_dict.pad()).long().sum() for s in samples])
        else:
            preliminaries = preliminary_lengths = None

        targets = data_utils.collate_tokens(
            values=[s['target'] for s in samples],
            pad_idx=self.target_dict.pad(),
            eos_idx=self.target_dict.eos(),
            move_eos_to_beginning=False
        )
    
        target_lengths = torch.LongTensor([
            s['target'].ne(self.target_dict.pad()).long().sum() for s in samples])
        ntokens = target_lengths.sum().item()

        prev_tokens = data_utils.collate_tokens(
            values=[s['target'] for s in samples],
            pad_idx=self.target_dict.pad(),
            eos_idx=self.target_dict.eos(),
            move_eos_to_beginning=True,
        )

        batch = {
            'id': id,
            'image_id': image_id,
            'ntokens': ntokens,
            'batch_size': len(samples),
            'features': features,
            'preliminaries': preliminaries,
            'preliminary_lengths': preliminary_lengths,
            'prev_tokens': prev_tokens,
            'targets': targets,
            'target_lengths': target_lengths,
        }

        return batch

    def size(self, index):
        if self.preliminary is None:
            return self.feature.sizes[index], self.target.sizes[index]
        else:
            return self.feature.sizes[index], self.preliminary.sizes[index], self.target_sizes[index]

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.target_sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        indices = indices[
            np.argsort(self.target_sizes[indices], kind='mergesort')
        ]
        return indices