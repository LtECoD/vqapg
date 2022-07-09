import torch
import numpy as np

from fairseq.data import data_utils

from my_module.dataset import JVAQGDataset


class VAQCDataset(JVAQGDataset):
    def __init__(
        self,
        feature,
        question, question_dict,
        answer, answer_dict,
        shuffle,
    ):
        super().__init__(
            feature,
            question, question_dict,
            answer, answer_dict,
            shuffle,
            False, False
        )
        
    def collater(self, samples):
        if len(samples) == 0:
            return {}

        image_id = [s['image_id'] for s in samples]        # list of strs

        features = [s['feature'] for s in samples]
        features = torch.stack(features, dim=0)            # bsz x 100 x 2048

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

        batch = {
            'image_id': image_id,
            'net_input': {
                'features': features,
                'questions': questions,
                'answers': answers,
                'question_lengths': question_lengths,
                'answer_lengths': answer_lengths,
            },
            'ntokens': ntokens,
        }

        return batch
    
    def size(self, index):
        return max(
            self.feature.sizes[index],
            self.question.sizes[index],
            self.answer.sizes[index],
        )
    
    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices