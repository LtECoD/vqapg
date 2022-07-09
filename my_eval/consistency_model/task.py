import os
import logging

from fairseq import utils
from fairseq.tasks import LegacyFairseqTask
from fairseq.tasks import register_task

from my_eval.consistency_model.dataset import VAQCDataset
from my_module.utils import load_vqa_datasets


logger = logging.getLogger(__name__)


@register_task("vqac_consistency")
class VQACTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
       # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')

        parser.add_argument("--question", default='question', metavar="SRC")
        parser.add_argument("--answer", default='answer', metavar="TARGET")
        parser.add_argument('--image', default='image', metavar="IMAGE")
        parser.add_argument('--feature', default='grid', choices=['grid'], metavar='FEATURE', help='feature type of image')
        parser.add_argument('--grid-shape', nargs='+', type=int, default=[10, 10])

    def __init__(self, args, question_dict, answer_dict):
        super().__init__(args)
        self.question_dict = question_dict
        self.answer_dict = answer_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # load dictionaries
        question_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.question)))
        answer_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.answer)))
        logger.info('[{}] dictionary: {} types'.format(args.question, len(question_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.answer, len(answer_dict)))

        return cls(
            args,
            question_dict=question_dict,
            answer_dict=answer_dict,
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        feature_dataset, question_dataset, answer_dataset = load_vqa_datasets(
            split, self.args, question_dict=self.question_dict, answer_dict=self.answer_dict)

        self.datasets[split] = VAQCDataset(
            feature=feature_dataset,
            question=question_dataset,
            question_dict=self.question_dict,
            answer=answer_dataset,
            answer_dict=self.answer_dict,
            shuffle=(split != 'test'),
        )
    
    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary