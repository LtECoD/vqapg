# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from fairseq.data.indexed_dataset import IndexedRawTextDataset
from fairseq import utils

from fairseq.tasks import register_task, LegacyFairseqTask

from my_module.dataset import JVAQGDataset, GridFeaturesDataset, VAQGS2SDataset, VAQGPipeDataset
from my_module.generator_jvaqg import JVAQGGenerator
from my_module.generator_vaqg_s2s import VAQGS2SGenerator
from my_module.utils import load_vqa_datasets


logger = logging.getLogger(__name__)


@register_task('jvaqg')
class JVAQGTask(LegacyFairseqTask):
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

        #! Whether do answer generation or question generation
        parser.add_argument('--do-answer', type=str, default="True")
        parser.add_argument('--do-question', type=str, default="True")

    def __init__(self, args, question_dict, answer_dict, do_answer, do_question):
        super().__init__(args)
        self.question_dict = question_dict
        self.answer_dict = answer_dict
        self.do_answer = do_answer
        self.do_question = do_question

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
            do_answer=utils.eval_bool(args.do_answer),
            do_question=utils.eval_bool(args.do_question),
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        feature_dataset, question_dataset, answer_dataset = load_vqa_datasets(
            split, self.args, question_dict=self.question_dict, answer_dict=self.answer_dict)

        self.datasets[split] = JVAQGDataset(
            feature=feature_dataset,
            question=question_dataset,
            question_dict=self.question_dict,
            answer=answer_dataset,
            answer_dict=self.answer_dict,
            shuffle=(split != 'test'),
            append_eos_to_question=True,
            append_eos_to_answer=True,
        )
    
    def build_generator(self, models, args, extra_gen_cls_kwargs=None):
        return JVAQGGenerator(
            models=models,
            do_answer=self.do_answer,
            do_question=self.do_question,
            answer_dict=self.answer_dict,
            question_dict=self.question_dict,
            beam_size=args.beam,
            temperature=args.temperature,
        )

    # def begin_epoch(self, epoch, model):
    #     """
    #     Hook function called before the start of each epoch.
    #     ! 2020.11.24 to anneal kl loss weight
    #     """
    #     model.anneal_kl_weight(epoch)


def load_vaqg_s2s_dataset(split, args, target_dict):
    prefix = os.path.join(args.data, '{}'.format(split))
    target_dataset = IndexedRawTextDataset(prefix+'.'+args.target+f".{args.qa_order}", target_dict)

    with open(prefix+'.'+args.image, 'r') as f:
        image_ids = f.readlines()
        image_ids = [image_id.strip() for image_id in image_ids]
    
    feature_dataset = GridFeaturesDataset(
        features_dir=prefix+f'-features-{args.feature}',
        image_ids=image_ids,
        grid_shape=tuple(args.grid_shape),
    )

    logger.info("loaded {} examples from: {}".format(len(target_dataset), prefix+'.'+args.target+f".{args.qa_order}"))
    logger.info("loaded {} examples from: {}".format(len(feature_dataset), prefix+'.'+args.image))

    return VAQGS2SDataset(
        feature=feature_dataset,
        target=target_dataset,
        target_dict=target_dict,
        shuffle=(split != 'test'),
        append_eos_to_target=True,
    )


@register_task('vaqg_s2s')
class VAQGS2STask(LegacyFairseqTask):
    def __init__(self, args, target_dict):
        super().__init__(args)
        self.target_dict = target_dict

    @staticmethod
    def add_args(parser):
        JVAQGTask.add_args(parser)
        parser.add_argument("--target", default='target', metavar="TGT")
        parser.add_argument("--qa-order", choices=['qa', 'aq'], metavar="STR")

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # load dictionaries
        target_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target)))
        logger.info('[{}] dictionary: {} types'.format(args.target, len(target_dict)))

        return cls(
            args,
            target_dict=target_dict,
        )

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.target_dict

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        self.datasets[split] = load_vaqg_s2s_dataset(
            split=split,
            args=self.args,
            target_dict=self.target_dict,
        )

    def build_generator(self, models, args, extra_gen_cls_kwargs=None):
        return VAQGS2SGenerator(
            models=models,
            target_dict=self.target_dict,
            beam_size=args.beam,
            temperature=args.temperature,
        )


def load_vaqg_pipe_dataset(split, args, target_dict, pre_dict):
    prefix = os.path.join(args.data, '{}'.format(split))
    target_dataset = IndexedRawTextDataset(prefix+'.'+args.target, target_dict)

    with open(prefix+'.'+args.image, 'r') as f:
        image_ids = f.readlines()
        image_ids = [image_id.strip() for image_id in image_ids]
    
    feature_dataset = GridFeaturesDataset(
        features_dir=prefix+f'-features-{args.feature}',
        image_ids=image_ids,
        grid_shape=tuple(args.grid_shape),
    )

    if args.preliminary is not None:
        if args.preliminary_path is not None:
            preliminary_dataset = IndexedRawTextDataset(args.preliminary_path, pre_dict)
        else:
            preliminary_dataset = IndexedRawTextDataset(prefix+'.'+args.preliminary, pre_dict)
    else:
        preliminary_dataset = None

    logger.info("loaded {} examples from: {}".format(len(target_dataset), prefix+'.'+args.target))
    logger.info("loaded {} examples from: {}".format(len(feature_dataset), prefix+'.'+args.image))

    return VAQGPipeDataset(
        feature=feature_dataset,
        preliminary=preliminary_dataset,
        pre_dict=pre_dict,
        target=target_dataset,
        target_dict=target_dict,
        shuffle=(split != 'test'),
        append_eos_to_target=True,
    )


@register_task('vaqg_pipe')
class VAQGPipeTask(LegacyFairseqTask):
    def __init__(self, args, target_dict, pre_dict):
        super().__init__(args)
        self.target_dict = target_dict
        self.pre_dict = pre_dict

    @staticmethod
    def add_args(parser):
        JVAQGTask.add_args(parser)
        parser.add_argument("--target", default=None, metavar="TGT")
        parser.add_argument("--preliminary", default=None, metavar="STR")
        parser.add_argument("--preliminary-path", default=None, metavar='STR')

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # load dictionaries
        target_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target)))
        if args.preliminary is not None:
            pre_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.preliminary)))
        else:
            pre_dict = None
        logger.info('[{}] dictionary: {} types'.format(args.target, len(target_dict)))

        return cls(
            args,
            target_dict=target_dict,
            pre_dict=pre_dict,
        )

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.target_dict

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = load_vaqg_pipe_dataset(
            split=split,
            args=self.args,
            target_dict=self.target_dict,
            pre_dict=self.pre_dict,
        )

    def build_generator(self, models, args, extra_gen_cls_kwargs=None):
        return VAQGS2SGenerator(
            models=models,
            target_dict=self.target_dict,
            beam_size=args.beam,
            temperature=args.temperature,
        )

