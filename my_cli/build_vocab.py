# modified version of fairseq_cli/preprocess.py
"""
Data pre-processing: build vocabularies.
"""

import os
import argparse

from fairseq.tasks import FairseqTask


def main(args):
    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames):
        return FairseqTask.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.threshold,
            nwords=args.nwords,
            padding_factor=args.padding_factor,
        )

    target_dict = build_dictionary([train_path(args.target+".qa")])
    if args.joined_dictionary:
        quesition_dict = answer_dict = target_dict
    else:
        quesition_dict = build_dictionary([train_path(args.question)])
        answer_dict = build_dictionary([train_path(args.answer)])

    target_dict.save(dict_path(args.target))
    quesition_dict.save(dict_path(args.question))
    answer_dict.save(dict_path(args.answer))


def cli_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--question", default=None, metavar="QUESTION")
    parser.add_argument("--answer", default=None, metavar="ANSWER")

    parser.add_argument("--target", default=None, metavar="TAEGET", help="joined dict")
    parser.add_argument("--sep-token", default=None, metavar='STR')

    parser.add_argument("--trainpref", metavar="FP", default=None,
                       help="train file prefix")

    parser.add_argument("--destdir", metavar="DIR", default="",
                       help="destination dir")

    parser.add_argument("--threshold", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    parser.add_argument("--nwords", metavar="N", default=-1, type=int,
                       help="number of target words to retain")

    parser.add_argument("--joined-dictionary", action="store_true",
                       help="Generate joined dictionary")
    parser.add_argument("--padding-factor", metavar="N", default=0, type=int,
                       help="Pad dictionary size to be multiple of N")
    parser.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
