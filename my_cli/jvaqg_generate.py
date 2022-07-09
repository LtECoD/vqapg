#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter


def main(args):
    assert args.path is not None, "--path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(
            args.results_path, "generate-{}.txt".format(args.gen_subset)
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    question_dict = task.question_dict
    answer_dict = task.answer_dict

    overrides = ast.literal_eval(args.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_weight": args.lm_weight}
    generator = task.build_generator(
        models, args, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        gen_timer.start()
        answer_hypos, question_hypos = task.inference_step(
            generator,
            models,
            sample,
        )
        num_generated_tokens = \
            sum(len(h[0]["tokens"]) for h in answer_hypos) if answer_hypos is not None else 0 + \
            sum(len(h[0]['tokens']) for h in question_hypos) if question_hypos is not None else 0
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):

            question_tokens = (
                utils.strip_pad(sample["questions"][i, :], question_dict.pad()).int().cpu()
            )

            answer_tokens = (
                utils.strip_pad(sample["answers"][i, :], answer_dict.pad()).int().cpu()
            )

            question_str = question_dict.string(
                question_tokens,
                args.remove_bpe,
                escape_unk=True,
            )
            
            answer_str = answer_dict.string(
                answer_tokens,
                args.remove_bpe,
                escape_unk=True,
            )

            image_id = sample["image_id"][i]
            question_str = decode_fn(question_str)
            answer_str = decode_fn(answer_str)

            print("I-{}\t{}".format(sample_id, image_id), file=output_file)
            print("X-{}\t{}".format(sample_id, answer_str), file=output_file)
            print("Y-{}\t{}".format(sample_id, question_str), file=output_file)

            # Process top answers
            if answer_hypos is not None:
                for j, hypo in enumerate(answer_hypos[i][: args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=None,
                        alignment=None,
                        align_dict=None,
                        tgt_dict=answer_dict,
                    )
                    detok_hypo_str = decode_fn(hypo_str)
                    if not args.quiet:
                        score = hypo["answer_score"] / math.log(2)  # convert to base 2
                        # original hypothesis (after tokenization and BPE)
                        print(
                            "A-{}\t{}\t{}".format(sample_id, score, hypo_str),
                            file=output_file,
                        )
                        print(
                            "S-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    map(
                                        lambda x: "{:.4f}".format(x),
                                        # convert from base e to base 2
                                        hypo["answer_positional_scores"]
                                        .div_(math.log(2))
                                        .tolist(),
                                    )
                                ),
                            ),
                            file=output_file,
                        )
            
            # Process top questions
            if question_hypos is not None:
                for j, hypo in enumerate(question_hypos[i][: args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=None,
                        alignment=None,
                        align_dict=None,
                        tgt_dict=question_dict,
                    )
                    detok_hypo_str = decode_fn(hypo_str)
                    if not args.quiet:
                        score = hypo["question_score"] / math.log(2)  # convert to base 2
                        # original hypothesis (after tokenization and BPE)
                        print(
                            "Q-{}\t{}\t{}".format(sample_id, score, hypo_str),
                            file=output_file,
                        )
                        print(
                            "S-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    map(
                                        lambda x: "{:.4f}".format(x),
                                        # convert from base e to base 2
                                        hypo["question_positional_scores"]
                                        .div_(math.log(2))
                                        .tolist(),
                                    )
                                ),
                            ),
                            file=output_file,
                        )
            print("", file=output_file)
        
        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
