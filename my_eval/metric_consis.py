import sys
sys.path.append('.')

import argparse
import torch

from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.logging import progress_bar
from fairseq import utils

from my_eval.consistency_model.dataset import VAQCDataset
from my_eval.consistency_model.task import VQACTask
from my_eval.utils import open_file, build_indexedrawdataset_from_strlist
from my_module.dataset import GridFeaturesDataset


def handle_null_sample(dataset, dictionary):
    empty_ids = []
    for idx in range(len(dataset)):
        if dataset.sizes[idx] <= 0:
            dataset.lines[idx] = "[unk]"
            dataset.tokens_list[idx] = torch.LongTensor([dictionary.unk()])
            dataset.sizes[idx] = 1
            empty_ids.append(idx)
    return empty_ids


def _main(args):
    # load_model
    state = load_checkpoint_to_cpu(args.ckpt)
    state_args = state["args"]
    task = VQACTask.setup_task(state_args)
    model = task.build_model(state_args)
    model.load_state_dict(state["model"], strict=True, args=state_args)
    model.cuda().eval()

    # build dataset
    image_ids = open_file(args.imageid_file)
    questions = open_file(args.question_file)
    answers = open_file(args.answer_file)

    filtered_image_ids = []
    filtered_questions = []
    filtered_answers = []
    hash_sets = set()

    for idx, (i, q, a) in enumerate(zip(image_ids, questions, answers)):
        if i+q+a in hash_sets:
            continue
        hash_sets.add(i+q+a)
        filtered_image_ids.append(i)
        filtered_questions.append(q)
        filtered_answers.append(a)
    
    feature_dataset = GridFeaturesDataset(
        features_dir=args.feature_dir,
        image_ids=filtered_image_ids,
    )
    question_dataset = build_indexedrawdataset_from_strlist(filtered_questions, task.question_dict)
    answer_dataset = build_indexedrawdataset_from_strlist(filtered_answers, task.answer_dict)

    # if there are length 0 samples, fill with [unk]
    empty_questions = handle_null_sample(question_dataset, task.question_dict)
    if len(empty_questions) > 0:
        print(f"There are {len(empty_questions)} empty questions: {empty_questions}")

    empty_answers = handle_null_sample(answer_dataset, task.answer_dict)
    if len(empty_answers) > 0:
        print(f"There are {len(empty_answers)} empty answers: {empty_answers}")

    dataset = VAQCDataset(
        feature=feature_dataset,
        question=question_dataset,
        question_dict=task.question_dict,
        answer=answer_dataset,
        answer_dict=task.answer_dict,
        shuffle=False,
    )

    # build iterator
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_sentences=args.batch_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(itr)

    predictions = []
    scores = []
    for sample in progress:
        sample = utils.move_to_cuda(sample)
        logits, _ = model(**sample["net_input"])
        # pred = torch.argmax(logits, dim=-1)
        pred = (logits > 0.5).float()

        scores.append(logits.cpu().detach())
        predictions.append(pred.cpu().detach())
    
    predictions = torch.cat(predictions)
    scores = torch.cat(scores)

    acc = float(torch.mean(predictions))
    avg_score = float(torch.mean(scores))
    pos_score = float(torch.sum(scores*predictions) / torch.sum(predictions))
    neg_score = float(torch.sum(scores*(1-predictions) / torch.sum(1-predictions)))

    if args.output_file is None:
        f = sys.stdout
    else:
        f = open(args.output_file, "a+")

    print(f'\tTot: \t{predictions.size(0)}', file=f)
    print(f'\tAcc: \t{round(acc * 100, 2)}', file=f)
    print(f'\tScore: \t{round(avg_score, 4)}', file=f)
    print(f'\tPos Score: \t{round(pos_score, 4)}', file=f)
    print(f'\tNeg Score: \t{round(neg_score, 4)}', file=f)

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--question-file', type=str)
    parser.add_argument('--answer-file', type=str)
    parser.add_argument('--imageid-file', type=str)
    parser.add_argument('--feature-dir', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-file', type=str, default=None)

    args = parser.parse_args()
    _main(args)



