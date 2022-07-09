import sys
sys.path.append('.')

import argparse
from nlgeval import NLGEval
from collections import Counter

from my_eval.utils import open_file


evaluator = NLGEval(no_skipthoughts=True, no_glove=True)


def evaluate(golds, preds, image_ids):
    refs = {}
    hypos = {}

    for idx, (g, p, _id) in enumerate(zip(golds, preds, image_ids)):
        if _id not in refs:
            assert _id not in hypos
            refs[_id] = set([g])
            hypos[_id] = set([p])
        else:
            assert _id in hypos
            refs[_id].add(g)
            hypos[_id].add(p)
    
    max_ref_num = max(len(refs[_id]) for _id in refs)

    references = [[] for i in range(max_ref_num)]
    hypothesis = []

    for _id in refs:
        for hyp in hypos[_id]:
            hypothesis.append(hyp)
            for idx, ref in enumerate(refs[_id]):
                references[idx].append(ref)
            for idx in range(len(refs[_id]), max_ref_num):
                references[idx].append("")

        # for idx, ref in enumerate(refs[_id]):
        #     references[idx].append(ref)
        # for idx in range(len(refs[_id]), max_ref_num):
        #     references[idx].append("")
        # hypothesis.append(hypos[_id])
    metrics_dict = evaluator.compute_metrics(references, hypothesis)
    return metrics_dict

def _distinct(seqs):
    """compute group distinct"""
    unigrams_all, bigrams_all, trigrams_all, quadgrams_all = Counter(), Counter(), Counter(), Counter()

    # to remove last punctuations
    # seqs = [s.strip()[:-1].strip() for s in seqs]

    for seq in seqs:
        seq = seq.split()
        unigrams_all.update(Counter(seq))
        bigrams_all.update(Counter(zip(seq, seq[1:])))
        trigrams_all.update(Counter(zip(seq, seq[1:], seq[2:])))
        quadgrams_all.update(Counter(zip(seq, seq[1:], seq[2:], seq[3:])))

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-12)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-12)
    inter_dist3 = (len(trigrams_all)+1e-12) / (sum(trigrams_all.values())+1e-12)
    inter_dist4 = (len(quadgrams_all)+1e-12) / (sum(quadgrams_all.values())+1e-12)
    return inter_dist1, inter_dist2, inter_dist3, inter_dist4


def distinct(seqs, _ids):
    """ Calculate intra/inter distinct 1/2. """
    
    id_seq_dict = {}
    for _id, seq in zip(_ids, seqs):
        if _id in id_seq_dict:
            id_seq_dict[_id].append(seq)
        else:
            id_seq_dict[_id] = [seq]

    all_seqs = [s for v in id_seq_dict.values() for s in v]
    inter_dist1, inter_dist2, inter_dist3, inter_dist4 = _distinct(all_seqs)

    intra_dist1s = []
    intra_dist2s = []
    intra_dist3s = []
    intra_dist4s = []
    for _id in id_seq_dict:
        intra_dist1, intra_dist2, intra_dist3, intra_dist4 = _distinct(id_seq_dict[_id])
        intra_dist1s.append(intra_dist1)
        intra_dist2s.append(intra_dist2)
        intra_dist3s.append(intra_dist3)
        intra_dist4s.append(intra_dist4)

    avg_intra_dist1 = sum(intra_dist1s) / len(intra_dist1s)
    avg_intra_dist2 = sum(intra_dist2s) / len(intra_dist2s)
    avg_intra_dist3 = sum(intra_dist3s) / len(intra_dist3s)
    avg_intra_dist4 = sum(intra_dist4s) / len(intra_dist4s)

    metric = {
        "inter_dist1": inter_dist1,
        "inter_dist2": inter_dist2,
        "inter_dist3": inter_dist3,
        "inter_dist4": inter_dist4,
        "intra_dist1": avg_intra_dist1,
        "intra_dist2": avg_intra_dist2,
        "intra_dist3": avg_intra_dist3,
        "intra_dist4": avg_intra_dist4,
    }
    return metric

def _main(args):

    gold_answers = open_file(args.gold_answer)
    gold_questions = open_file(args.gold_question)
    image_ids = open_file(args.image_id)

    pred_answers = open_file(args.pred_answer)
    pred_questions = open_file(args.pred_question)

    if args.output_file is not None:
        fp = open(args.output_file, "a+", buffering=1, encoding="utf-8")
    else:
        fp = sys.stdout

    distinct_qas = set([q + " " + a for q, a in zip(pred_questions, pred_answers)])
    distinct_image_qa = set([_id + " " + q + " " + a \
         for _id, q, a in zip(image_ids, pred_questions, pred_answers)])
    distinct_groups = set(image_ids)

    print(f'Test dataset size:\t {len(image_ids)}', file=fp)
    print(f'Distinct images:\t {len(distinct_groups)}', file=fp)
    print(f'Distinct qa pairs:\t {len(distinct_qas)}', file=fp)
    print(f'Distinct iqa pairs:\t {len(distinct_image_qa)}', file=fp)

    # if pred_answers is not None:
    #     answer_overlap = evaluate(gold_answers, pred_answers, image_ids)
    #     # answer_diversity = distinct(pred_answers, image_ids)
    #     print('Answer N-grams Metrics:', file=fp)
    #     for k, v in answer_overlap.items():
    #         print(f'\t{k}:\t{v}', file=fp)
    #     # for k, v in answer_diversity.items():
    #     #     print(f'\t{k}:\t{round(v, 4)}', file=fp)
        
    # if pred_questions is not None:
    #     question_overlap = evaluate(gold_questions, pred_questions, image_ids)
    #     # question_diversity = distinct(pred_questions, image_ids)
    #     print('Question N-grams Metrics:', file=fp)
    #     for k, v in question_overlap.items():
    #         print(f'\t{k}:\t{v}', file=fp)
    #     # for k, v in question_diversity.items():
    #     #     print(f'\t{k}:\t{round(v, 4)}', file=fp)

    preds = [q + " " + a for q, a in zip(pred_questions, pred_answers)]

    golds = [q + " " + a for q, a in zip(gold_questions, gold_answers)]
    metrics = evaluate(golds, preds, image_ids)
    print('N-grams Metrics:', file=fp)
    for k, v in metrics.items():
        print(f'\t{k}:\t{v}', file=fp)

    diversity = distinct(preds, image_ids)

    print('Diversity Metrics:', file=fp)
    for k, v in diversity.items():
        print(f'\t{k}:\t{round(v, 4)}', file=fp)
    fp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-answer", type=str)
    parser.add_argument("--gold-question", type=str)
    parser.add_argument("--image-id", type=str)
    parser.add_argument("--pred-answer", type=str)
    parser.add_argument("--pred-question", type=str)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    _main(args)