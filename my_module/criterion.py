from logging import log
import math
import torch

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

from my_module.utils import sequence_mask


def get_avg_loss(meters):
    loss = 0.
    assert not (('rl' in meters) and ('al' in meters or 'ql' in meters))
    for name in ['rl', 'al', 'ql', 'kl', 'bl']:
        if name in meters:
            loss = loss + meters[name].avg
    return round(float(loss), 2)


def compute_crossentropy(
    logits,
    target,
    padding_idx,
    num_tokens,
):
    lprobs = utils.log_softmax(logits, dim=-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    preds = torch.argmax(lprobs, dim=-1)

    target = target.view(-1)
    assert preds.size() == target.size()

    valid_target = target != padding_idx
    assert torch.sum(valid_target) == num_tokens

    right_tokens = torch.sum(
        torch.where(
            valid_target,
            (preds == target).long(),
            0
        )
    )

    loss = F.nll_loss(lprobs, target, ignore_index=padding_idx, reduction="sum")
    return loss, float(right_tokens)


def compute_kl(model_output, free_bits_p):
    prior_mus = model_output['distributions']['prior_mus']
    prior_log_vars = model_output['distributions']['prior_log_vars']
    posterior_mus = model_output['distributions']['posterior_mus']
    posterior_log_vars = model_output['distributions']['posterior_log_vars']

    dimension_kl_loss = 0.5 * (
        torch.div(
            torch.exp(posterior_log_vars) + torch.pow((prior_mus - posterior_mus), 2),
            torch.exp(prior_log_vars)
        ) - 1 + prior_log_vars - posterior_log_vars
    )
    kl_loss = torch.sum(dimension_kl_loss, dim=-1)

    if free_bits_p > 0.:
        kl_loss = torch.clamp(kl_loss, min=free_bits_p)

    kl_loss = torch.sum(kl_loss)
    return kl_loss


def forward_latent_model(self, model_output, sample, free_bits_p):
    reconstruction_loss, sample_size, logging_output = self.reconstruction_loss(sample, model_output)

    if 'distributions' not in model_output:
        return reconstruction_loss, sample_size, logging_output

    kl_loss = compute_kl(model_output, free_bits_p)
    mean_kl_loss = torch.div(kl_loss, sample['batch_size'])
    loss = reconstruction_loss + mean_kl_loss
    
    logging_output['kl_loss'] = kl_loss
    return loss, sample_size, logging_output


@register_criterion('jvaqg_base')
class JVAQGBaseCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(None)
        self.task = task
        self.question_padding_idx = task.question_dict.pad()
        self.answer_padding_idx = task.answer_dict.pad()

        self.do_answer = task.do_answer
        self.do_question = task.do_question

    def forward(self, model, sample, reduce=True):
        model_output = model(sample)
        return self.reconstruction_loss(sample, model_output)

    def reconstruction_loss(self, sample, model_output):
        nsentences = sample['batch_size']
        ntokens = sample['ntokens']

        if self.do_answer:
            answer_logits = model_output['answer_logits']
            answer_loss, right_answers = self.compute_answer_loss(answer_logits, sample)
            mean_answer_loss = torch.div(answer_loss, sample['answer_tokens'])
        else:
            answer_loss = right_answers = -1.
            mean_answer_loss = 0.

        if self.do_question:
            question_logits = model_output['question_logits']
            question_loss, right_tokens = self.compute_question_loss(question_logits, sample)
            mean_question_loss = torch.div(question_loss, sample['question_tokens'])
        else:
            question_loss = right_tokens = -1.
            mean_question_loss = 0.

        sample_size = 1
        loss = 0.5 * mean_answer_loss + 0.5 * mean_question_loss
        # loss = (question_loss + answer_loss) / ntokens

        logging_output = {
            'do_answer': self.do_answer,
            'do_question': self.do_question,
            'nsentences': nsentences,
            'ntokens': sample['ntokens'],
            'answer_tokens': sample['answer_tokens'],
            'question_tokens': sample['question_tokens'],
            'answer_loss': 0.5 * answer_loss,
            'right_answers': right_answers,
            'question_loss' : 0.5 * question_loss,
            'right_tokens': right_tokens,
        }
        return loss, sample_size, logging_output

    def compute_answer_loss(self, answer_logits, sample):
        return compute_crossentropy(
            logits=answer_logits,
            target=sample['answers'],
            padding_idx=self.answer_padding_idx,
            num_tokens=sample['answer_tokens'],
        )

    def compute_question_loss(self, question_logits, sample):
        return compute_crossentropy(
            logits=question_logits,
            target=sample['questions'],
            padding_idx=self.question_padding_idx,
            num_tokens=sample['question_tokens'],
        )

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        answer_loss_sum = sum(log.get("answer_loss", 0) for log in logging_outputs)
        answer_tokens = sum(log.get("answer_tokens", 0) for log in logging_outputs)
        right_answers = sum(log.get('right_answers', 0) for log in logging_outputs)

        question_loss_sum = sum(log.get("question_loss", 0) for log in logging_outputs)
        question_tokens = sum(log.get("question_tokens", 0) for log in logging_outputs)
        right_tokens = sum(log.get("right_tokens", 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)

        if logging_outputs[0]['do_answer']:
            metrics.log_scalar('al', answer_loss_sum / ntokens / math.log(2), answer_tokens, round=2)
            metrics.log_scalar('ap', right_answers / answer_tokens, answer_tokens, round=2)

        if logging_outputs[0]['do_question']:        
            metrics.log_scalar('ql', question_loss_sum / ntokens / math.log(2), question_tokens, round=2)
            metrics.log_scalar('qp', right_tokens / question_tokens, question_tokens, round=2)

        metrics.log_derived('loss', get_avg_loss)


@register_criterion('jvaqg_latent')
class JVAQGLatentCriterion(JVAQGBaseCriterion):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--free-bits-p', type=float, default=0)

    def __init__(self, task, free_bits_p):
        super().__init__(task)
        self.free_bits_p = free_bits_p

    def forward(self, model, sample, reduce=True):
        model_output = model(sample)
        loss, sample_size, logging_output = forward_latent_model(self, model_output, sample, self.free_bits_p)
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        JVAQGBaseCriterion.reduce_metrics(logging_outputs)
        if 'kl_loss' in logging_outputs[0]:
            kl_loss_sum = sum(log.get('kl_loss', 0) for log in logging_outputs)
            nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

            metrics.log_scalar('kl', kl_loss_sum / nsentences / math.log(2), nsentences, round=3)
            metrics.log_derived('loss', get_avg_loss)


@register_criterion('jvaqg_latent_scale')
class JVAQGLatentScaleCriterion(JVAQGLatentCriterion):
    def __init__(self, task, free_bits_p, free_bits_b, block_loss_weight):
        super().__init__(task, free_bits_p)
        self.free_bits_b = free_bits_b
        self.block_loss_weight = block_loss_weight

    @staticmethod
    def add_args(parser):
        JVAQGLatentCriterion.add_args(parser)
        parser.add_argument('--free-bits-b', type=float, default=0.)
        parser.add_argument('--block-loss-weight', type=float, default=0.)

    def forward(self, model, sample, reduce=True):
        model_output = model(sample)
        loss, sample_size, logging_output = forward_latent_model(self, model_output, sample, free_bits_p=self.free_bits_p)

        if logging_output['do_question'] and logging_output['do_answer'] \
                 and self.block_loss_weight > 0.:
            answer_attn_scores = model_output['answer_attn_scores']     # bsz x atgt x seq
            answer_lengths = sample['answer_lengths']                   # bsz
            assert torch.max(answer_lengths) == answer_attn_scores.size(1)

            answer_mask = (~sequence_mask(answer_lengths)).float().unsqueeze(-1)
            answer_attn_scores = answer_attn_scores * answer_mask
            mean_ans_attn_score = torch.div(                            # bsz x seq
                torch.sum(answer_attn_scores, dim=1),
                answer_lengths.unsqueeze(-1)
            )

            question_attn_scores = model_output['question_attn_scores'] # bsz x qtgt x seq
            question_lengths = sample['question_lengths']               # bsz
            assert torch.max(question_lengths) == question_attn_scores.size(1)

            question_mask = (~sequence_mask(question_lengths)).float().unsqueeze(-1)
            question_attn_scores = question_attn_scores * question_mask
            mean_ques_attn_score = torch.div(                           # bsz x seq
                torch.sum(question_attn_scores, dim=1),
                question_lengths.unsqueeze(-1)
            )

            attn_loss = torch.sum(
                mean_ans_attn_score * torch.log(torch.div(mean_ans_attn_score, mean_ques_attn_score))
                ,
                dim=-1
            )                   # batch_size
            attn_loss = torch.clamp(attn_loss, min=self.free_bits_b)

            # avg_attn_score = 0.5 * (mean_ans_attn_score + mean_ques_attn_score)
            # js_loss = 0.5 * (mean_ans_attn_score * torch.log(torch.div(mean_ans_attn_score, avg_attn_score))) + \
            #     0.5 * (mean_ques_attn_score * torch.log(torch.div(mean_ques_attn_score, avg_attn_score)))
            # js_loss = torch.sum(js_loss, dim=-1)                        # bsz

            # if self.free_bits_b > 0.:
            #     js_loss = torch.clamp(js_loss, min=self.free_bits_b)

            attn_loss = torch.sum(attn_loss)
            mean_attn_loss = torch.div(attn_loss, sample['batch_size'])
            loss = loss + self.block_loss_weight * mean_attn_loss

            logging_output['block_loss'] = self.block_loss_weight * attn_loss

        return loss, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        JVAQGLatentCriterion.reduce_metrics(logging_outputs)
        if 'block_loss' in logging_outputs[0]:
            bl_loss_sum = sum(log.get('block_loss', 0) for log in logging_outputs)
            nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

            metrics.log_scalar('bl', bl_loss_sum / nsentences / math.log(2), nsentences, round=3)
            metrics.log_derived('loss', get_avg_loss)


@register_criterion('vaqg_s2s_base')
class VAQGS2SBaseCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(None)
        self.task = task
        self.target_padding_idx = task.target_dict.pad()

    def forward(self, model, sample, reduce=True):
        model_output = model(sample)
        loss, sample_size, logging_output = self.reconstruction_loss(sample, model_output)
        return loss, sample_size, logging_output

    def reconstruction_loss(self, sample, model_output):
        nsentences = sample['batch_size']
     
        target_logits = model_output['target_logits']
        target_loss, right_tokens = self.compute_target_loss(target_logits, sample)
        mean_loss = torch.div(target_loss, sample['ntokens'])

        sample_size = 1
        loss = mean_loss

        logging_output = {
            'loss': loss.data,
            'nsentences': nsentences,
            'ntokens': sample['ntokens'],
            'target_loss' : target_loss,
            'right_tokens': right_tokens,
        }
        return loss, sample_size, logging_output

    def compute_target_loss(self, target_logits, sample):
        return compute_crossentropy(
            logits=target_logits,
            target=sample['targets'],
            padding_idx=self.target_padding_idx,
            num_tokens=sample['ntokens'],
        )

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        target_loss_sum = sum(log.get("target_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        right_tokens = sum(log.get("right_tokens", 0) for log in logging_outputs)
       
        metrics.log_scalar('loss', target_loss_sum / ntokens / math.log(2), ntokens, round=2)
        metrics.log_scalar('acc', right_tokens / ntokens, ntokens, round=2)


@register_criterion('vaqg_s2s_latent')
class VAQGS2SLatentCriterion(VAQGS2SBaseCriterion):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--free-bits-p', type=float, default=0)

    def __init__(self, task, free_bits_p):
        super().__init__(task)
        self.free_bits_p = free_bits_p

    def forward(self, model, sample, reduce=True):
        model_output = model(sample)
        loss, sample_size, logging_output = forward_latent_model(self, model_output, sample, self.free_bits_p)
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        VAQGS2SBaseCriterion.reduce_metrics(logging_outputs)
        if 'kl_loss' in logging_outputs[0]:
            kl_loss_sum = sum(log.get('kl_loss', 0) for log in logging_outputs)
            nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

            metrics.log_scalar('kl', kl_loss_sum / nsentences / math.log(2), nsentences, round=3)
            metrics.log_derived('loss', get_avg_loss)


@register_criterion('vaqg_s2s_latent_scale')
class VAQGS2SLatentScaleCriterion(VAQGS2SLatentCriterion):
    pass


@register_criterion('vaqg_pipe_base')
class VAQGPipeBaseCriterion(VAQGS2SBaseCriterion):
    pass


@register_criterion('vaqg_pipe_latent')
class VAQGPipeLatentCriterion(VAQGS2SLatentCriterion):
    pass


@register_criterion('vaqg_pipe_latent_scale')
class VAQGPipeLatentScaleCriterion(VAQGS2SLatentScaleCriterion):
    pass