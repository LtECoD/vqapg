import math
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.nn.modules.loss import MultiLabelMarginLoss


@register_criterion("vqac")
class JVACCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(None)
        self.task = task

    def forward(self, model, sample, reduce=True):
        logits, targets = model(**sample["net_input"], requires_neg=True)

        sample_size = targets.numel()

        # lprobs = F.log_softmax(logits, dim=-1)
        # loss = F.nll_loss(lprobs, targets, reduction='sum')
        loss = F.mse_loss(logits, targets.float(), reduction='sum')

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        # preds = logits.argmax(dim=-1)
        preds = (logits > 0.5).long()
        logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
