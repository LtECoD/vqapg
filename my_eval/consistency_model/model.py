import torch
import torch.nn as nn
import numpy as np
np.random.seed(10)

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models import FairseqEncoderModel

from my_module.encoder import FQAEncoder

@register_model('vqac')
class VQACModel(FairseqEncoderModel):
    @staticmethod
    def add_args(parser):
        # model arguments
        parser.add_argument('--feature-dim', type=int, metavar='N')
        parser.add_argument('--embed-dim', type=int, metavar='N')
        parser.add_argument('--hidden-size', type=int, metavar='N')
        parser.add_argument('--layers', type=int, metavar='N')
        parser.add_argument('--dropout', type=float, metavar='D')
        parser.add_argument('--token-embed-path', type=str, metavar='S')
        parser.add_argument('--share-token-embeddings', type=str, metavar='BOOL')

    def __init__(self, encoder, classifier):
        super().__init__(encoder)
        self.classifier = classifier

    @classmethod
    def build_model(cls, args, task):
        encoder = FQAEncoder(
            question_dict=task.question_dict,
            answer_dict=task.answer_dict,
            feature_dim=args.feature_dim,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            dropout=args.dropout,
            share_token_embeddings=utils.eval_bool(args.share_token_embeddings),
            token_embed_path=args.token_embed_path,
        )
        classifier = nn.Sequential(
            nn.Linear(2*args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid()
        )
        return cls(encoder, classifier)

    def get_logits(
        self,
        feature_encodings,
        question_state,
        answer_state,
    ):
        quesattn, _, ansattn, _ = self.encoder.get_attns(
            feature_encodings=feature_encodings,
            question_state=question_state,
            answer_state=answer_state,
        )
        logits = self.classifier(torch.cat((quesattn, ansattn), dim=-1))
        return logits.view(-1)      # bsz

    def forward(
        self,
        features,
        questions,
        answers,
        question_lengths,
        answer_lengths,
        requires_neg=False,
    ):
        bsz = features.size(0)
        feature_out, question_out, answer_out = self.encoder(
            features=features,
            questions=questions,
            answers=answers,
            question_lengths=question_lengths,
            answer_lengths=answer_lengths,
            enforce_sorted=False,
        )
        feature_encodings = feature_out[0]     # seqlen x bsz x hidden
        question_state = question_out[1][-1]
        answer_state = answer_out[1][-1]
        logits = self.get_logits(
            feature_encodings,
            question_state,
            answer_state,
        )
        
        targets = torch.ones(bsz).to(logits.device).long()

        if requires_neg:
            mode = np.random.choice([0,1,2,3])
            random_indices = np.random.permutation(bsz)
            if mode == 0:
                # shuffle image
                feature_encodings = feature_encodings[:, random_indices]
            elif mode == 1:
                # shuffle question
                question_state = question_state[random_indices]
            elif mode == 2:
                # shuffle answer
                answer_state = answer_state[random_indices]
            elif mode == 3:
                # shuffle all
                feature_encodings = feature_encodings[:, random_indices]

                random_indices = np.random.permutation(bsz)
                question_state = question_state[random_indices]

                random_indices = np.random.permutation(bsz)
                answer_state = answer_state[random_indices]
            else:
                raise ValueError(f'random mode can not be {mode}')
            
            negtive_logits = self.get_logits(
                feature_encodings,
                question_state,
                answer_state,
            )

            negtive_targets = targets.new_zeros(bsz).to(negtive_logits.device)
            
            logits = torch.cat((logits, negtive_logits), dim=0)
            targets = torch.cat((targets, negtive_targets), dim=0)
        return logits, targets


@register_model_architecture('vqac', 'vqac')
def vqac_architecture(args):
    args.feature_dim = getattr(args, 'feature_dim', 2048)
    args.embed_dim = getattr(args, 'embed_dim', 512)
    args.hidden_size = getattr(args, 'hidden_size', 512)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.layers = getattr(args, 'layers', 2)
    args.token_embed_path = getattr(args, 'token_embed_path', None)
    args.share_token_embeddings = getattr(args, 'share_token_embeddings', 'True')


