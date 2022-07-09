from functools import reduce
from operator import mul

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models.lstm import LSTMEncoder, AttentionLayer, Linear
from fairseq.data import Dictionary

from my_module.modules import FeatureProjection, build_answer_question_embed, build_target_embed
from my_module.utils import sequence_mask, reparameterize


class FeatureEncoder(LSTMEncoder):
    """Give image features, further encoded with a bidirectional LSTM"""
    def __init__(
        self,
        feature_dim,
        embed_dim,
        hidden_size,
        num_layers,
        dropout,
    ):
        super().__init__(
            dictionary=Dictionary(),
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_in=dropout,
            dropout_out=dropout,
            bidirectional=True,
            left_pad=False,
            pretrained_embed=FeatureProjection(feature_dim, embed_dim),
        )

    def forward(self, features):
        bsz, seqlen = features.size()[:2]
        feature_lengths = features.new_ones(bsz).long() * seqlen

        # embed tokens
        x = self.embed_tokens(features)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)

        x, (final_hiddens, final_cells) = self.lstm(x, (h0, c0))
        x = self.dropout_out_module(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, bsz)
            final_cells = self.combine_bidir(final_cells, bsz)

        encoder_padding_mask = sequence_mask(feature_lengths).t()

        return tuple((
                x,  # seq_len x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                final_cells,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seq_len x batch
        ))


class SequenceEncoder(LSTMEncoder):
    """An lstm wrapper to encode question or answer"""
    def __init__(
        self,
        srcdict,
        embed_dim,
        hidden_size,
        num_layers,
        dropout,
        pretrained_embed,
    ):
        super().__init__(
            dictionary=srcdict,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_in=dropout,
            dropout_out=dropout,
            bidirectional=True,
            left_pad=False,
            pretrained_embed=pretrained_embed,
        )

    def forward(self, tokens, lengths, enforce_sorted):
        return super().forward(
            src_tokens=tokens, src_lengths=lengths, enforce_sorted=enforce_sorted)


class FXEncoder(FairseqEncoder):
    def __init__(self, feature_dim, embed_dim, hidden_size, num_layers, dropout, with_feature_encoder=True):
        super().__init__(dictionary=None)
        if with_feature_encoder:
            self.feature_encoder = FeatureEncoder(
                feature_dim=feature_dim,
                embed_dim=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.feature_encoder = None


class FQAEncoder(FXEncoder):
    """Encode features, questions, and answers"""
    def __init__(
        self,
        question_dict,
        answer_dict,
        feature_dim,
        embed_dim,
        hidden_size,
        num_layers,
        dropout,
        share_token_embeddings,
        token_embed_path,
        with_feature_encoder=True
    ):
        super().__init__(feature_dim, embed_dim, hidden_size, num_layers, dropout, with_feature_encoder)
        answer_embed, question_embed = build_answer_question_embed(
            share_token_embeddings=share_token_embeddings,
            question_dict=question_dict,
            answer_dict=answer_dict,
            embed_dim=embed_dim,
            token_embed_path=token_embed_path,
        )

        self.question_encoder = SequenceEncoder(
            srcdict=question_dict,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_embed=question_embed,
        )
        self.answer_encoder = SequenceEncoder(
            srcdict=answer_dict,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_embed=answer_embed,
        )
        self.question_attn = AttentionLayer(
            input_embed_dim=2*hidden_size,
            source_embed_dim=2*hidden_size,
            output_embed_dim=hidden_size
        )
        self.answer_attn = AttentionLayer(
            input_embed_dim=2*hidden_size,
            source_embed_dim=2*hidden_size,
            output_embed_dim=hidden_size
        )

    def forward(
        self,
        features,
        questions,
        answers,
        question_lengths,
        answer_lengths,
        enforce_sorted=False,
    ):
        if self.feature_encoder:
            feature_out = self.feature_encoder(features)
        else:
            feature_out = None
        question_out = self.question_encoder(questions, question_lengths, enforce_sorted)
        answer_out = self.answer_encoder(answers, answer_lengths, enforce_sorted)
        return feature_out, question_out, answer_out

    def get_attns(self, feature_encodings, question_state, answer_state):
        """
        args:
            feature_encodings: seqlen x bsz x 2*hidden_size
            state: bsz x 2*hidden_size
        """
        quesattn, qattn_score = self.question_attn(question_state, feature_encodings, None)
        ansattn, aattn_score = self.answer_attn(answer_state, feature_encodings, None)
        return quesattn, qattn_score, ansattn, aattn_score


class FTEncoder(FXEncoder):
    """Encode features, questions, and answers"""
    def __init__(
        self,
        target_dict,
        feature_dim,
        embed_dim,
        hidden_size,
        num_layers,
        dropout,
        token_embed_path,
        with_feature_encoder=True,
    ):
        super().__init__(feature_dim, embed_dim, hidden_size, num_layers, dropout, with_feature_encoder)

        target_embed = build_target_embed(
            target_dict=target_dict,
            embed_dim=embed_dim,
            token_embed_path=token_embed_path,
        )
        self.target_encoder = SequenceEncoder(
            srcdict=target_dict,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_embed=target_embed,
        )
        self.target_attn = AttentionLayer(
            input_embed_dim=2*hidden_size,
            source_embed_dim=2*hidden_size,
            output_embed_dim=hidden_size
        )

    def forward(
        self,
        features,
        targets,
        target_lengths,
        enforce_sorted=False,
    ):
        if self.feature_encoder:
            feature_out = self.feature_encoder(features)
        else:
            feature_out = None
        target_out = self.target_encoder(targets, target_lengths, enforce_sorted)
        return feature_out, target_out

    def get_attns(self, feature_encodings, target_state):
        """
        args:
            feature_encodings: seqlen x bsz x 2*hidden_size
            state: bsz x 2*hidden_size
        """
        tgtattn, tattn_score = self.target_attn(target_state, feature_encodings, None)
        return tgtattn, tattn_score


class PriorEncoder(nn.Module):
    """Get the prior distribution of latent variable"""
    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.mu = nn.Linear(2*hidden_size, latent_dim)
        self.log_var = nn.Linear(2*hidden_size, latent_dim)

    def forward(self, feature_out):
        #! may be final state or mean
        # feature_state = feature_out[1][-1]          # bsz x 2*hidden
        feature_encodings = feature_out[0]
        feature_state = torch.mean(feature_encodings, dim=0)
        mus = self.mu(feature_state)
        log_vars = self.log_var(feature_state)
        return mus, log_vars


class FTPosteriorEncoder(nn.Module):
    def __init__(self, target_dict, feature_dim, embed_dim, hidden_size, latent_dim, num_layers, dropout, token_embed_path):
        super().__init__()
        self.ft_encoder = FTEncoder(
            target_dict=target_dict,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            with_feature_encoder=False,
            token_embed_path=token_embed_path,
        )
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_var = nn.Linear(hidden_size, latent_dim)
    
    def forward(self, feature_out, sample):
        _, target_out = self.ft_encoder(
            features=None,
            targets=sample['targets'],
            target_lengths=sample['target_lengths'],
            enforce_sorted=False,
        )
        feature_encodings = feature_out[0]     # seqlen x bsz x hidden
        target_state = target_out[1][-1]
        tgtattn, tattn_score = self.ft_encoder.get_attns(
            feature_encodings=feature_encodings,
            target_state=target_state,
        )
        mus = self.mu(tgtattn)
        log_vars = self.log_var(tgtattn)
        return mus, log_vars


class FQAPosteriorEncoder(nn.Module):
    """Get the posterior distribution of latent variable"""
    def __init__(
        self,
        question_dict,
        answer_dict,
        feature_dim,
        embed_dim,
        hidden_size,
        latent_dim,
        num_layers,
        dropout,
        token_embed_path,
        share_token_embeddings=True,
    ):
        super().__init__()
        self.fqa_encoder = FQAEncoder(
            question_dict=question_dict,
            answer_dict=answer_dict,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            token_embed_path=token_embed_path,
            share_token_embeddings=share_token_embeddings,
            with_feature_encoder=False
        )

        self.mu = nn.Linear(2*hidden_size, latent_dim)
        self.log_var = nn.Linear(2*hidden_size, latent_dim)

    def forward(self, feature_out, sample):
        _, question_out, answer_out = self.fqa_encoder(
            features=None,
            questions=sample['questions'],
            answers=sample['answers'],
            question_lengths=sample['question_lengths'],
            answer_lengths=sample['answer_lengths'],
            enforce_sorted=False,
        )
        feature_encodings = feature_out[0]     # seqlen x bsz x hidden
        question_state = question_out[1][-1]
        answer_state = answer_out[1][-1]
        
        quesattn, qattn_score, ansattn, aattn_score = self.fqa_encoder.get_attns(
            feature_encodings=feature_encodings,
            question_state=question_state,
            answer_state=answer_state,
        )
        cat_attn = torch.cat((quesattn, ansattn), dim=-1)
        mus = self.mu(cat_attn)
        log_vars = self.log_var(cat_attn)
        return mus, log_vars


class CompositeEncoder(FairseqEncoder):
    def __init__(self, args, task, use_latent=False, use_latent_scale=False):
        super().__init__(dictionary=None)
        self.feature_encoder = FeatureEncoder(
            feature_dim=args.feature_dim,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            dropout=args.dropout
        )
        self.use_latent = use_latent
        self.use_latent_scale = use_latent_scale

        if self.use_latent:
            self.prior_encoder = PriorEncoder(hidden_size=args.hidden_size, latent_dim=args.latent_dim)
            if hasattr(task, 'target_dict'):
                self.posterior_encoder = FTPosteriorEncoder(
                    target_dict=task.target_dict,
                    feature_dim=args.feature_dim,
                    embed_dim=args.embed_dim,
                    hidden_size=args.hidden_size,
                    latent_dim=args.latent_dim,
                    num_layers=args.layers,
                    dropout=args.dropout,
                    token_embed_path=args.token_embed_path,
                )
            else:
                self.posterior_encoder = FQAPosteriorEncoder(
                    question_dict=task.question_dict,
                    answer_dict=task.answer_dict,
                    feature_dim=args.feature_dim,
                    embed_dim=args.embed_dim,
                    hidden_size=args.hidden_size,
                    latent_dim=args.latent_dim,
                    num_layers=args.layers,
                    dropout=args.dropout,
                    share_token_embeddings=utils.eval_bool(args.share_token_embeddings),
                    token_embed_path=args.token_embed_path,
                )

            self.expand_latent = Linear(args.latent_dim, args.layers * 2 * args.hidden_size)
            if self.use_latent_scale:
                # grid_size = reduce(mul, args.grid_shape, 1)
                # self.block_scale = nn.Sequential(Linear(args.latent_dim, grid_size), nn.Sigmoid())
                self.block_scale = Linear(args.latent_dim, out_features=2*args.hidden_size, bias=False)
        else:
            self.prior_encoder = self.posterior_encoder = None

    def forward(self, sample):
        feature_out = list(self.feature_encoder(features=sample['features']))

        if self.use_latent:
            prior_mus, prior_log_vars = self.prior_encoder(feature_out=feature_out)
            if self.training:
                posterior_mus, posterior_log_vars = self.posterior_encoder(feature_out=feature_out, sample=sample)
                latent_variables = reparameterize(mus=posterior_mus, log_vars=posterior_log_vars)
            else:
                latent_variables = reparameterize(mus=prior_mus, log_vars=prior_log_vars)

            #! use latent variable to update hidden_state or scale encodings
            hiddens = feature_out[1]        # num_layers x batch x 2*hidden
            zs = self.expand_latent(latent_variables)
            zs = zs.view(zs.size(0), self.feature_encoder.num_layers, -1).permute(1, 0, 2)

            hiddens = hiddens + zs
            feature_out[1] = hiddens

            if self.use_latent_scale:
                intermediat_z = self.block_scale(latent_variables)          # bsz x 2hidden_size
                feature_encodings = feature_out[0]    # seq_len x bsz x hidden

                # compute attention
                _block_prob = (feature_encodings * intermediat_z.unsqueeze(0)).sum(dim=2)
                _block_prob = torch.softmax(_block_prob, dim=0) * 100
                _block_prob = torch.clamp(_block_prob, max=1)
                block_prob = _block_prob.unsqueeze(-1)  # srclen x bsz x 1

                # print(block_prob.view(-1))

                feature_encodings = block_prob * feature_encodings
                feature_out[0] = feature_encodings

                # _block_prob = self.block_scale(latent_variables)     # bsz x latent_dim
                # block_prob = _block_prob.transpose(0, 1).unsqueeze(-1)   # latent_dim x bsz x 1
                # feature_encodings = feature_out[0]    # seq_len x bsz x hidden
                # assert feature_encodings.size(0) == block_prob.size(0)
                # feature_encodings = block_prob * feature_encodings
                # feature_out[0] = feature_encodings

        out_dict = {
            'feature_encodings': feature_out[0],
            'feature_hiddens': feature_out[1],
            'feature_cells': feature_out[2],
            'feature_padding_mask': feature_out[3],
        }

        if self.use_latent:
            out_dict.update({
                'latent_variables': latent_variables        # bsz x hidden_size
            })
            if self.training:
                out_dict.update({
                    "distributions":{
                        'prior_mus': prior_mus,
                        'prior_log_vars': prior_log_vars,
                        'posterior_mus': posterior_mus,
                        'posterior_log_vars': posterior_log_vars,
                    },
                })
            if self.use_latent_scale:
                out_dict.update({'block_prob': _block_prob})

        return out_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        reordered_dict = {
            'feature_encodings': encoder_out['feature_encodings'].index_select(1, new_order),
            'feature_hiddens': encoder_out['feature_hiddens'].index_select(1, new_order),
            'feature_cells': encoder_out['feature_cells'].index_select(1, new_order),
            'feature_padding_mask': encoder_out['feature_padding_mask'].index_select(1, new_order),
        }
        if 'latent_variables' in encoder_out:
            reordered_dict.update({
                'latent_variables': encoder_out['latent_variables'].index_select(0, new_order)
            })

            if 'block_prob' in encoder_out:
                reordered_dict.update({
                    'block_prob': encoder_out['block_prob'].index_select(1, new_order)
                })
        return reordered_dict


class PipeFPPriorEncoder(nn.Module):
    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.pre_attn = AttentionLayer(
            input_embed_dim=2*hidden_size,
            source_embed_dim=2*hidden_size,
            output_embed_dim=hidden_size
        )
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_var = nn.Linear(hidden_size, latent_dim)

    def forward(self, feature_out, preliminary_out):
        feature_encodings = feature_out[0]
        pre_state = preliminary_out[1][-1]      # bsz x 2*hidden
        preattn, pattn_score = self.pre_attn(pre_state, feature_encodings, None)
        mus = self.mu(preattn)
        log_vars = self.log_var(preattn)
        return mus, log_vars


class PipeFTEncoder(nn.Module):
    """Encode features, questions, and answers"""
    def __init__(
        self,
        target_dict,
        embed_dim,
        hidden_size,
        num_layers,
        dropout,
        token_embed_path,
    ):
        super().__init__()

        target_embed = build_target_embed(
            target_dict=target_dict,
            embed_dim=embed_dim,
            token_embed_path=token_embed_path,
        )
        self.target_encoder = SequenceEncoder(
            srcdict=target_dict,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_embed=target_embed,
        )
        self.target_attn = AttentionLayer(
            input_embed_dim=2*hidden_size,
            source_embed_dim=2*hidden_size,
            output_embed_dim=hidden_size
        )

    def forward(
        self,
        targets,
        target_lengths,
        enforce_sorted=False,
    ):
        target_out = self.target_encoder(targets, target_lengths, enforce_sorted)
        return target_out

    def get_attns(self, feature_encodings, target_state):
        """
        args:
            feature_encodings: seqlen x bsz x 2*hidden_size
            state: bsz x 2*hidden_size
        """
        tgtattn, tattn_score = self.target_attn(target_state, feature_encodings, None)
        return tgtattn, tattn_score


class PipeFPTEncoder(nn.Module):
    """Encode features, questions, and answers"""
    def __init__(
        self,
        target_dict,
        embed_dim,
        hidden_size,
        num_layers,
        dropout,
        token_embed_path,
    ):
        super().__init__()
        target_embed = build_target_embed(
            target_dict=target_dict,
            embed_dim=embed_dim,
            token_embed_path=token_embed_path,
        )
        self.target_encoder = SequenceEncoder(
            srcdict=target_dict,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_embed=target_embed,
        )

        self.preliminary_attn = AttentionLayer(
            input_embed_dim=2*hidden_size,
            source_embed_dim=2*hidden_size,
            output_embed_dim=hidden_size
        )
        self.target_attn = AttentionLayer(
            input_embed_dim=2*hidden_size,
            source_embed_dim=2*hidden_size,
            output_embed_dim=hidden_size
        )

    def forward(
        self,
        targets,
        target_lengths,
        enforce_sorted=False,
    ):
        target_out = self.target_encoder(targets, target_lengths, enforce_sorted)
        return target_out

    def get_attns(self, feature_encodings, preliminary_state, target_state):
        """
        args:
            feature_encodings: seqlen x bsz x 2*hidden_size
            state: bsz x 2*hidden_size
        """
        preattn, pattn_score = self.preliminary_attn(preliminary_state, feature_encodings, None)
        targetattn, tattn_score = self.target_attn(target_state, feature_encodings, None)
        return preattn, pattn_score, targetattn, tattn_score


class PipeFTPosteriorEncoder(nn.Module):
    def __init__(self, target_dict, embed_dim, hidden_size, latent_dim, num_layers, dropout, token_embed_path):
        super().__init__()
        self.encoder = PipeFTEncoder(
            target_dict=target_dict,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            token_embed_path=token_embed_path,
        )
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_var = nn.Linear(hidden_size, latent_dim)

    def forward(self, feature_out, preliminary_out, sample):
        assert preliminary_out is None
        target_out = self.encoder(
            targets=sample['targets'],
            target_lengths=sample['target_lengths'],
        )
        feature_encodings = feature_out[0]     # seqlen x bsz x hidden
        target_state = target_out[1][-1]
        tgtattn, tattn_score = self.encoder.get_attns(
            feature_encodings=feature_encodings,
            target_state=target_state,
        )
        mus = self.mu(tgtattn)
        log_vars = self.log_var(tgtattn)
        return mus, log_vars


class PipeFPTPosteriorEncoder(nn.Module):
    def __init__(
        self,
        target_dict,
        embed_dim,
        hidden_size,
        latent_dim,
        num_layers,
        dropout,
        token_embed_path,
    ):
        super().__init__()
        self.fpt_encoder = PipeFPTEncoder(
            target_dict=target_dict,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            token_embed_path=token_embed_path,
        )
        self.mu = nn.Linear(2*hidden_size, latent_dim)
        self.log_var = nn.Linear(2*hidden_size, latent_dim)

    def forward(self, feature_out, preliminary_out, sample):
        target_out = self.fpt_encoder(
            targets=sample['targets'],
            target_lengths=sample['target_lengths'],
        )
        feature_encodings = feature_out[0]     # seqlen x bsz x hidden
        preliminary_state = preliminary_out[1][-1]
        target_state = target_out[1][-1]
        
        preattn, pattn_score, tgtattn, tattn_score = self.fpt_encoder.get_attns(
            feature_encodings=feature_encodings,
            preliminary_state=preliminary_state,
            target_state=target_state,
        )
        cat_attn = torch.cat((preattn, tgtattn), dim=-1)
        mus = self.mu(cat_attn)
        log_vars = self.log_var(cat_attn)
        return mus, log_vars


class PipelineEncoder(FairseqEncoder):
    def __init__(self, args, task, use_latent=False, use_latent_scale=False):
        super().__init__(dictionary=None)
        self.num_layers = args.layers
        if args.preliminary is None:
            self.encoder = FeatureEncoder(
                feature_dim=args.feature_dim,
                embed_dim=args.embed_dim,
                hidden_size=args.hidden_size,
                num_layers=args.layers,
                dropout=args.dropout
            )
        else:
            self.encoder = FTEncoder(
                target_dict=task.pre_dict,
                feature_dim=args.feature_dim,
                embed_dim=args.embed_dim,
                hidden_size=args.hidden_size,
                num_layers=args.layers,
                dropout=args.dropout,
                token_embed_path=args.token_embed_path,
                with_feature_encoder=True,
            )
            del self.encoder.target_attn
        self.use_latent = use_latent
        self.use_latent_scale = use_latent_scale

        if self.use_latent:
            if args.preliminary is None:
                self.prior_encoder = PriorEncoder(hidden_size=args.hidden_size, latent_dim=args.latent_dim)
                self.posterior_encoder = PipeFTPosteriorEncoder(
                    target_dict=task.target_dict,
                    embed_dim=args.embed_dim,
                    hidden_size=args.hidden_size,
                    latent_dim=args.latent_dim,
                    num_layers=args.layers,
                    dropout=args.dropout,
                    token_embed_path=args.token_embed_path,
                )
            else:
                self.prior_encoder = PipeFPPriorEncoder(hidden_size=args.hidden_size, latent_dim=args.latent_dim)
                self.posterior_encoder = PipeFPTPosteriorEncoder(
                    target_dict=task.target_dict,
                    embed_dim=args.embed_dim,
                    hidden_size=args.hidden_size,
                    latent_dim=args.latent_dim,
                    num_layers=args.layers,
                    dropout=args.dropout,
                    token_embed_path=args.token_embed_path,
                )

            self.expand_latent = Linear(args.latent_dim, args.layers * 2 * args.hidden_size)
            if self.use_latent_scale:
                # grid_size = reduce(mul, args.grid_shape, 1)
                # self.block_scale = nn.Sequential(Linear(args.latent_dim, grid_size), nn.Sigmoid())
                self.block_scale = Linear(args.latent_dim, out_features=2*args.hidden_size, bias=False)
        else:
            self.prior_encoder = self.posterior_encoder = None

    def forward(self, sample):
        if sample['preliminaries'] is not None:
            feature_out, preliminary_out = self.encoder(
                features=sample['features'],
                targets=sample['preliminaries'],
                target_lengths=sample['preliminary_lengths'],
            )
            feature_out = list(feature_out)
            preliminary_out = list(preliminary_out)
        else:
            feature_out = list(self.encoder(features=sample['features']))
            preliminary_out = None

        encoder_out = feature_out
        if preliminary_out is not None:
            encoder_lengths = feature_out[0].size(0) + sample['preliminary_lengths']
            encoder_out[3] = sequence_mask(encoder_lengths).t()

            encoder_out[1] = encoder_out[1] + preliminary_out[1]
            encoder_out[2] = encoder_out[2] + preliminary_out[2]

        if self.use_latent:
            if preliminary_out is not None:
                assert isinstance(self.prior_encoder, PipeFPPriorEncoder)
                prior_mus, prior_log_vars = self.prior_encoder(feature_out=feature_out, preliminary_out=preliminary_out)
            else:
                prior_mus, prior_log_vars = self.prior_encoder(feature_out=feature_out)

            if self.training:
                posterior_mus, posterior_log_vars = self.posterior_encoder(feature_out=feature_out, preliminary_out=preliminary_out, sample=sample)
                latent_variables = reparameterize(mus=posterior_mus, log_vars=posterior_log_vars)
            else:
                latent_variables = reparameterize(mus=prior_mus, log_vars=prior_log_vars)

            #! use latent variable to update hidden_state or scale encodings
            hiddens = encoder_out[1]        # num_layers x batch x 2*hidden
            zs = self.expand_latent(latent_variables)
            zs = zs.view(zs.size(0), self.num_layers, -1).permute(1, 0, 2)
            hiddens = hiddens + zs
            encoder_out[1] = hiddens

            if self.use_latent_scale:
                intermediat_z = self.block_scale(latent_variables)          # bsz x 2hidden_size
                feature_encodings = feature_out[0]    # seq_len x bsz x hidden

                # compute attention
                _block_prob = (feature_encodings * intermediat_z.unsqueeze(0)).sum(dim=2)
                _block_prob = torch.softmax(_block_prob, dim=0) * 100
                _block_prob = torch.clamp(_block_prob, max=1)
                block_prob = _block_prob.unsqueeze(-1)  # srclen x bsz x 1
                feature_encodings = block_prob * feature_encodings
                feature_out[0] = feature_encodings
                # _block_prob = self.block_scale(latent_variables)     # bsz x latent_dim
                # block_prob = _block_prob.transpose(0, 1).unsqueeze(-1)   # latent_dim x bsz x 1
                # feature_encodings = encoder_out[0]    # seq_len x bsz x hidden
                # assert feature_encodings.size(0) == block_prob.size(0)
                # feature_encodings = block_prob * feature_encodings
                # encoder_out[0] = feature_encodings

        if preliminary_out is not None:
            encoder_out[0] = torch.cat((encoder_out[0], preliminary_out[0]), dim=0)

        out_dict = {
            'feature_encodings': encoder_out[0],
            'feature_hiddens': encoder_out[1],
            'feature_cells': encoder_out[2],
            'feature_padding_mask': encoder_out[3],
        }

        if self.use_latent:
            out_dict.update({
                'latent_variables': latent_variables        # bsz x hidden_size
            })
            if self.training:
                out_dict.update({
                    "distributions":{
                        'prior_mus': prior_mus,
                        'prior_log_vars': prior_log_vars,
                        'posterior_mus': posterior_mus,
                        'posterior_log_vars': posterior_log_vars,
                    },
                })
            if self.use_latent_scale:
                out_dict.update({'block_prob': _block_prob})

        return out_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        reordered_dict = {
            'feature_encodings': encoder_out['feature_encodings'].index_select(1, new_order),
            'feature_hiddens': encoder_out['feature_hiddens'].index_select(1, new_order),
            'feature_cells': encoder_out['feature_cells'].index_select(1, new_order),
            'feature_padding_mask': encoder_out['feature_padding_mask'].index_select(1, new_order),
        }
        if 'latent_variables' in encoder_out:
            reordered_dict.update({
                'latent_variables': encoder_out['latent_variables'].index_select(0, new_order)
            })

            if 'block_prob' in encoder_out:
                reordered_dict.update({
                    'block_prob': encoder_out['block_prob'].index_select(1, new_order)
                })
        return reordered_dict