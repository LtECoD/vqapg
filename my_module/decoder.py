import torch

from typing import Dict, Optional
from torch import Tensor

from fairseq import utils
from fairseq.models.lstm import LSTMDecoder
from fairseq.models.fairseq_decoder import FairseqDecoder

from my_module.modules import build_answer_question_embed, build_target_embed

class Decoder(LSTMDecoder):
    def __init__(
        self,
        dictionary,
        embed_dim,
        hidden_size,
        num_layers,
        dropout,
        encoder_output_units,
        pretrained_embed,
        share_input_output_embed,
    ):
        super().__init__(
            dictionary=dictionary,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            out_embed_dim=hidden_size,
            num_layers=num_layers,
            dropout_in=dropout,
            dropout_out=dropout,
            attention=True,
            encoder_output_units=encoder_output_units,
            pretrained_embed=pretrained_embed,
            share_input_output_embed=share_input_output_embed
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state,
    ):
        """
        Similar to *forward* but only return features.
        """
        # get outputs from encoder
        encoder_outs = encoder_out['feature_encodings']
        encoder_hiddens = encoder_out['feature_hiddens']
        encoder_cells = encoder_out['feature_cells']
        encoder_padding_mask = encoder_out['feature_padding_mask']

        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        elif encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            assert ValueError('encoder_out should not be None')

        assert (
            srclen > 0 or self.attention is None
        ), "attention is not supported if there are no encoder outputs"
        attn_scores = (
            x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        )
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            assert attn_scores is not None
            out, attn_scores[:, j, :] = self.attention(
                hidden, encoder_outs, encoder_padding_mask
            )

            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        assert attn_scores is not None
        attn_scores = attn_scores.transpose(0, 2)

        return self.output_layer(x), attn_scores


class SingleDecoder(FairseqDecoder):
    def __init__(self, args, task):
        super().__init__(dictionary=task.target_dict)
        target_embed = build_target_embed(
            target_dict=task.target_dict,
            embed_dim=args.embed_dim,
            token_embed_path=args.token_embed_path,
        )
        self.decoder = Decoder(
            dictionary=task.target_dict,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            dropout=args.dropout,
            encoder_output_units=2*args.hidden_size,
            pretrained_embed=target_embed,
            share_input_output_embed=utils.eval_bool(args.share_input_output_embed),
        )

    def forward(self, prev_tokens, encoder_out, incremental_state):
        target_logits, target_attn_scores = self.decoder(
            prev_output_tokens=prev_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        decoder_out = {
            "target_logits": target_logits,
            'target_attn_scores': target_attn_scores,
        }
        
        if "distributions" in encoder_out:
            decoder_out.update({
                "distributions": encoder_out['distributions']
            })
        return decoder_out


class CompositeDecoder(FairseqDecoder):
    def __init__(self, args, task):
        super().__init__(dictionary=None)

        answer_embed, question_embed = build_answer_question_embed(
            share_token_embeddings=utils.eval_bool(args.share_token_embeddings),
            question_dict=task.question_dict,
            answer_dict=task.answer_dict,
            token_embed_path=args.token_embed_path,
            embed_dim=args.embed_dim,
        )

        if task.do_answer:
            self.answer_decoder = Decoder(
                dictionary=task.answer_dict,
                embed_dim=args.embed_dim,
                hidden_size=args.hidden_size,
                num_layers=args.layers,
                dropout=args.dropout,
                encoder_output_units=2*args.hidden_size,
                pretrained_embed=answer_embed,
                share_input_output_embed=utils.eval_bool(args.share_input_output_embed),
            )
        else:
            self.answer_decoder = None
        
        if task.do_question:
            self.question_decoder = Decoder(
                dictionary=task.question_dict,
                embed_dim=args.embed_dim,
                hidden_size=args.hidden_size,
                num_layers=args.layers,
                dropout=args.dropout,
                encoder_output_units=2*args.hidden_size,
                pretrained_embed=question_embed,
                share_input_output_embed=utils.eval_bool(args.share_input_output_embed),
            )
        else:
            self.question_decoder = None

    def forward_answer_decoder(self, prev_tokens, encoder_out, incremental_state):
        if self.answer_decoder is not None:
            answer_logits, answer_attn_scores = self.answer_decoder(
                prev_output_tokens=prev_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )
        else:
            answer_logits = answer_attn_scores = None
        return answer_logits, answer_attn_scores

    def forward_question_decoder(self, prev_tokens, encoder_out, incremental_state):
        if self.question_decoder is not None:
            question_logits, question_attn_scores = self.question_decoder(
                prev_output_tokens=prev_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )
        else:
            question_logits = question_attn_scores = None
        return question_logits, question_attn_scores

    def forward(
        self,
        prev_answer_tokens,
        prev_question_tokens,
        encoder_out,
        answer_incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        question_incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):

        answer_logits, answer_attn_scores = self.forward_answer_decoder(
            prev_tokens=prev_answer_tokens,
            encoder_out=encoder_out,
            incremental_state=answer_incremental_state,
        )

        question_logits, question_attn_scores = self.forward_question_decoder(
            prev_tokens=prev_question_tokens,
            encoder_out=encoder_out,
            incremental_state=question_incremental_state,
        )

        decoder_out = {
            "answer_logits": answer_logits,
            'answer_attn_scores': answer_attn_scores,
            "question_logits": question_logits,
            "question_attn_scores": question_attn_scores,
        }

        if "distributions" in encoder_out:
            decoder_out.update({
                "distributions": encoder_out['distributions']
            })
        return decoder_out
