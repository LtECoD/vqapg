from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from my_module.encoder import CompositeEncoder, PipelineEncoder
from my_module.decoder import CompositeDecoder, SingleDecoder


class BaseModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        # model arguments
        parser.add_argument('--feature-dim', type=int, metavar='N')
        parser.add_argument('--embed-dim', type=int, metavar='N')
        parser.add_argument('--hidden-size', type=int, metavar='N')
        parser.add_argument('--layers', type=int, metavar='N')
        parser.add_argument('--dropout', type=float, metavar='D')

        # for decoder
        parser.add_argument('--token-embed-path', type=str, metavar='S')
        parser.add_argument('--share-token-embeddings', type=str, metavar='BOOL')
        parser.add_argument('--share-input-output-embed', type=str, metavar='BOOL')        

        # for latent variable
        parser.add_argument('--latent-dim', type=int, metavar='N')

    @staticmethod
    def build_encoder(cls, args, task):
        return CompositeEncoder(args=args, task=task, use_latent=False)
    
    @staticmethod
    def build_decoder(cls, args, task):
        return CompositeDecoder(args, task)

    @classmethod
    def build_model(cls, args, task):
        assert utils.eval_bool(args.do_answer) or utils.eval_bool(args.do_question)
        encoder = cls.build_encoder(cls, args, task)
        decoder = cls.build_decoder(cls, args, task)
        return cls(encoder, decoder)


@register_model('jvaqg_base')
class JVAQG(BaseModel):
    def forward(self, sample):
        encoder_out = self.encoder(sample)
        decoder_out = self.decoder(
            prev_answer_tokens=sample['prev_answer_tokens'],
            prev_question_tokens=sample['prev_question_tokens'],
            encoder_out=encoder_out,
        )
        return decoder_out


@register_model('jvaqg_latent')
class JAVQGLatent(JVAQG):
    @staticmethod
    def build_encoder(cls, args, task):
        return CompositeEncoder(args=args, task=task, use_latent=True, use_latent_scale=False)


@register_model('jvaqg_latent_scale')
class JVAQGLatentScale(JAVQGLatent):
    @staticmethod
    def build_encoder(cls, args, task):
        return CompositeEncoder(args=args, task=task, use_latent=True, use_latent_scale=True)


@register_model('vaqg_s2s_base')
class VAQGS2S(BaseModel):
    @staticmethod
    def build_decoder(cls, args, task):
        return SingleDecoder(args=args, task=task)

    def forward(self, sample):
        encoder_out = self.encoder(sample)
        decoder_out = self.decoder(
            prev_tokens=sample['prev_tokens'],
            encoder_out=encoder_out,
            incremental_state=None,
        )
        return decoder_out


@register_model('vaqg_s2s_latent')
class VAQGS2SLatent(VAQGS2S):
    @staticmethod
    def build_encoder(cls, args, task):
        return CompositeEncoder(args=args, task=task, use_latent=True, use_latent_scale=False)


@register_model('vaqg_s2s_latent_scale')
class VAQGS2SLatentScale(VAQGS2SLatent):
    @staticmethod
    def build_encoder(cls, args, task):
        return CompositeEncoder(args=args, task=task, use_latent=True, use_latent_scale=True)


@register_model('vaqg_pipe_base')
class VAQGPipe(VAQGS2S):
    @staticmethod
    def build_encoder(cls, args, task):
        return PipelineEncoder(args=args, task=task)


@register_model('vaqg_pipe_latent')
class VAQGPipeLatent(VAQGPipe):
    @staticmethod
    def build_encoder(cls, args, task):
        return PipelineEncoder(args=args, task=task, use_latent=True)


@register_model('vaqg_pipe_latent_scale')
class VAQGPipeLatentScale(VAQGPipeLatent):
    @staticmethod
    def build_encoder(cls, args, task):
        return PipelineEncoder(args=args, task=task, use_latent=True, use_latent_scale=True)


def base_architecture(args):
    args.feature_dim = getattr(args, 'feature_dim', 2048)
    args.embed_dim = getattr(args, 'embed_dim', 512)
    args.hidden_size = getattr(args, 'hidden_size', 512)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.layers = getattr(args, 'layers', 2)

    args.share_token_embeddings = getattr(args, 'share_token_embeddings', 'False')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', 'False')
    args.token_embed_path = getattr(args, 'token_embed_path', None)


def latent_architecture(args):
    base_architecture(args)
    args.latent_dim = getattr(args, 'latent_dim', 512)


@register_model_architecture('jvaqg_base', 'jvaqg_base')
def jvaqg_base_architecture(args):
    base_architecture(args)


@register_model_architecture('jvaqg_latent', 'jvaqg_latent')
def jvaqg_latent_architecture(args):
    latent_architecture(args)


@register_model_architecture('jvaqg_latent_scale', 'jvaqg_latent_scale')
def jvaqg_latent_scale_architecture(args):
    latent_architecture(args)


@register_model_architecture('vaqg_s2s_base', 'vaqg_s2s_base')
def vaqg_s2s_base_architecture(args):
    base_architecture(args)


@register_model_architecture('vaqg_s2s_latent', 'vaqg_s2s_latent')
def vaqg_s2s_latent_architecture(args):
    latent_architecture(args)


@register_model_architecture('vaqg_s2s_latent_scale', 'vaqg_s2s_latent_scale')
def vaqg_s2s_latent_scale_architecture(args):
    latent_architecture(args)


@register_model_architecture('vaqg_pipe_base', 'vaqg_pipe_base')
def vaqg_pipe_base_architecture(args):
    base_architecture(args)


@register_model_architecture('vaqg_pipe_latent', 'vaqg_pipe_latent')
def vaqg_pipe_latent_architecture(args):
    latent_architecture(args)


@register_model_architecture('vaqg_pipe_latent_scale', 'vaqg_pipe_latent_scale')
def vaqg_pipe_latent_scale_architecture(args):
    latent_architecture(args)