from pyhocon import ConfigTree
# from torchpie.logging import logger

import torch
from torch import nn, Tensor


class ModelFactory:

    def __init__(self, opt):
        self.opt = opt

    def build(self, device=torch.device('cpu'), wrapper=lambda x: x):
        emb_dim = self.opt.emb_dim
        hid_dim = self.opt.hid_dim
        duration = self.opt.duration
        fps = self.opt.fps
        layout = self.opt.pose_layout
        events_per_sec = self.opt.events_per_sec
        ckpt = self.opt.ckpt
        streams = self.opt.streams
        audio_duration = duration

        if self.opt.model_name == 'music_transformer':
            from .music_transformer_dev.music_transformer import music_transformer_dev_baseline
            pose_seq2seq = music_transformer_dev_baseline(
                240 + 3,
                d_model=emb_dim,
                dim_feedforward=emb_dim * 2,
                encoder_max_seq=int(duration * fps),
                decoder_max_seq=self.opt.decoder_max_seq,
                layout=layout,
                num_encoder_layers=self.opt.num_encoder_layers,
                num_decoder_layers=self.opt.num_decoder_layers,
                rpr=self.opt.rpr,
                use_control='control',
                rnn=self.opt.rnn,
                layers=self.opt.pose_net_layers
            )
            if ckpt != 'ckpt':
                pass
                # TODO load weight for finetune

        else:
            raise Exception

        pose_seq2seq = pose_seq2seq.to(device)
        # pose_seq2seq = wrapper(pose_seq2seq)

        return pose_seq2seq
