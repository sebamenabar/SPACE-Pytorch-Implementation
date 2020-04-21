import torch
import torch.nn as nn
import torchvision.transforms as T
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

from ssv import VPNet
from utils.ssv import rsample_gaussean, std, calculate_center_scale_wrt_img
from space_modules import GlimpseEncoder, GlimpseDecoder, ComponentEncoder
from config import cfg
from clevr_dataset import CLEVRDataset, collate_fn


class PLReconstructer(pl.LightningModule):
    def __init__(self, hparams=None):
        if hparams is None:
            self.cfg = cfg
        else:
            self.cfg = cfg

        self.use_cuda = self.cfg.gpu_id != "-1"
        self.__build_model()

    def __build_model(self):
        self.vpnet = VPNet(instance_norm=True)
        self.glimpse_encoder = GlimpseEncoder()
        self.glimpse_decoder = GlimpseDecoder()
        self.component_encoder = ComponentEncoder()
        self.beta = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def efficient_render_scene(
        self,
        scene,
        beta=None,
        sample_presence=None,
        sample_presence_std=2,
        presence_thresh=0.001,
    ):
        if beta is None:
            beta = self.beta
        if sample_presence is None:
            sample_presence = self.training

        bsz, img_channels, img_height, img_width = inp.size()
        z_bg = scene["bg"]["z_bg"]
        scale = scene["fg"]["scale"]
        shift = scene["fg"]["shift"]
        depth = scene["fg"]["depth"]
        presence_logits = scene["fg"]["presence_logits"]
        if sample_presence:
            _presence_logits = rsample_gaussean(presence_logits, sample_presence_std)
        else:
            _presence_logits = presence_logits
        presence = torch.sigmoid(_presence_logits * beta)

        presence_indices = presence >= presence_thresh
        k = max(presence_indices.flatten(1).sum(1))
        topk_p_values, topk_indices = sampled_presence.flatten(1, 2).topk(k=k, dim=1)
        topk_indices = topk_indices.squeeze(-1)
        batch_indices = torch.arange(bsz).unsqueeze(-1).repeat_interleave(k.item(), 1)

        glimpses = stn(inp, topk_center_wrt_img__11, topk_scale_wrt_img, glimpse_shape)
        encoded_glimpses = glimpse_encoder(glimpses.flatten(0, 1))
        decoded_glimpses = glimpse_decoder(encoded_glimpses)
        batch_decoded_glimpses = decoded_glimpses.view(
            bsz, k, channels + 1, *glimpse_shape
        )
        alpha_hat_glimpse = torch.sigmoid(
            beta * batch_decoded_glimpses[:, :, [0]]
        ) * topk_p_values.unsqueeze(-1).unsqueeze(-1)
        y_glimpse = torch.sigmoid(batch_decoded_glimpses[:, :, 1:]) * alpha_hat_glimpse

        cat_decoded_glimpse = torch.cat((alpha_hat_glimpse, y_glimpse), 2).flatten(0, 1)
        cat_decoded_img = stn(
            cat_decoded_glimpse,
            topk_center_wrt_img__11,
            topk_scale_wrt_img,
            (height, width),
            inverse=True,
        )
        cat_decoded_img = cat_decoded_img.view(bsz, k, 4, height, width)

        alpha_hat_img = cat_decoded_img[:, :, [0]]
        y_img = cat_decoded_img[:, :, 1:]

        topk_depth = depth.flatten(1, 2)[batch_indices, topk_indices, :]
        w = (100 * topk_depth.view(bsz, k, 1, 1, 1) * alpha_hat_img).softmax(1)
        alpha = (w * alpha_hat_img).sum(1)
        mu_fg = (y_img * w).sum(1)

        decoded_bg = torch.sigmoid(component_decoder(z_bg))

        return {
            "alpha": alpha,
            "mu_fg": mu_fg,
            "decoded_bg": decoded_bg,
            "presence": torch.sigmoid(presence_logits),
            "alpha_hat_glimpse": alpha_hat_glimpse.detach(),
            "glimpses": glimpses.detach(),
            "y_glimpse": y_glimpse.detach(),
            "topk_p_values": topk_p_values.detach(),
            "topk_indices": topk_indices,
            "k": k,
        }
