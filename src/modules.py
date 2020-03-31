from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg
from layers import NeuralTensor
from misc_utils import select_and_pad_on_presence, process_decoded_transform
from sample_utils import sample_gaussean, sample_bernoulli, sample_gaussean_sp


class ImageEncoder(nn.Module):
    def __init__(
        self,
        ncs=[16, 32, 64, 128, 256, 128],  # num channels
        kss=[4, 4, 4, 3, 3, 1],  # kernel sizes
        ss=[2, 2, 2, 1, 1, 1],  # strides
        ngs=[4, 8, 8, 16, 32, 16],  # num groups
        pds=[1, 1, 1, 1, 1, 0],  # paddings
        res_ncs=[128, 128],
        res_kss=[3, 3],
        res_ss=[1, 1],
        res_ngs=[16, 16],
        act="celu",
    ):
        super().__init__()

        if act == "celu":
            act = nn.CELU

        conv_layers = []
        prev_num_channels = 3
        for nc, ks, stride, ng, pd in zip(ncs, kss, ss, ngs, pds):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=prev_num_channels,
                        out_channels=nc,
                        kernel_size=ks,
                        stride=stride,
                        padding=pd,
                    ),
                    nn.GroupNorm(num_groups=ng, num_channels=nc),
                    act(),
                )
            )
            prev_num_channels = nc
        self.conv_layers = nn.Sequential(*conv_layers)

        res_conn = []
        for nc, ks, stride, ng in zip(res_ncs, res_kss, res_ss, res_ngs):
            res_conn.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=prev_num_channels,
                        out_channels=nc,
                        kernel_size=ks,
                        stride=stride,
                        padding=1,
                    ),
                    nn.GroupNorm(num_groups=ng, num_channels=nc),
                    act(),
                )
            )
            prev_num_channels = nc
        self.res_conn = nn.Sequential(*res_conn)
        self.res_enc = nn.Sequential(
            nn.Conv2d(
                in_channels=prev_num_channels * 2,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(num_groups=16, num_channels=128),
            act(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        res = self.res_conn(x)
        out = self.res_enc(torch.cat((x, res), dim=1))

        return out


class GlimpseEncoder(nn.Module):
    def __init__(
        self,
        ncs=[16, 32, 32, 64, 128, 256],
        kss=[3, 4, 3, 4, 4, 4, 4],
        ss=[1, 2, 1, 2, 2, 1],
        ngs=[4, 8, 4, 8, 8, 16],
        out_proj_dim=64,
    ):
        super().__init__()
        conv_layers = []
        prev_nc = 3
        for nc, ks, s, ng in zip(ncs, kss, ss, ngs):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=prev_nc,
                        out_channels=nc,
                        kernel_size=ks,
                        stride=s,
                        padding=1,
                    ),
                    nn.GroupNorm(num_groups=ng, num_channels=nc),
                    nn.CELU(),
                )
            )
            prev_nc = nc
        self.conv_layers = nn.Sequential(*conv_layers)
        self.out_proj = nn.Conv2d(
            in_channels=prev_nc, out_channels=out_proj_dim, kernel_size=1,
        )
        self.out_pool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        return self.out_pool(self.out_proj(self.conv_layers(x)))


class GlimpseDecoder(nn.Module):
    def __init__(
        self,
        conv_ncs=[256, 128, 128, 64, 32],
        conv_ss=[1, 1, 1, 1, 1, 1],
        conv_gns=[16, 16, 16, 8, 8],
        conv_kss=[1, 3, 3, 3, 3],
        conv_pds=[0, 1, 1, 1, 1],
        sub_conv_factors=[2, 2, 2, 2, 2],
        sub_conv_ncs=[128, 128, 64, 32, 16],  # last item for first element edge case
        sub_conv_ss=[1, 1, 1, 1, 1],
        sub_conv_gns=[16, 16, 8, 8, 4],
        out_proj_dim=16,
        out_proj_ks=3,
        out_proj_stride=1,
    ):
        super().__init__()

        conv_layers = OrderedDict([])
        prev_num_channels = 32
        for i, (nc, ks, s, gn, pd, sc_f, sc_nc, sc_s, sc_gn) in enumerate(
            zip(
                conv_ncs,
                conv_kss,
                conv_ss,
                conv_gns,
                conv_pds,
                sub_conv_factors,
                sub_conv_ncs,
                sub_conv_ss,
                sub_conv_gns,
            )
        ):
            conv_layers[f"group_conv{i}"] = nn.Sequential(
                OrderedDict(
                    [
                        (
                            f"conv{i}",
                            nn.Sequential(
                                nn.Conv2d(
                                    in_channels=prev_num_channels,
                                    out_channels=nc,
                                    kernel_size=ks,
                                    stride=s,
                                    padding=pd,
                                ),
                                nn.GroupNorm(num_groups=gn, num_channels=nc),
                                nn.CELU(),
                            ),
                        ),
                        (
                            f"sub_conv{i}",
                            nn.Sequential(
                                nn.Conv2d(
                                    in_channels=nc,
                                    out_channels=sc_nc * sc_f ** 2,
                                    stride=sc_s,
                                    kernel_size=1,
                                ),
                                nn.PixelShuffle(sc_f),
                                nn.GroupNorm(num_groups=sc_gn, num_channels=sc_nc),
                                nn.CELU(),
                            ),
                        ),
                    ]
                )
            )
            prev_num_channels = sc_nc
        self.conv_layers = nn.Sequential(conv_layers)
        self.out_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=prev_num_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.CELU(),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1,),
        )

    def forward(self, x):
        return self.out_proj(self.conv_layers(x))


class TransformDecoder(nn.Module):
    def __init__(self, in_feat_dim, hidden_dim=128, out_proj_dim=9, act="elu"):
        super().__init__()
        if act == "elu":
            self.act = nn.ELU()
        dim1 = in_feat_dim // 2
        self.scale_center_encoder = nn.Linear(4, in_feat_dim)
        # self.center_encoder = nn.Linear(2, dim1)
        # self.interaction1 = NeuralTensor(in_feat_dim, in_feat_dim, in_feat_dim)
        self.interaction1 = nn.Bilinear(
            in_feat_dim, in_feat_dim, in_feat_dim, bias=True
        )
        self.interaction2 = NeuralTensor(in_feat_dim, in_feat_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_proj_dim)

    def forward(self, img_features, scale, center):
        scale_center = self.scale_center_encoder(torch.cat((scale, center), -1))
        scale_center = self.act(scale_center)
        int1 = self.interaction1(scale_center, scale_center)
        int1 = self.act(int1)
        int2 = self.interaction2(img_features, int1)
        int2 = self.act(int2)
        out = self.act(self.linear(int2))
        out = self.out_proj(out)

        return out


class SceneEncoder(nn.Module):
    def __init__(
        self,
        z_pres_dim=1,
        z_depth_dim=1,
        z_scale_dim=2,
        z_shift_dim=2,
        z_what_dim=32,
        img_feats_nc=128,
        glimpse_shape=(32, 32),
        anchor_shape=(48, 48),
        glimpse_enc_out_proj_dim=256,
    ):
        super().__init__()

        self.z_pres_dim = z_pres_dim
        # self.z_depth_dim = z_depth_dim
        self.z_scale_dim = z_scale_dim
        self.z_shift_dim = z_shift_dim
        glimpse_shape = glimpse_shape if glimpse_shape else cfg.GLIMPSE_SHAPE
        self.register_buffer("glimpse_shape", torch.tensor(glimpse_shape))
        self.register_buffer("anchor_shape", torch.tensor(anchor_shape))

        self.z_dim = (
            self.z_pres_dim
            # + self.z_depth_dim * 2
            + self.z_scale_dim * 2
            + self.z_shift_dim * 2
        )
        self.znet = nn.Conv2d(
            in_channels=img_feats_nc, out_channels=self.z_dim, kernel_size=(1, 1),
        )
        self.img_encoder = ImageEncoder()
        self.bg_encoder = nn.Sequential(
            nn.Conv2d(img_feats_nc, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64, affine=True),
            nn.MaxPool2d(2),
            nn.ELU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(num_features=64, affine=True),
            # nn.ELU(),
            nn.Conv2d(64, 64 + 9, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64 + 9, affine=True),
            nn.MaxPool2d(2),
            # nn.ELU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(num_features=64, affine=True), # spatial dim 1x1
            # nn.ELU(),
        )
        # self.glimpse_encoder = GlimpseEncoder(out_proj_dim=glimpse_enc_out_proj_dim)
        self.transform_decoder = TransformDecoder(64)

    def forward(self, img, presenceT=1):
        img_feats = self.img_encoder(img)
        bg_feats = self.bg_encoder(img_feats).squeeze(-1).squeeze(-1)
        bg_transform_params = process_decoded_transform(bg_feats[..., -9:])
        glimpses_info = self.get_glimpses(img, img_feats, presenceT)
        # glimpses = glimpses_info['glimpses']
        # what_mean, what_log_std = self.glimpse_encoder(glimpses).chunk(2, 1)
        # z_what = sample_gaussean_sp(what_mean, what_log_std)
        # Predict object transformation
        # rotation x,y,z, scale x,y,z translation x,y,z
        obj_pres = glimpses_info["obj_pres"].squeeze(-1)
        img_feats = img_feats.permute(0, 2, 3, 1).contiguous()
        scale, center = (
            glimpses_info["z_scale_wrt_img"],
            glimpses_info["z_center_wrt_img__11"],
        )
        valid_img_feats = select_and_pad_on_presence(img_feats, obj_pres)
        valid_scale = select_and_pad_on_presence(scale, obj_pres)
        valid_center = select_and_pad_on_presence(center, obj_pres)
        img_transforms = self.transform_decoder(
            valid_img_feats, valid_scale, valid_center
        )
        fg_transform_params = process_decoded_transform(img_transforms)

    def get_glimpses(self, img, img_feats, presenceT=1):
        bsz, C, H, W = img.size()
        image_shape = torch.tensor((H, W), dtype=torch.float, device=img_feats.device)

        Hp, Wp = img_feats.size()[-2:]
        fmap_shape = torch.tensor((Hp, Wp), dtype=torch.float, device=img_feats.device)
        ch, cw = image_shape / fmap_shape  #  pseudo-receptive field

        Z = self.znet(img_feats).permute(0, 2, 3, 1)
        (
            pres_p_logits,
            # depth_mean,
            # depth_log_std,
            scale_mean,
            scale_log_std,
            center_shift_mean,
            center_shift_log_std,
        ) = Z.split(
            [
                self.z_pres_dim,
                # *(self.z_depth_dim,) * 2,
                *(self.z_scale_dim,) * 2,
                *(self.z_shift_dim,) * 2,
            ],
            dim=-1,
        )
        # SAMPLE RANDOM VARIABLES
        # obj_pres is the sampled value and obj_pres_prob is
        obj_pres, obj_pres_prob = sample_bernoulli(
            pres_p_logits, hard=False, temperature=presenceT,
        )  # (bsz, Hp, Wp, 1), # (bsz, Hp, Wp, 1)
        # SPACE mentioned using softplus activation for std-dev
        # z_depth = sample_gaussean(
        #     depth_mean, nn.functional.softplus(depth_log_std)
        # )  # (bsz, Hp, Wp, 1)
        z_scale = sample_gaussean(
            scale_mean, nn.functional.softplus(scale_log_std)
        )  # (bsz, Hp, Wp, 2)
        z_shift = sample_gaussean(
            center_shift_mean, nn.functional.softplus(center_shift_log_std)
        )  # (bsz, Hp, Wp, 2)

        z_scale_wrt_anchor = torch.sigmoid(z_scale)  # Initial expected value of 0.5
        # z_scale_abs: size in pixels, according to SPAIR, using a constant anchor instead of a possible
        # variable image size is better
        z_scale_abs = z_scale_wrt_anchor * self.anchor_shape
        z_scale_wrt_img = (
            z_scale_abs / image_shape
        )  # size in range (0, GLIMPSE_SIZE / IMAGE_SIZE)

        # Initial expected value of 0.5, in the center of each fmap cell
        z_shift_wrt_fmap = torch.sigmoid(z_shift)
        z_shift_wrt_img = z_shift_wrt_fmap / fmap_shape
        # ij_grid: coordinate map with values (x, y) in ((0, 1), (0, 1)) where
        # 0 is the left/top-most position of the image and 1 the right/bottom-most
        # each coordinate denotes the top-left corner of the projected
        # pseudo-receptive field in the original image for each cell of the
        # feature map
        ij_grid = torch.stack(
            torch.meshgrid((torch.arange(0, 1, 1 / Hp), torch.arange(0, 1, 1 / Wp))),
            dim=-1,
        )
        z_center_wrt_img = ij_grid.unsqueeze(0) + z_shift_wrt_img
        # z_center_wrt_img__11: transformed values of centers relative
        # to the image to range (-1, 1) to use with torch.nn.functional.make_grid
        z_center_wrt_img__11 = (z_center_wrt_img * 2) - 1

        # transformation height and width
        # theta_h, theta_w = z_scale_wrt_img.view(-1, 2).split(1, -1)
        # # transformation translation
        # theta_tx, theta_ty = z_center_wrt_img__11.view(-1, 2).split(1, -1)
        # # transformation, zero values are crop skewness, for now without skew
        # theta = torch.cat(
        #     [
        #         theta_w,
        #         torch.zeros_like(theta_w),
        #         theta_ty,
        #         torch.zeros_like(theta_w),
        #         theta_h,
        #         theta_tx,
        #     ],
        #     dim=-1,
        # )

        # _theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(
        #     _theta, (_theta.size(0), C, *self.glimpse_shape), align_corners=False
        # )
        # # repeat each image Hp * Wp times for each cropping
        # _img = img.unsqueeze(1).expand(-1, Hp * Wp, -1, -1, -1)
        # _img = _img.view(-1, C, H, W)
        # xs = F.grid_sample(_img, grid, align_corners=False)

        return {
            # 'glimpses': xs,
            "pres_p_logits": pres_p_logits,
            # 'depth_mean': depth_mean,
            # 'depth_log_std': depth_log_std,
            "scale_mean": scale_mean,
            "scale_log_std": scale_log_std,
            "center_shift_mean": center_shift_mean,
            "center_shift_log_std": center_shift_log_std,
            "obj_pres": obj_pres,
            # 'z_depth': z_depth,
            "z_scale": z_scale,
            "z_shift": z_shift,
            "z_scale_wrt_img": z_scale_wrt_img,
            "z_center_wrt_img__11": z_center_wrt_img__11,
        }
