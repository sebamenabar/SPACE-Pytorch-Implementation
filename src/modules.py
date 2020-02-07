import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg
from sample_utils import sample_gaussean, sample_bernoulli


class ImgEncoder(nn.Module):
    def __init__(self, num_channels=[3, 64, 64, 64, 64], out_proj_dim=64):
        super().__init__()
        
        self.num_channels = num_channels
        self.elu = nn.ELU()
        
        conv_layers = []
        for i in range(len(self.num_channels) - 1):
            conv_layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=self.num_channels[i],
                    out_channels=self.num_channels[i + 1],
                    kernel_size=(3, 3),
                    stride=2,
                ),
                nn.BatchNorm2d(num_features=self.num_channels[i + 1]),
                self.elu,
            ))
        self.conv_layers = nn.Sequential(*conv_layers)
        self.out_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_channels[-1],
                out_channels=out_proj_dim,
                kernel_size=(1, 1),
            ),
            # nn.Linear(in_features=self.num_channels[-1], out_features=out_proj_dim),
            self.elu,
        )
        
    def forward(self, x):
        # return self.out_proj(self.conv_layers(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.out_proj(self.conv_layers(x))

class FG(nn.Module):
    def __init__(self,
                 z_pres_dim=1,
                 z_depth_dim=1,
                 z_scale_dim=2,
                 z_shift_dim=2,
                 z_what_dim=32,
                 in_channels=64,
                 glimpse_shape=None,
                ):
        super().__init__()
        
        self.z_pres_dim = z_pres_dim
        self.z_depth_dim = z_depth_dim
        self.z_scale_dim = z_scale_dim
        self.z_shift_dim = z_shift_dim
        glimpse_shape = glimpse_shape if glimpse_shape else cfg.GLIMPSE_SHAPE
        self.register_buffer('glimpse_shape', torch.tensor(glimpse_shape))
        
        self.z_dim = self.z_pres_dim + self.z_depth_dim * 2 + self.z_scale_dim * 2 + self.z_shift_dim * 2
        self.znet = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.z_dim,
            kernel_size=(1, 1),
        )
        self.encoder = ImgEncoder()

    def forward(self, img):
        xs = self.get_glimpses(img)

    def get_glimpses(self, img):
        bsz, C, H, W = img.size()
        image_shape = torch.tensor((H, W), dtype=torch.float, device=img.device)
        
        img_feats = self.encoder(img)
        Hp, Wp = img_feats.size()[-2:]
        fmap_shape = torch.tensor((Hp, Wp), dtype=torch.float, device=img.device)
        ch, cw = image_shape / fmap_shape #  pseudo-receptive field

        Z = self.znet(img_feats).permute(0, 2, 3, 1)
        (pres_p_logits, depth_mean, depth_log_std, scale_mean, 
            scale_log_std, center_shift_mean, center_shift_log_std) = \
            Z.split(
                [
                    self.z_pres_dim,
                    *(self.z_depth_dim,)* 2,
                    *(self.z_scale_dim,) * 2,
                    *(self.z_shift_dim,) * 2,
                ],
                dim=-1,
            )
        # SAMPLE RANDOM VARIABLES
        # obj_pres is the sampled value and obj_pres_prob is 
        obj_pres, obj_pres_prob = sample_bernoulli(pres_p_logits, hard=True, temperature=1) # (bsz, Hp, Wp, 1), # (bsz, Hp, Wp, 1)
        # SPACE mentioned using softplus activation for std-dev
        z_depth = sample_gaussean(depth_mean, nn.functional.softplus(depth_log_std)) # (bsz, Hp, Wp, 1)
        z_scale = sample_gaussean(scale_mean, nn.functional.softplus(scale_log_std)) # (bsz, Hp, Wp, 2)
        z_shift = sample_gaussean(center_shift_mean, nn.functional.softplus(center_shift_log_std)) # (bsz, Hp, Wp, 2)

        z_scale_wrt_anchor = torch.sigmoid(z_scale) # Initial expected value of 0.5
        # z_scale_abs: size in pixels, according to SPAIR, using a constant anchor instead of a possible
        # variable image size is better
        z_scale_abs = z_scale_wrt_anchor * self.glimpse_shape
        z_scale_wrt_img = z_scale_abs / image_shape # size in range (0, GLIMPSE_SIZE / IMAGE_SIZE)

        # Initial expected value of 0.5, in the center of each fmap cell
        z_shift_wrt_fmap = torch.sigmoid(z_shift)
        z_shift_wrt_img = z_shift_wrt_fmap / fmap_shape
        # ij_grid: coordinate map with values (x, y) in ((0, 1), (0, 1)) where 
        # 0 is the left/top-most position of the image and 1 the right/bottom-most
        # each coordinate denotes the top-left corner of the projected
        # pseudo-receptive field in the original image for each cell of the
        # feature map
        ij_grid = torch.stack(torch.meshgrid((torch.arange(0, 1, 1 / Hp), torch.arange(0, 1, 1 / Wp))), dim=-1)
        z_center_wrt_img = (ij_grid.unsqueeze(0) + z_shift_wrt_img)
        # z_center_wrt_img__11: transformed values of centers relative 
        # to the image to range (-1, 1) to use with torch.nn.functional.make_grid
        z_center_wrt_img__11 = (z_center_wrt_img * 2) - 1

        # transformation height and width
        theta_h, theta_w = z_scale_wrt_img.view(-1, 2).split(1, -1)
        # transformation translation
        theta_tx, theta_ty = z_center_wrt_img__11.view(-1, 2).split(1, -1)
        # transformation, zero values are crop skewness, for now without skew
        theta = torch.cat([theta_w, torch.zeros_like(theta_w), theta_ty, torch.zeros_like(theta_w), theta_h, theta_tx], dim=-1)

        _theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(_theta, (_theta.size(0), C, *self.glimpse_shape), align_corners=False)
        # repeat each image Hp * Wp times for each cropping
        _img = img.unsqueeze(1).expand(-1, Hp * Wp, -1, -1, -1)
        _img = _img.view(-1, C, H, W)
        xs = F.grid_sample(_img, grid, align_corners=False)

        return xs