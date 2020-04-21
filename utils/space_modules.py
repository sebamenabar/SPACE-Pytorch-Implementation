from collections import OrderedDict as odict

import torch
import torch.nn as nn


class ComponentDecoder(nn.Module):
    def __init__(
        self, in_nc=64, kss=[3, 3, 3], ncs=[64, 32, 32], ss=[1, 1, 1], act="elu"
    ):
        super().__init__()
        # if act == "elu":
        #     act = nn.ELU

        in_nc = in_nc + 2
        conv_layers = []
        for ks, nc, s in zip(kss, ncs, ss):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        kernel_size=ks,
                        in_channels=in_nc,
                        out_channels=nc,
                        stride=s,
                        padding=1,
                    ),
                    nn.BatchNorm2d(nc),
                    nn.LeakyReLU(0.2),
                )
            )
            in_nc = nc
        self.conv_layers = nn.Sequential(*conv_layers)

        self.out_proj = nn.Conv2d(
            kernel_size=3, in_channels=in_nc, out_channels=3, stride=1, padding=1
        )
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def forward(self, inp, shape=(128, 128)):
        x = torch.linspace(-1, 1, shape[0])
        y = torch.linspace(-1, 1, shape[1])
        x, y = torch.meshgrid(x, y)

        x = x.view(1, 1, *x.size()).expand(inp.size(0), -1, -1, -1).to(inp.device)
        y = y.view(1, 1, *y.size()).expand(inp.size(0), -1, -1, -1).to(inp.device)

        _inp = inp.view(*inp.size(), 1, 1).expand(-1, -1, shape[0], shape[1])
        broadcasted_bg = torch.cat((_inp, x, y,), 1)
        return self.out_proj(self.conv_layers(broadcasted_bg))


class GlimpseEncoder(nn.Module):
    def __init__(
        self,
        ncs=[16, 32, 64, 64, 128, 128, 128],
        kss=[3, 4, 3, 4, 3, 4, 4],
        ss=[1, 2, 1, 2, 1, 2, 1],
        gs=[4, 8, 8, 8, 4, 8, 16, 8],
        pds=[1, 1, 1, 1, 1, 1, 0],
        out_proj_dim=128,
        act="relu",
    ):
        super().__init__()

        conv_layers = []
        prev_nc = 3
        for nc, ks, s, pd, ng in zip(ncs, kss, ss, pds, gs):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=prev_nc,
                        out_channels=nc,
                        kernel_size=ks,
                        stride=s,
                        padding=pd,
                    ),
                    nn.GroupNorm(num_groups=ng, num_channels=nc),
                    nn.LeakyReLU(0.2),
                )
            )
            prev_nc = nc
        self.conv_layers = nn.Sequential(*conv_layers)
        self.out_proj = nn.Conv2d(
            in_channels=prev_nc, out_channels=out_proj_dim, kernel_size=1,
        )
        # self.out_pool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        return self.out_proj(self.conv_layers(x))


class GlimpseDecoder(nn.Module):
    def __init__(
        self,
        in_nc=128,
        out_proj_nc=16,
        conv_ncs=[128, 128, 64, 64, 32],
        conv_ss=[1, 1, 1, 1, 1, 1],
        conv_gns=[16, 16, 16, 8, 8],
        conv_kss=[1, 3, 3, 3, 3],
        conv_pds=[0, 1, 1, 1, 1],
        sub_conv_factors=[2, 2, 2, 2, 2],
        sub_conv_ncs=[128, 64, 64, 32, 16],
        sub_conv_ss=[1, 1, 1, 1, 1],
        sub_conv_gns=[16, 16, 8, 8, 4],
        out_proj_dim=16,
        out_proj_ks=3,
        out_proj_stride=1,
        act="celu",
    ):
        super().__init__()

        act = nn.LeakyReLU(0.2)
        conv_layers = odict([])
        prev_num_channels = in_nc
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
                odict(
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
                                nn.LeakyReLU(0.2),
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
                                nn.LeakyReLU(0.2),
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
                out_channels=out_proj_nc,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.GroupNorm(num_groups=out_proj_nc // 4, num_channels=out_proj_nc),
            nn.InstanceNorm2d(num_features=out_proj_nc),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=out_proj_nc, out_channels=4, kernel_size=1,),
        )
        self.apply(self.weights_init)
        self.out_proj[-1].weight.data.normal_()

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.out_proj(self.conv_layers(x))
