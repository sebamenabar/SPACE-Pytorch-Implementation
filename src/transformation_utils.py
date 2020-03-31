import torch
from torch.nn import function as F


def inverse_affine(theta, image, out_shape):
    bsz = theta.size(0)
    t = torch.tensor([0., 0., 1.]).repeat(bsz, 1, 1).to(theta.device)
    t = torch.cat([theta, t], dim=-2)
    t = t.inverse()
    theta = t[:, :2, :]
    grid = F.affine_grid(theta, (bsz, image.size(1), *out_shape), align_corners=False)
    input_glimpses = F.grid_sample(image, grid, align_corners=False)
    return input_glimpses
