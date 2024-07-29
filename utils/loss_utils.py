#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pytorch3d.ops.knn import knn_points


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def tv_loss(x):
    # K-Plane
    tv_h = torch.abs(x[:,1:,:] - x[:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,1:] - x[:,:,:-1]).sum()
    return (tv_h + tv_w) * 2 / x.numel()



def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def def_reg_loss(gs_can, d_xyz, d_rotation, d_scaling, K=5):
    xyz_can = gs_can.get_xyz
    xyz_obs = xyz_can + d_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_can.get_covariance_obs(d_rotation, d_scaling)

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0), xyz_can.unsqueeze(0), K=K, return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    dis_xyz_can = torch.cdist(xyz_can.unsqueeze(1), xyz_can[nn_ix])[:, 0, 1:]
    dis_xyz_obs = torch.cdist(xyz_obs.unsqueeze(1), xyz_obs[nn_ix])[:, 0, 1:]
    loss_pos = F.l1_loss(dis_xyz_can, dis_xyz_obs)

    dis_cov_can = torch.cdist(cov_can.unsqueeze(1), cov_can[nn_ix])[:, 0, 1:]
    dis_cov_obs = torch.cdist(cov_obs.unsqueeze(1), cov_obs[nn_ix])[:, 0, 1:]
    loss_cov = F.l1_loss(dis_cov_can, dis_cov_obs)

    return loss_pos, loss_cov
