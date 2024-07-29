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

import os
import sys
import torch
from PIL import Image
from typing import NamedTuple, Optional

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np

import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

from utils.initial_utils import get_pointcloud, get_all_initial_data_endo


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    mask_depth: Optional[np.array] = None
    mask: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose

def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readCamerasdavinci(path, data_type, is_depth, depth_scale, is_mask, npy_file, split, hold_id, num_images):
    cam_infos = []
    poses_bounds = np.load(os.path.join(path, npy_file))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]
    video_path = sorted(glob(os.path.join(path, 'images/*')))
    if is_mask:
        masks_path = sorted(glob(os.path.join(path, 'gt_masks/*')))
    if is_depth:
        depths_path = sorted(glob(os.path.join(path, 'depth/*')))
        bds = poses_bounds[:, -2:]
        close_depth, inf_depth = np.ndarray.min(bds), np.ndarray.max(bds)

    n_cameras = poses.shape[0]
    poses = poses[:, :, :4]
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        c2w = poses[i]
        images_path = video_path[i]
        image_name = Path(images_path).stem
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        image = Image.open(images_path)

        mask_image = None
        if is_mask:
            mask_path = masks_path[i]
            mask_image = np.array(imread(mask_path) / 255.0)
            # mask is 0 or 1
            mask_image = np.where(mask_image > 0.5, 1.0, 0.0)
            # Convert 0 for tool, 1 for not tool
            mask_image = 1.0 - mask_image
            if data_type == 'endonerf':
                mask_image[-12:, :] = 0

        depth_image = None
        mask_depth = None
        if is_depth:
            depth_path = depths_path[i]
            depth_image = np.array(imread(depth_path) * 1.0)
            depth_image = depth_image / depth_scale
            near = np.percentile(depth_image, 3)
            far = np.percentile(depth_image, 98)
            mask_depth = np.bitwise_and(depth_image > near, depth_image < far)
            if is_mask:
                mask_depth = mask_depth * mask_image
            depth_image = depth_image * mask_depth

        frame_time = i / (n_frames - 1)
        FovX = focal2fov(focal, image.size[0])
        FovY = focal2fov(focal, image.size[1])
        cam_infos.append(CameraInfo(uid=i, R=R, T=T, FovX=FovX, FovY=FovY,
                                    image=image,
                                    image_path=images_path, image_name=image_name,
                                    width=image.size[0], height=image.size[1], fid=frame_time, depth=depth_image, mask_depth=mask_depth, mask=mask_image))


    return cam_infos


def readEndonerfInfo(path, data_type, eval, is_depth, depth_scale, is_mask, depth_initial, num_images, hold_id):  # hold_id选择test的帧数ID,这个是endosurf的测试集
    print("Reading Training Camera")
    train_cam_infos = readCamerasdavinci(
        path, data_type, is_depth, depth_scale, is_mask, 'poses_bounds.npy', split="train", hold_id=hold_id, num_images=num_images)

    print("Reading Test Camera")
    test_cam_infos = readCamerasdavinci(
        path, data_type, is_depth, depth_scale, is_mask, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization["radius"] = 1

    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        if depth_initial:
            color, depth, intrinsics, mask = get_all_initial_data_endo(path, data_type, depth_scale, is_mask, 'poses_bounds.npy')

            xyz, RGB = get_pointcloud(color, depth, intrinsics, mask)
            storePly(ply_path, xyz, RGB)

        else:
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Endonerf": readEndonerfInfo,
}
