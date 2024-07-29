import os
import numpy as np
import imageio
from glob import glob
import cv2 as cv


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def get_all_initial_data_endo(path, data_type, depth_scale, is_mask, npy_file):
    poses_bounds = np.load(os.path.join(path, npy_file))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]
    H = H.astype(int)
    W = W.astype(int)

    video_path = sorted(glob(os.path.join(path, 'images/*')))
    if is_mask:
        masks_path = sorted(glob(os.path.join(path, 'gt_masks/*')))
        GT_masks_path = os.path.join(path, "gt_masks")

        inpaint_mask_all = np.zeros((512, 640))

        for i in range(0, len(masks_path)):
            img_name = masks_path[i]
            f_mask = os.path.join(GT_masks_path, img_name)
            m = 1.0 - np.array(imread(f_mask) / 255.0)
            inpaint_mask_all = inpaint_mask_all + m
            inpaint_mask_all[inpaint_mask_all >= 1] = 1

        inpaint_mask_all = (1.0 - inpaint_mask_all) * 255.0
        inpaint_mask_all = inpaint_mask_all.astype(np.uint8)

        fn = os.path.join(path, f"invisible_mask.png")
        imageio.imwrite(fn, inpaint_mask_all)
        kernel = np.ones((5, 5), np.uint8)

        dilated_mask = cv.dilate(inpaint_mask_all, kernel, iterations=2)
        if data_type == 'endonerf':
            dilated_mask[-12:, :] = 255
        fn = os.path.join(path, f"dilated_invisible_mask.png")
        imageio.imwrite(fn, dilated_mask)

    depths_path = sorted(glob(os.path.join(path, 'depth/*')))

    print(f"Using the all depth map as the initial point cloud")
    print(f"Using the depth_scale:{depth_scale} scale the depth map")

    depth_all = np.zeros((H, W))
    color_all = np.zeros((3, H, W))
    mask_all = np.zeros((H, W))
    inv_mask_all = np.ones((H, W))
    for i in range(poses_bounds.shape[0]):
        images_path = video_path[i]
        image = np.array(imread(images_path) * 1.0)
        color = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        mask_image = None
        if is_mask:
            mask_path = masks_path[i]
            mask_image = np.array(imread(mask_path) / 255.0)
            # mask is 0 or 1
            mask_image = np.where(mask_image > 0.5, 1.0, 0.0)
            # Convert 0 for tool, 1 for not tool
            mask_image = 1.0 - mask_image

        color_mask = np.expand_dims(mask_image, axis=0)
        color = color * color_mask

        depth_path = depths_path[i]
        depth_image = np.array(imread(depth_path) * 1.0)
        depth_image = depth_image / depth_scale
        near = np.percentile(depth_image, 3)
        far = np.percentile(depth_image, 98)
        mask_depth = np.bitwise_and(depth_image > near, depth_image < far)
        mask_depth = mask_depth * mask_image
        depth = depth_image * mask_depth
        mask_plus = mask_depth * inv_mask_all
        color_all_mask = np.expand_dims(mask_plus, axis=0)
        depth_all = depth_all + depth * mask_plus
        color_all = color_all + color * color_all_mask

        mask_all = mask_all + mask_depth
        mask_all[mask_all >= 1] = 1
        inv_mask_all = 1.0 - mask_all

    depth_all = np.expand_dims(depth_all, axis=0)

    intrinsics = np.zeros((3, 3))
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = W / 2.0  # CX: W/2
    intrinsics[1, 2] = H / 2.0  # CY: H/2
    intrinsics[2, 2] = 1.0


    return color_all, depth_all, intrinsics, mask_all

def get_pointcloud(color, depth, intrinsics, mask, w2c=None, transform_pts=False):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = np.meshgrid(np.arange(width).astype(np.float32),
                                    np.arange(height).astype(np.float32),
                                    indexing='xy')
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)  #
    yy = yy.reshape(-1)  #
    depth_z = depth[0].reshape(-1)  #

    # Initialize point cloud
    pts_cam = np.stack((xx * depth_z, yy * depth_z, depth_z), axis=-1)

    if transform_pts:
        pix_ones = np.ones(height * width, 1).astype(np.float32)
        pts4 = np.concatenate((pts_cam, pix_ones), axis=1)
        c2w = np.linalg.inv(w2c)
        pts = np.dot(pts4, c2w.T)[:, :3]
    else:
        pts = pts_cam

    # Colorize point cloud
    cols = np.transpose(color, (1, 2, 0)).reshape(-1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    mask_sample = sample_pts(height, width, 3)
    mask_sample = (mask_sample != 0)
    mask_sample = mask_sample.reshape(-1)

    mask = mask.reshape(-1)
    mask = (mask != 0)
    pts = pts[mask & mask_sample]

    cols = cols[mask & mask_sample]
    print(f"Using the {pts.shape[0]} points as initial")

    return pts, cols

def sample_pts(height, width, factor=2):
    mask_sample_h = np.zeros((height, width)).astype(np.int)
    mask_sample_w = np.zeros((height, width)).astype(np.int)
    mask_sample_h[:, 1::factor] = 1
    mask_sample_w[1::factor, :] = 1
    mask_sample = mask_sample_h & mask_sample_w

    return mask_sample
