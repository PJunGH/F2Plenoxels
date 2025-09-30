# Copyright 2021 Alex Yu
# Eval

import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, lin_norm, viridis_cmap, Rays, add_text_overlay
from util import config_util

import imageio
import cv2
from tqdm import tqdm
from PIL import Image

import open3d as o3d
import time
import pyexr
import matplotlib.pyplot as plt
import json
from matplotlib.transforms import Affine2D

# np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--use_gpu', action='store_true', default=True)
parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--n_train', type=int)
parser.add_argument('--hardcode_train_views', type=int, nargs='+', help='hardcode training views')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--timing',
                    action='store_true',
                    default=False,
                    help="Run only for timing (do not save images or use LPIPS/SSIM; "
                    "still computes PSNR to make sure images are being generated)")
parser.add_argument('--no_lpips',
                    action='store_true',
                    default=False,
                    help="Disable LPIPS (faster load)")
parser.add_argument('--no_vid',
                    action='store_true',
                    default=False,
                    help="Disable video generation")
parser.add_argument('--no_imsave',
                    action='store_true',
                    default=False,
                    help="Disable image saving (can still save video; MUCH faster)")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")

parser.add_argument('--export_vol',
                    action='store_true')
parser.add_argument('--vis_grid',
                    action='store_true',
                    help='visualize voxel grid')
parser.add_argument('--radius',
                    type=float,
                    nargs='+',
                    default=[1.0,1.0,1.0],
                    help='radius of voxel grid')
parser.add_argument('--center',
                    type=float,
                    nargs='+',
                    default=[0.0,0.0,0.0],
                    help=('center of voxel grid'))
parser.add_argument('--dir_scale',
                    type=float,
                    default=1.0,
                    help='scale factor for direction')
parser.add_argument('--prune', 
                    action='store_true',
                    default=False)
parser.add_argument('--sigma_threshold',
                    type=float,
                    default=5.0) 
parser.add_argument('--vis_slice',
                    action='store_true',
                    help='visualize voxel grid slice')

parser.add_argument('--vis_train_ray',
                    action='store_true',
                    default=False)
parser.add_argument('--train_ray_id',
                    type=str,
                    default="[0,0,0]",
                    help='ids for image, height and width respectively to specify ray id')
parser.add_argument('--vis_test_ray',
                    action='store_true',
                    default=False)
parser.add_argument('--test_ray_id',
                    type=str,
                    default="[0,0,0]",
                    help='ids for image, height and width respectively to specify ray id')
# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
parser.add_argument('--use_kernel',
                    action='store_true',
                    default=True,
                    help="use cuda kernel")
parser.add_argument('--ray_len',
                    action='store_true',
                    default=False,
                    help="Render the ray lengths")

parser.add_argument('--add_noise',
                    action='store_true',
                    default=False,
                    help="Add noise to pred")

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = f'cuda:{args.gpu_id}' if args.use_gpu else 'cpu'

if args.timing:
    args.no_lpips = True
    args.no_vid = True
    args.ray_len = False

if not args.no_lpips:
    import lpips
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

render_dir = path.join(path.dirname(args.ckpt),
             f'train_renders{args.n_train}' \
             if args.train and args.n_train != 0 and args.n_train != None else 'train_renders' \
             if args.train else 'test_renders')
want_metrics = True
if args.render_path:
    assert not args.train
    render_dir += '_path'
    want_metrics = False
ckpt_name = path.basename(args.ckpt).split(".")[0].split("_")
if len(ckpt_name) > 1:
    render_dir += '_' + ckpt_name[-1]

# Handle various image transforms
if not args.render_path:
    # Do not crop if not render_path
    args.crop = 1.0
if args.crop != 1.0:
    render_dir += f'_crop{args.crop}'
if not args.use_kernel:
    render_dir +=f'_nousekernel'
if args.ray_len:
    render_dir += f'_raylen'
    want_metrics = False

dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                   depth_source=args.depth_source, use_vggt_intri=args.use_vggt_intri,
                                   hardcode_train_views=args.hardcode_train_views if args.train else None,
                                    n_images = args.n_train if args.train and args.n_train != 0 else None, 
                                    **config_util.build_data_options(args))
if dset.split == "test_train":
    train_dset = dset
    train_dset.gen_rays()
else:
    train_dset = datasets[args.dataset_type](args.data_dir, split="train", 
                                             n_images = args.n_train if args.n_train != 0 else None,
                                             **config_util.build_data_options(args))

grid = svox2.SparseGrid.load(args.ckpt, device=device)

if args.export_vol:
    vol = torch.ones(grid.links.shape).to(grid.density_data.data) * grid.density_data.data.min()
    vol[grid.links >= 0] = grid.density_data.data[grid.links[grid.links >= 0].to(torch.long)].squeeze()
    import mrc
    mrc.imsave(path.join(path.dirname(args.ckpt), 'sigma_vol.mrc'), vol.cpu().numpy())

'''
if args.vis_grid:
    voxel_grid = o3d.geometry.VoxelGrid()  
    voxel_grid.voxel_size = 0.5
    voxel_grid.origin = np.array([0.,0.,0.])
    M, N, K = grid.links.shape
    X = np.arange(M)
    Y = np.arange(N)
    Z = np.arange(K)
    X, Y, Z = np.meshgrid(X,Y,Z,indexing='ij')
    idxs = np.stack([X,Y,Z], axis=-1).reshape(-1,3)
    mask = (grid.links >= 0).cpu().numpy().reshape(-1)
    idxs = idxs[mask]
    links = grid.links.reshape(-1)[mask]
    if args.prune:
        mask = (grid.density_data.data[links.to(torch.long)] >= args.sigma_threshold).squeeze().cpu().numpy()
        idxs = idxs[mask]
    start = time.time()
    color = np.random.uniform(0, 1, size=(3))
    for idx in idxs:
        # ptr = grid.links[idx[0],idx[1],idx[2]]
        # if grid.density_data.data[ptr] >= -0.05:
            # voxel = o3d.geometry.Voxel(idx, np.random.uniform(0, 1, size=(3)))
        voxel = o3d.geometry.Voxel(idx, np.random.uniform(0, 1, size=(3)))
        voxel_grid.add_voxel(voxel)
        # else:
        # if grid.density_data.data[ptr] <= -0.15:
            # voxel = o3d.geometry.Voxel(idx, [0.,0.5,0.])
            # voxel_grid.add_voxel(voxel)
            
    elapse = time.time() - start
    print(elapse)
    if args.prune:
        save_dir = path.join(path.dirname(args.ckpt), f'voxel_grid_pruning{args.sigma_threshold}.ply')
    else:
        save_dir = path.join(path.dirname(args.ckpt), 'voxel_grid.ply')
    o3d.io.write_voxel_grid(save_dir ,voxel_grid)
    o3d.visualization.draw_geometries([voxel_grid], top=150)
'''

if args.vis_grid:

    def world2grid(points):
        center = torch.asarray(args.center, device='cpu')
        radius = torch.asarray(args.radius, device='cpu')
        _offset = 0.5 * (1.0 - center / radius)
        _scaling = 0.5 / radius
        gsz = grid._grid_size()
        offset = _offset * gsz - 0.5
        scaling = _scaling * gsz
        return torch.addcmul(
            offset.to(device=points.device), points, scaling.to(device=points.device)
        )
        
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False
    
    # cameras
    N, H, W = train_dset.n_images, train_dset.w, train_dset.h
    origins = train_dset.rays.origins.reshape(N, H, W, -1)
    dirs = train_dset.rays.dirs.reshape(N, H, W, -1)
    cam_lst = []
    for i_img in range(N):
        p0 = origins[i_img, 0, 0]
        p1 = p0 + dirs[i_img, 0, 0]*args.dir_scale
        p2 = p0 + dirs[i_img, 0, W-1]*args.dir_scale
        p3 = p0 + dirs[i_img, H-1, W-1]*args.dir_scale
        p4 = p0 + dirs[i_img, H-1, 0]*args.dir_scale
        points = torch.stack([p0, p1, p2, p3, p4], dim=0)
        cam_lst.append(world2grid(points).cpu().numpy())
    cam_frustrm_lst = []
    for cam in cam_lst:
        cam_frustrm = o3d.geometry.LineSet()
        cam_frustrm.points = o3d.utility.Vector3dVector(cam)
        cam_frustrm.colors = o3d.utility.Vector3dVector([[0,0,0] for i in range(8)])
        cam_frustrm.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]])
        cam_frustrm_lst.append(cam_frustrm)
    # bounding box
    aabb_01 = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0]])
    out_bbox = o3d.geometry.LineSet()
    L, W, H = grid.links.shape[0], grid.links.shape[1], grid.links.shape[2]
    out_bbox.points = o3d.utility.Vector3dVector([0,0,0] + aabb_01 * [L, W, H])
    out_bbox.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
    # voxel grid
    if args.prune:
        density_grid = torch.zeros_like(grid.links, dtype=grid.density_data.data.dtype)
        ptrs = grid.links >= 0
        density_grid[ptrs] = grid.density_data.data[grid.links[ptrs].to(torch.long)].squeeze()
        IJK = np.stack((density_grid > args.sigma_threshold).cpu().numpy().nonzero(), -1)
    else:
        IJK = np.stack((grid.links >= 0).cpu().numpy().nonzero(), -1)
    sh_dim = grid.sh_data.data.size(-1) // 3
    color = grid.sh_data.data[grid.links[IJK[:,0], IJK[:,1], IJK[:,2]].to(torch.long)][:,::sh_dim].cpu().numpy()
    color = np.clip(color*svox2.utils.SH_C0+0.5, 0.0, 1.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(IJK)
    pcd.colors = o3d.utility.Vector3dVector(color)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    # o3d.visualization.draw_geometries([voxel_grid], top=150)
    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=int(min(L,W,H)*0.1), origin=[0,0,0]),
        out_bbox, pcd, #*cam_frustrm_lst
    ], window_name=path.dirname(args.ckpt), top=150)
    # exit(0)

if args.vis_slice:
    reso = grid.links.shape
    density_grid = torch.zeros(reso, device=device)
    mask = grid.links >= 0
    density_grid[mask] = grid.density_data.data.squeeze()[grid.links[mask].to(torch.long)]
    # density_grid[density_grid < 5] = -2000. 
    density_slice = density_grid[128,:,:]
    fig, ax = plt.subplots()
    im = ax.imshow(torch.rot90(density_slice,1,[0,1]).cpu().numpy())
    fig.colorbar(im, ax=ax)
    # trans = Affine2D().rotate_deg(-90)
    # ax.transData = ax.transData + trans
    plt.show()
    exit()

@torch.no_grad()
def vis_ray(dset, ray_id, grid):
    img_id, h_id, w_id = ray_id
    dset.gen_rays()
    origins = dset.rays.origins.reshape(dset.n_images, dset.h, dset.w, -1)
    dirs = dset.rays.dirs.reshape(dset.n_images, dset.h, dset.w, -1)                                 
    gt = dset.rays.gt.reshape(dset.n_images, dset.h, dset.w, -1)
    img = (gt[img_id]*255).numpy().clip(0,255).astype(np.uint8)
    # cv2.rectangle(img, [w_id-10, h_id-10], [w_id+10, h_id+10], [255, 0, 0], 2, 4)
    cv2.circle(img, [w_id, h_id], 5, [255, 0, 0], -1, 0)
    cv2.putText(img, f"[{h_id}, {w_id}]", (w_id+12, h_id+4), 0, 0.5, [255, 0, 0])
    # imageio.imwrite('.\\test.png', img)
    if dset.rays.depths != None:
        depths = dset.rays.depths.reshape(dset.n_images, dset.h, dset.w, -1)
    origin = origins[img_id, h_id, w_id].unsqueeze(0)
    dir = dirs[img_id, h_id, w_id].unsqueeze(0)
    color = gt[img_id, h_id, w_id].unsqueeze(0)
    # print(f'The color of ray is {gt[img_id, h_id, w_id].numpy().tolist()}')
    if dset.rays.depths != None:
        depth_a = depths[img_id, h_id, w_id].item()
        # print(f'The depth of ray is {depth_a:.5f}')
    else:
        depth_a = None
    ray = Rays(origins=origin, dirs=dir, gt=color).to(device)

    origin = grid.world2grid(ray.origins)
    dir = ray.dirs / torch.norm(ray.dirs, dim=-1, keepdim=True)
    viewdir = dir
    gsz = torch.tensor(grid.links.shape).to(device)
    _scaling = (grid._scaling.to(device) * gsz)
    dir = dir * _scaling
    delta_scale = 1.0 / dir.norm(dim=1)
    dir *= delta_scale.unsqueeze(-1)

    sh_mult = svox2.utils.eval_sh_bases(grid.basis_dim, viewdir)
    invdir = 1.0 / dir

    t1 = (-0.5 - origin) * invdir
    t2 = (gsz - 0.5 - origin) * invdir

    t = torch.min(t1, t2)
    t[dir == 0] = -1e9
    t = torch.max(t, dim=-1).values.clamp_min_(grid.opt.near_clip)

    tmax = torch.max(t1, t2)
    tmax[dir == 0] = 1e9
    tmax = torch.min(tmax, dim=-1).values

    log_light_intensity = torch.zeros(1, device=origin.device)
    out_depth = torch.zeros(1, device=origin.device)
    out_rgb = torch.zeros((1, 3), device=origin.device)

    t_x_grid = []
    t_x = []
    sigma_y = []
    alpha_y = []
    transmission_y = []
    weight_y = []
    sh_coeff_y = []
    rgb_comp_y = []
    rgb_y = []

    while t <= tmax:
        pos = origin + t[:, None] * dir
        pos = pos.clamp_min_(0.0)
        pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0] - 1)
        pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1] - 1)
        pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2] - 1)

        l = pos.to(torch.long)
        l.clamp_min_(0)
        l[:, 0] = torch.clamp_max(l[:, 0], gsz[0] - 2)
        l[:, 1] = torch.clamp_max(l[:, 1], gsz[1] - 2)
        l[:, 2] = torch.clamp_max(l[:, 2], gsz[2] - 2)
        pos -= l

        lx, ly, lz = l.unbind(-1)
        links000 = grid.links[lx, ly, lz]
        links001 = grid.links[lx, ly, lz + 1]
        links010 = grid.links[lx, ly + 1, lz]
        links011 = grid.links[lx, ly + 1, lz + 1]
        links100 = grid.links[lx + 1, ly, lz]
        links101 = grid.links[lx + 1, ly, lz + 1]
        links110 = grid.links[lx + 1, ly + 1, lz]
        links111 = grid.links[lx + 1, ly + 1, lz + 1]

        sigma000, rgb000 = grid._fetch_links(links000)
        sigma001, rgb001 = grid._fetch_links(links001)
        sigma010, rgb010 = grid._fetch_links(links010)
        sigma011, rgb011 = grid._fetch_links(links011)
        sigma100, rgb100 = grid._fetch_links(links100)
        sigma101, rgb101 = grid._fetch_links(links101)
        sigma110, rgb110 = grid._fetch_links(links110)
        sigma111, rgb111 = grid._fetch_links(links111)

        wa, wb = 1.0 - pos, pos
        c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
        c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
        c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
        c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
        c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
        c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
        sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

        c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
        c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
        c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
        c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
        c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
        c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
        rgb_c = c0 * wa[:, :1] + c1 * wb[:, :1]

        log_att = (
            -grid.opt.step_size
            * torch.relu(sigma[..., 0])
            * delta_scale[0]
        )
        alpha = 1.0 - torch.exp(log_att)
        transmission = torch.exp(log_light_intensity[0])
        weight = transmission * alpha

        sh_coeff = rgb_c.reshape(-1, 3, grid.basis_dim)
        rgb_comp = sh_mult.unsqueeze(-2) * sh_coeff
        rgb = torch.clamp_min(
            torch.sum(rgb_comp, dim=-1) + 0.5,
            0.0
        ) # [1, 3]
        w_rgb = weight * rgb[0, :3]
        out_rgb[0] += w_rgb

        depth = t * torch.norm(dir / _scaling[None, :], dim=-1)
        w_depth = weight * depth
        out_depth += w_depth

        log_light_intensity += log_att

        # t_x.append(t.item())
        t_x_grid.append(t.item())
        t_x.append(depth.item())
        sigma_y.append(sigma.item())
        alpha_y.append(alpha.item())
        transmission_y.append(transmission.item())
        weight_y.append(weight.item())
        sh_coeff_y.append(sh_coeff.squeeze(dim=0).cpu().numpy().tolist())
        rgb_comp_y.append(rgb_comp.squeeze(dim=0).cpu().numpy().tolist())
        rgb_y.append(rgb[0].cpu().numpy().tolist())

        t += grid.opt.step_size

    # print(f'The depth of ray is {out_depth.item()}')
    

    split="train" if args.train else "test"
    paths = path.dirname(args.ckpt).split('/')
    figname = f'[{paths[2]}][{paths[3]}][{split}_({ray_id[0]},{ray_id[1]},{ray_id[2]})]'
    fig, ax = plt.subplots(4, 2, sharex='col', sharey='row', num=figname+'_sigma')
    fig.suptitle(f'depth_a={depth_a:.5f}\ndepth={out_depth.item():.5f}')
    ax[0,0].plot(t_x_grid, sigma_y)
    # ax[0,0].set_title(f'depth={out_depth.item():.5f}')
    # ax[0,0].set_xlabel('z_grid')
    ax[0,0].set_ylabel('sigma')
    # ax[0,0].set_xlim(left=t_x_grid[0], right=t_x_grid[-1])
    ax[1,0].plot(t_x_grid, alpha_y)
    # ax[1,0].set_xlabel('z_grid')
    ax[1,0].set_ylabel('alpha')
    # ax[1,0].set_xlim(left=t_x_grid[0], right=t_x_grid[-1])
    ax[2,0].plot(t_x_grid, transmission_y)
    # ax[2,0].set_xlabel('z_grid')
    ax[2,0].set_ylabel('transmittance')
    # ax[2,0].set_xlim(left=t_x_grid[0], right=t_x_grid[-1])
    ax[3,0].plot(t_x_grid, weight_y)
    ax[3,0].set_xlabel('z_grid')
    ax[3,0].set_ylabel('weight')
    ax[3,0].set_xlim(left=t_x_grid[0], right=t_x_grid[-1])

    ax[0,1].plot(t_x, sigma_y)
    # ax[1,0].set_title(f'depth={out_depth.item():.5f}')
    # ax[0,1].set_xlabel('z')
    # ax[0,1].set_ylabel('sigma')
    # ax[0,1].set_xlim(left=t_x[0], right=t_x[-1])
    ax[1,1].plot(t_x, alpha_y)
    # ax[1,1].set_xlabel('z')
    # ax[1,1].set_ylabel('alpha')
    # ax[1,1].set_xlim(left=t_x[0], right=t_x[-1])
    ax[2,1].plot(t_x, transmission_y)
    # ax[2,1].set_xlabel('z')
    # ax[2,1].set_ylabel('transmission')
    # ax[2,1].set_xlim(left=t_x[0], right=t_x[-1])
    ax[3,1].plot(t_x, weight_y)
    ax[3,1].set_xlabel('z')
    # ax[3,1].set_ylabel('weight')
    ax[3,1].set_xlim(left=t_x[0], right=t_x[-1])

    fig, ax = plt.subplots(3, 3, num=figname+'_rgb')
    r0, g0, b0 = color.squeeze().numpy().tolist()
    r1, g1, b1 = out_rgb[0].cpu().numpy().tolist()
    fig.suptitle(f'rgb_gt=({r0:.3f},{g0:.3f},{b0:.3f})\nrgb_eval=({r1:.3f},{g1:.3f},{b1:.3f})')
    rgb_y = np.asarray(rgb_y)
    ax[0,0].plot(t_x, rgb_y[:, 0], 'r')
    ax[0,0].plot(t_x, rgb_y[:, 1], 'g')
    ax[0,0].plot(t_x, rgb_y[:, 2], 'b')
    # ax[0,0].set_title(f'rgb=({r:.3f},{g:.3f},{b:.3f})')
    ax[0,0].set_xlabel('z')
    ax[0,0].set_ylabel('rgb')
    ax[0,0].legend(['r', 'g', 'b'])
    ax[0,0].set_xlim(left=t_x[0], right=t_x[-1])

    for i in range(grid.basis_dim):
        ax[1,0].plot(t_x, np.full_like(t_x, sh_mult.squeeze(dim=0)[i].item()))
    ax[1,0].set_ylabel('sh_basis')
    ax[1,0].legend([f'sh_basis{i}' for i in range(grid.basis_dim)])

    sh_coeff_y = np.asarray(sh_coeff_y)
    for i in range(grid.basis_dim):
        ax[0,1].plot(t_x, sh_coeff_y[:,0,i])
    ax[0,1].set_ylabel('red coeffs')
    ax[0,1].legend([f'sh_coeff{i}' for i in range(grid.basis_dim)])
    for i in range(grid.basis_dim):
        ax[1,1].plot(t_x, sh_coeff_y[:,1,i])
    ax[1,1].set_ylabel('green coeffs')
    ax[1,1].legend([f'sh_coeff{i}' for i in range(grid.basis_dim)])
    for i in range(grid.basis_dim):
        ax[2,1].plot(t_x, sh_coeff_y[:,2,i])
    ax[2,1].set_ylabel('blue coeffs')
    ax[2,1].legend([f'sh_coeff{i}' for i in range(grid.basis_dim)])

    rgb_comp_y = np.asarray(rgb_comp_y)
    ax[0,2].plot(t_x, rgb_comp_y[:,0,:].sum(-1), 'r')
    for i in range(grid.basis_dim):
        ax[0,2].plot(t_x, rgb_comp_y[:,0,i])
    ax[0,2].set_ylabel('red components')
    ax[0,2].legend(['r'] + [f'r_comp{i}' for i in range(grid.basis_dim)])
    ax[1,2].plot(t_x, rgb_comp_y[:,1,:].sum(-1), 'g')
    for i in range(grid.basis_dim):
        ax[1,2].plot(t_x, rgb_comp_y[:,1,i])
    ax[1,2].set_ylabel('green components')
    ax[1,2].legend(['g'] + [f'g_comp{i}' for i in range(grid.basis_dim)])
    ax[2,2].plot(t_x, rgb_comp_y[:,2,:].sum(-1), 'b')
    for i in range(grid.basis_dim):
        ax[2,2].plot(t_x, rgb_comp_y[:,2,i])
    ax[2,2].set_ylabel('blue components')
    ax[2,2].legend(['b'] + [f'b_comp{i}' for i in range(grid.basis_dim)])

    paths = args.data_dir.split('/')
    savepath = path.join('.\\vis_rays', paths[-2], paths[-1])
    os.makedirs(savepath, exist_ok=True)
    imageio.imwrite(path.join(savepath, figname+'_img.png'), img)
    plt.show(block=True)
    exit(0)
    
if args.vis_train_ray:
    ray_id = json.loads(args.train_ray_id)
    vis_ray(train_dset, ray_id, grid)

if args.vis_test_ray:
    ray_id = json.loads(args.test_ray_id)
    vis_ray(dset, ray_id, grid)

if grid.use_background:
    if args.nobg:
        #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        grid.background_data.data[..., -1] = 0.0
        render_dir += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        render_dir += '_nofg'

    # DEBUG
    #  grid.links.data[grid.links.size(0)//2:] = -1
    #  render_dir += "_chopx2"

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_dir += '_blackbg'
    grid.opt.background_brightness = 0.0

if args.background_brightness != 1.0:
    render_dir += f'_bg{args.background_brightness}'
    
print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

if not args.no_imsave:
    print('Will write out all frames as PNG (this take most of the time)')

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)
    # DEBUGGING
    #  rad = [1.496031746031746, 1.6613756613756614, 1.0]
    #  half_sz = [grid.links.size(0) // 2, grid.links.size(1) // 2]
    #  pad_size_x = int(half_sz[0] - half_sz[0] / 1.496031746031746)
    #  pad_size_y = int(half_sz[1] - half_sz[1] / 1.6613756613756614)
    #  print(pad_size_x, pad_size_y)
    #  grid.links[:pad_size_x] = -1
    #  grid.links[-pad_size_x:] = -1
    #  grid.links[:, :pad_size_y] = -1
    #  grid.links[:, -pad_size_y:] = -1
    #  grid.links[:, :, -8:] = -1

    #  LAYER = -16
    #  grid.links[:, :, :LAYER] = -1
    #  grid.links[:, :, LAYER+1:] = -1

    frames = []
    frames_noise = []
    vis_depths = []
    #  im_gt_all = dset.gt.to(device=device)

    index_rows = []
    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = dset.get_image_size(img_id)
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', img_id),
                           dset.intrins.get('fy', img_id),
                           dset.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                           dset.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                           w, h,
                           ndc_coeffs=dset.ndc_coeffs)
        # start = time.time()
        im = grid.volume_render_image(cam, use_kernel=args.use_kernel, batch_size=10000, return_raylen=args.ray_len)
        # print(f'Elapsed time is {time.time()-start:.2f}s for rendering {dset.ids[img_id]}th image')
        # start = time.time()
        depth = grid.volume_render_depth_image(cam, batch_size=10000, use_kernel=True)
        # print(f'Elapsed time is {time.time()-start:.2f}s for rendering {dset.ids[img_id]}th depth')

        if args.ray_len:
            minv, meanv, maxv = im.min().item(), im.mean().item(), im.max().item()
            im = viridis_cmap(im.cpu().numpy())
            cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                        0, 0.5, [255, 0, 0])
            im = torch.from_numpy(im).to(device=device)
        im.clamp_(0.0, 1.0)
        depth_gray = lin_norm(depth.cpu().numpy()) # (H,W)
        depth_color = viridis_cmap(depth.cpu().numpy())

        if not args.render_path:
            indexes = {}
            indexes['img_id'] = dset.ids[img_id]
            im_gt = dset.gt[img_id].to(device=device)
            mse = (im - im_gt) ** 2
            mse_num : float = mse.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            indexes['PSNR'] = psnr
            avg_psnr += psnr
            if not args.timing:
                ssim = compute_ssim(im_gt, im).item()
                indexes['SSIM'] = ssim
                avg_ssim += ssim
                if not args.no_lpips:
                    lpips_i = lpips_vgg(im_gt.permute([2, 0, 1]).contiguous(),
                            im.permute([2, 0, 1]).contiguous(), normalize=True).item()
                    indexes['LPIPS'] = lpips_i
                    avg_lpips += lpips_i
                    print(img_id, 'PSNR', psnr, 'SSIM', ssim, 'LPIPS', lpips_i)
                else:
                    print(img_id, 'PSNR', psnr, 'SSIM', ssim)
            index_rows.append(indexes)
        id = img_id if args.render_path else dset.ids[img_id]
        gt_img_path = path.join(render_dir, f'{id:04d}_gt.png')
        pred_img_path = path.join(render_dir, f'{id:04d}_pred.png')
        img_path = path.join(render_dir, f'{id:04d}.png')
        gdepth_path = path.join(render_dir, f'{id:04d}_gdepth.png')
        cdepth_path = path.join(render_dir, f'{id:04d}_cdepth.png')
        exrdepth_path = path.join(render_dir, f'{id:04d}_depth.exr')
        im = im.cpu().numpy()
        im_pred = im
        if not args.render_path:
            im_gt = dset.gt[img_id].numpy()
            im = np.concatenate([im_gt, im], axis=1)
        if not args.timing:
            if not args.render_path:
                im_gt = (im_gt * 255).astype(np.uint8)
            im_pred = (im_pred * 255).astype(np.uint8)
            im = (im * 255).astype(np.uint8)
            if args.add_noise:
                im_noise = np.array(add_text_overlay(Image.fromarray(im), 0.25, 1))
            depth_gray = (depth_gray * 255).astype(np.uint8)
            depth_color = (depth_color * 255).astype(np.uint8)
            if not args.no_imsave:
                if not args.render_path:
                    imageio.imwrite(gt_img_path, im_gt)
                imageio.imwrite(pred_img_path,im_pred)
                imageio.imwrite(img_path,im)
                imageio.imwrite(gdepth_path, depth_gray)
                imageio.imwrite(cdepth_path, depth_color)
                pyexr.write(exrdepth_path, depth.cpu().numpy(), channel_names='Z')
            if not args.no_vid:
                frames.append(im)
                if args.add_noise:
                    frames_noise.append(im_noise)
                vis_depths.append(depth_color)
        im = None
        n_images_gen += 1

    if len(index_rows) != 0:
        import csv
        with open(f'{render_dir}/indexes.csv', 'w', newline='') as csvfile:
            fieldnames = index_rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            sorted_index_rows = sorted(index_rows, key=lambda x: x['PSNR'])
            for row in sorted_index_rows:
                writer.writerow(row)

    if want_metrics:
        print('AVERAGES')

        avg_psnr /= n_images_gen
        with open(path.join(render_dir, 'psnr.txt'), 'w') as f:
            f.write(str(avg_psnr))
        print('PSNR:', avg_psnr)
        if not args.timing:
            avg_ssim /= n_images_gen
            print('SSIM:', avg_ssim)
            with open(path.join(render_dir, 'ssim.txt'), 'w') as f:
                f.write(str(avg_ssim))
            if not args.no_lpips:
                avg_lpips /= n_images_gen
                print('LPIPS:', avg_lpips)
                with open(path.join(render_dir, 'lpips.txt'), 'w') as f:
                    f.write(str(avg_lpips))

                def geo_mean(psnr, ssim, lpips):
                    mse = np.power(10, -psnr/10)
                    return np.power((mse * np.sqrt(1-ssim) * lpips), 1/3).item()
                
                comp = geo_mean(avg_psnr, avg_ssim, avg_lpips)
                print('Mean:', comp)
                with open(path.join(render_dir, 'mean.txt'), 'w') as f:
                    f.write(str(comp))

    if not args.no_vid and len(frames):
        vid_path = render_dir + '.mp4'
        imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg
        if len(vis_depths):
            vid_path_depth = render_dir + '_depth.mp4'
            imageio.mimwrite(vid_path_depth, vis_depths, fps=args.fps, macro_block_size=8)
        if len(frames_noise):
            vid_path_noise = render_dir + '_noise.mp4'
            imageio.mimwrite(vid_path_noise, frames_noise, fps=args.fps, macro_block_size=8)


