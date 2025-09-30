# Copyright 2021 Alex Yu

# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import random
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap, compute_ssim
from util import config_util

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union
import matplotlib.pyplot as plt
import time

import open3d as o3d


parser = argparse.ArgumentParser()
config_util.define_common_args(parser)


group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')
group.add_argument('--ckpt', type=str)
group.add_argument('--only_use_links', type=int, default=1)
group.add_argument('--reso',
                        type=str,
                        default=
                        "[[256, 256, 256], [512, 512, 512]]",
                       help='List of grid resolution (will be evaled as json);'
                            'resamples to the next one every upsamp_every iters, then ' +
                            'stays at the last one; ' +
                            'should be a list where each item is a list of 3 ints or an int')
group.add_argument('--use_prune', action='store_true', default=False)
# group.add_argument('--prune_every', type=int, default = 
                     # 3 * 12800,
                     # help='prune the grid every x iters')
group.add_argument('--prune_every', type=int, default = 3,
                     help='prune the grid every x epochs')
group.add_argument('--use_upsample', type=int, default=1)
group.add_argument('--upsample_no_prune', type=int, default=0)
group.add_argument('--upsamp_every', type=int, default=
                    3,
                    # 3 * 12800,
                    help='upsample the grid every x iters')
group.add_argument('--dilate', type=int, default=2)
group.add_argument('--init_dilate', action="store_true", default=False)
group.add_argument('--first_upsample_dilate', action="store_true", default=False)
group.add_argument('--init_iters', type=int, default=
                     0,
                    help='do not upsample for first x iters')
group.add_argument('--upsample_density_add', type=float, default=
                    0.0,
                    help='add the remaining density by this amount when upsampling')

group.add_argument('--basis_type',
                    choices=['sh', '3d_texture', 'mlp'],
                    default='sh',
                    help='Basis function type')

group.add_argument('--basis_reso', type=int, default=32,
                   help='basis grid resolution (only for learned texture)')
group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

group.add_argument('--background_nlayers', type=int, default=0,#32,
                   help='Number of background layers (0=disable BG model)')
group.add_argument('--background_reso', type=int, default=512, help='Background resolution')



group = parser.add_argument_group("optimization")
group.add_argument('--n_iters', type=int, default=10 * 12800, help='total number of iters to optimize for')
group.add_argument('--n_epochs', type=int, default=10)
group.add_argument('--batch_size', type=int, default=
                     5000,
                     #100000,
                     #  2000,
                   help='batch size')


# TODO: make the lr higher near the end
group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=
                    1e-2,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,
                      default=
                    5e-6
                    )
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

# BG LRs
group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_color_bg', type=float, default=1e-1,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
# END BG LRs

group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
group.add_argument('--lr_basis', type=float, default=#2e6,
                      1e-6,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_basis_final', type=float,
                      default=
                      1e-6
                    )
group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_basis_begin_step', type=int, default=0)#4 * 12800)
group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--vis_training', action='store_true', default=False)
group.add_argument('--vis_training_every', type=int, default=11, help='visualize training for some training view every')
group.add_argument('--vis_img_id', type=int, default=0, help='image id for visualizing')
group.add_argument('--fps', type=int, default=30, help="FPS of video")
group.add_argument('--log_grad_sparsity', action='store_true', help='log the sparsity of gradient')
group.add_argument('--log_maxmin_density', action='store_true')
group.add_argument('--save_every', type=int, default=5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')
group.add_argument('--init_sigma_bg', type=float,
                   default=0.1,
                   help='initialization sigma (for BG)')

# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_depth_map', action='store_true', default=False)
group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
        help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


group = parser.add_argument_group("misc experiments")
group.add_argument('--thresh_type',
                    choices=["weight", "sigma"],
                    default="weight",
                   help='Upsample threshold type')
group.add_argument('--weight_thresh', type=float,
                    default=0.0005 * 512,
                    #  default=0.025 * 512,
                   help='Upsample weight threshold; will be divided by resulting z-resolution')
group.add_argument('--density_thresh', type=float,
                    default=5.0,
                   help='Upsample sigma threshold')
group.add_argument('--background_density_thresh', type=float,
                    default=1.0+1e-9,
                   help='Background sigma threshold for sparsification')
group.add_argument('--max_grid_elements', type=int,
                    default=44_000_000,
                   help='Max items to store after upsampling '
                        '(the number here is given for 22GB memory)')

group.add_argument('--tune_mode', action='store_true', default=False,
                   help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=False,
                   help='do not save any checkpoint even at the end')



group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument('--lambda_tv', type=float, default=1e-5)
group.add_argument('--tv_sparsity', type=float, default=0.01)
group.add_argument('--tv_logalpha', action='store_true', default=False,
                   help='Use log(1-exp(-delta * sigma)) as in neural volumes')

group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

group.add_argument('--tv_decay', type=float, default=1.0)

group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

group.add_argument('--tv_contiguous', type=int, default=1,
                        help="Apply TV only on contiguous link chunks, which is faster")
# End Foreground TV

group.add_argument('--lambda_sparsity', type=float, default=
                    0.0,
                    help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                         "(but applied on the ray)")
group.add_argument('--lambda_beta', type=float, default=
                    0.0,
                    help="Weight for beta distribution sparsity loss as in neural volumes")


# Background TV
group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

group.add_argument('--tv_background_sparsity', type=float, default=0.01)
# End Background TV

# Basis TV
group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                   help='Learned basis total variation loss')
# End Basis TV

group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)

group.add_argument('--lr_decay', action='store_true', default=True)

group.add_argument('--hardcode_train_views', type=int, nargs='+', help='hardcode training views')
group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

# my addition
group.add_argument('--skip', type=int, default=1, help='select images for test every skip.')
group.add_argument('--noacc', action='store_true', default=False)
group.add_argument('--use_z_order', action='store_true', default=True)
'''
group.add_argument('--N_noise_samples', type=int, default=8, help='the number of noisy samples every camera pose')
group.add_argument('--gaussian_noise', type=bool, default=False, help='add gaussian noise to g.t. color')
group.add_argument('--std', type=float, default=50./255, help='standard deviation')
group.add_argument('--textOverlay', type=bool, default=False, help='add text noise to g.t. color')
group.add_argument('--max_occupancy', type=float, default=0.5, help='text overlay proportion at the whole image')
'''
group.add_argument('--loss', choices=['l1', 'l2'], default='l2', type=str, help='loss type')

group.add_argument('--noise_aug', action='store_true', default=False)
group.add_argument('--noise_type', choices=['gaussian', 'impulse'], default='gaussian')
group.add_argument('--max_s', type=float, default=25./255, help='maximum standard deviation')
group.add_argument('--std', type=float, default=15./255, help='standard deviation')
group.add_argument('--max_p', type=float, default=0.3)
group.add_argument('--prob', type=float, default=0.1)

group.add_argument('--vis_grid', type=bool, default=False)
group.add_argument('--vis_depth2pcd', type=bool, default=False)
group.add_argument('--depth_ndc2pcd', type=bool, default=False)
group.add_argument('--depth_cam2pcd', type=bool, default=False)

# Annealing
group.add_argument('--annealing', type=bool, default=False)
group.add_argument('--Nt', type=int, default=960, help='how many iterations until the full range is reached')
group.add_argument('--Ps', type=float, default=0.5, help='start range')

# increment spherical harmonics basis during training
group.add_argument('--sh_basis_incre', type=bool, default=False)
group.add_argument('--Nf', type=int, default=320*3, help='how many iterations until all spherical harmonics basis functions set by sh_dim are used')
group.add_argument('--used_sh_dim', type=int, default=9, help='SH/learned basis dimensions used')

group.add_argument('--low_freq_init', type=bool, default=False)
group.add_argument('--N_init', type=int, default=3)

group.add_argument('--use_gradient_scaling', type=bool, default=False)
group.add_argument('--Ns', type=int, default=1280, help="how many iterations for gradient scaling")

group.add_argument('--use_alpha_prune', action='store_true', default=False)
group.add_argument('--use_extra_alpha_prune', action='store_true', default=False)
group.add_argument('--extra_prune_factor', type=float, default=0.2)
group.add_argument('--use_depth_prune', action='store_true', default=False)
group.add_argument('--d_prop', type=float, default=0.99)
group.add_argument('--unrobust_d_pro', type=float, default=0.5)
group.add_argument('--use_isotropy_init', action='store_true', default=False)
group.add_argument('--density_init_withDepth', type=float, default=128.0)
group.add_argument('--use_sphere_bound', type=int, default=1)
group.add_argument('--nosphereinit', action='store_true', default=False,
                     help='do not start with sphere bounds (please do not use for 360)')

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

os.makedirs(args.train_dir, exist_ok=True)
# logdir = os.path.join(args.train_dir, 'logs')
# os.makedirs(logdir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)
random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
               args.data_dir,
               split="train",
               device=device,
               factor=factor,
               hardcode_train_views = args.hardcode_train_views,
               n_images=args.n_train,
               extra_pruned_rays=args.use_extra_alpha_prune,
               prune_factor=args.extra_prune_factor,
               depth_source=args.depth_source if args.use_depth_prune else None,
               use_vggt_intri=args.use_vggt_intri,
               use_zdepth=args.use_zdepth,
               **config_util.build_data_options(args))
'''
noise = torch.normal(0., 25/255, size=dset.rays.gt.shape, device=dset.rays.gt.device)
dset.rays.gt = torch.clamp((dset.rays.gt + noise), 0., 1.)
'''

if args.background_nlayers > 0 and not dset.should_use_background:
    warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

dset_test = datasets[args.dataset_type](
                        args.data_dir, 
                        split="test", 
                        skip=args.skip, 
                        depth_source=args.depth_source if args.use_depth_prune else None,
                        use_vggt_intri=args.use_vggt_intri, 
                        **config_util.build_data_options(args))

global_start_time = datetime.now()

if args.ckpt != None:
    grid = svox2.SparseGrid.load(args.ckpt, device=device)
    # if args.only_use_links:
        # grid.sh_data = 0.0
        # grid.density_data = 0.1
else:
    grid = svox2.SparseGrid(reso=reso_list[reso_id],
                        center=dset.scene_center,
                        # center=[-1.0, -1.0, 0.0],
                        radius=dset.scene_radius,
                        # radius=[1.0, 1.0, 0.5],
                        use_sphere_bound=dset.use_sphere_bound and args.use_sphere_bound and not args.nosphereinit,
                        # use_sphere_bound = False,
                        basis_dim=args.sh_dim,
                        use_z_order=args.use_z_order,
                        device=device,
                        basis_reso=args.basis_reso,
                        basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()],
                        mlp_posenc_size=args.mlp_posenc_size,
                        mlp_width=args.mlp_width,
                        background_nlayers=args.background_nlayers,
                        background_reso=args.background_reso,
                        acc = not args.noacc,
                        alpha_c = dset.alpha_c,
                        use_alpha_prune = True if dset.alpha_c and args.use_alpha_prune else False,
                        use_extra_alpha_prune = True if dset.alpha_c and args.use_extra_alpha_prune else False,
                        # use_alpha_prune = False,
                        use_depth_prune = args.use_depth_prune,
                        # use_depth_prune = True,
                        d_prop = args.d_prop,
                        unrobust_d_pro = args.unrobust_d_pro,
                        use_isotropy_init = args.use_isotropy_init,
                        density_init_withDepth = args.density_init_withDepth,
                        dataset_type = args.dataset_type,
                        use_gradient_scaling = args.use_gradient_scaling,
                        init_sigma = args.init_sigma,
                        rays = dset.rays_init) # NDC for llff


if args.init_dilate:
    grid.init_dilate(reso_list[reso_id], args.dilate)

# if args.only_use_links:
if True:
    # DC -> gray; mind the SH scaling!
    # grid.sh_data.data[:] = 0.0
    # grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma

    if dset.rays.depths != None and args.vis_depth2pcd:
        points = (dset.rays.origins + dset.rays.dirs * dset.rays.depths).numpy().reshape(-1,dset.h,dset.w,3)
        colors = dset.rays.gt.numpy().reshape(-1,dset.h,dset.w,3)
        pcd = o3d.geometry.PointCloud()
        aabb_01 = np.array([[-1, -1, -1],
                        [-1, -1, 1],
                        [-1, 1, 1],
                        [-1, 1, -1],
                        [1, -1, -1],
                        [1, -1, 1],
                        [1, 1, 1],
                        [1, 1, -1]])
        out_bbox = o3d.geometry.LineSet()
        out_bbox.points = o3d.utility.Vector3dVector(aabb_01)
        out_bbox.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
        out_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
        for i, (points_perimg, colors_perimg) in enumerate(zip(points,colors)):
            pcd.points = o3d.utility.Vector3dVector(points_perimg.reshape(-1,3))
            pcd.colors = o3d.utility.Vector3dVector(colors_perimg.reshape(-1,3))
            o3d.visualization.draw_geometries([
                o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[-1,-1,-1]),
                pcd,out_bbox], window_name=f'img_id{dset.ids[i]}', top=150)
        pass

    # only for debug
    '''
    depth_path = "./opt/ckpt_llff/llff_flower/train_renders_bg0.5/0033_depth.exr"
    # depth_path = "F:\\Experimental Results\\03\\ckpt_llff_3v_baseline\\llff_flower_258_tv5x_decay0.2_sigmalr15\\train_renders_bg0.5\\0033_depth.exr"
    img_path = "./opt/ckpt_llff/llff_flower/train_renders_bg0.5/0033_pred.png"
    # img_path = "F:\\Experimental Results\\03\\ckpt_llff_3v_baseline\\llff_flower_258_tv5x_decay0.2_sigmalr15\\train_renders_bg0.5\\0033_pred.png"

    import pyexr
    depth = pyexr.read(depth_path)
    dset.rays.depths = torch.ones_like(dset.rays.origins[..., 0]).reshape(-1, dset.h, dset.w, 1)
    img_id = 33 - (33 // 8 + 1)
    dset.rays.depths[img_id] = torch.from_numpy(depth)
    dset.rays.depths = dset.rays.depths.reshape(-1, 1)
    depth_cam = dset.depth_ndc2cam().unsqueeze(-1)
    points = (dset.rays.origins + dset.rays.dirs * dset.rays.depths).numpy().reshape(-1,dset.h,dset.w,3)[img_id]
    # points = (dset.rays_beforeNDC.origins + dset.rays_beforeNDC.dirs * depth_cam).numpy().reshape(-1,dset.h,dset.w,3)[img_id]
    # colors = dset.rays.gt.numpy().reshape(-1,dset.h,dset.w,3)[img_id]
    colors = imageio.v2.imread(img_path) / 255.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3))
    pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1,3))
    # dir_name = os.path.join(args.data_dir, os.path.dirname(dset.imgs[img_id]['d_path']))
    # base_name = os.path.basename(dset.imgs[i]['d_path']).split('.')[0]
    # o3d.io.write_point_cloud(os.path.join(dir_name, base_name + '_ndc.ply'), pcd)
    # o3d.io.write_point_cloud("F:\\Experimental Results\\03\\ckpt_llff_3v_baseline\\llff_flower_258_tv5x_decay0.2_sigmalr15\\train_renders_bg0.5\\0033_cam.ply", pcd)
    o3d.io.write_point_cloud("./opt/ckpt_llff/llff_flower/train_renders_bg0.5/0033_ndc.ply", pcd)
    '''

    '''
    if dset.rays.depths != None and dset.sfm.dataset_type == 'llff' and args.depth_ndc2pcd:
        points = (dset.rays.origins + dset.rays.dirs * dset.rays.depths).numpy().reshape(-1,dset.h,dset.w,3)
        colors = dset.rays.gt.numpy().reshape(-1,dset.h,dset.w,3)
        masks = dset.rays.masks.squeeze().reshape(-1,dset.h,dset.w,1).squeeze()
        for i, (points_perimg, colors_perimg, mask_perimg) in enumerate(zip(points,colors,masks)): 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_perimg[mask_perimg].reshape(-1,3))
            pcd.colors = o3d.utility.Vector3dVector(colors_perimg[mask_perimg].reshape(-1,3))
            dir_name = os.path.join(args.data_dir, os.path.dirname(dset.imgs[i]['d_path']))
            base_name = os.path.basename(dset.imgs[i]['d_path']).split('.')[0]
            o3d.io.write_point_cloud(os.path.join(dir_name, base_name + '_ndc.ply'), pcd)
    


    if dset.rays.depths != None and dset.sfm.dataset_type == 'llff' and args.depth_cam2pcd:
        depth_cam = dset.depth_ndc2cam().unsqueeze(-1)
        points = (dset.rays_beforeNDC.origins + dset.rays_beforeNDC.dirs * depth_cam).numpy().reshape(-1,dset.h,dset.w,3)
        colors = dset.rays.gt.numpy().reshape(-1,dset.h,dset.w,3)
        masks = dset.rays.masks.squeeze().reshape(-1,dset.h,dset.w,1).squeeze()
        for i, (points_perimg, colors_perimg, mask_perimg) in enumerate(zip(points,colors,masks)): 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_perimg[mask_perimg].reshape(-1,3))
            pcd.colors = o3d.utility.Vector3dVector(colors_perimg[mask_perimg].reshape(-1,3))
            dir_name = os.path.join(args.data_dir, os.path.dirname(dset.imgs[i]['d_path']))
            base_name = os.path.basename(dset.imgs[i]['d_path']).split('.')[0]
            o3d.io.write_point_cloud(os.path.join(dir_name, base_name + '_cam.ply'), pcd)
    '''
    
    # vis_grid = False
    if args.vis_grid:
        def world2grid(points):
            # center = torch.asarray([0.0,0.0,0.0], device='cpu')
            center = grid.center
            # radius = torch.asarray([1.0,1.0,1.0], device='cpu')
            radius = grid.radius
            _offset = 0.5 * (1.0 - center / radius)
            _scaling = 0.5 / radius
            gsz = grid._grid_size()
            offset = _offset * gsz - 0.5
            scaling = _scaling * gsz
            return torch.addcmul(
                offset.to(device=points.device), points, scaling.to(device=points.device)
            )
        def grid2world(points):
            gsz = grid._grid_size()
            roffset = grid.radius * (1.0 / gsz - 1.0) + grid.center
            rscaling = 2.0 * grid.radius / gsz
            return torch.addcmul(
            roffset.to(device=points.device), points, rscaling.to(device=points.device)
        )
        # cameras
        N, H, W = dset.n_images, dset.w, dset.h
        origins = dset.rays.origins.reshape(N, H, W, -1)
        dirs = dset.rays.dirs.reshape(N, H, W, -1)
        cam_lst = []
        for i_img in range(N):
            p0 = origins[i_img, 0, 0]
            p1 = p0 + dirs[i_img, 0, 0]*1.0
            p2 = p0 + dirs[i_img, 0, W-1]*1.0
            p3 = p0 + dirs[i_img, H-1, W-1]*1.0
            p4 = p0 + dirs[i_img, H-1, 0]*1.0
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
        
        IJK = np.stack((grid.links >= 0).cpu().numpy().nonzero(), -1)
        ndc_points = grid2world(torch.from_numpy(IJK.astype(np.float32)).to(device)).cpu().numpy()
        sh_dim = grid.sh_data.data.size(-1) // 3
        color = grid.sh_data.data[grid.links[IJK[:,0], IJK[:,1], IJK[:,2]].to(torch.long)][:,::sh_dim].cpu().numpy()
        color = np.clip(color*svox2.utils.SH_C0+0.5, 0.0, 1.0)
        # color = np.random.uniform(size=(IJK.shape))
        pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(IJK)
        pcd.points = o3d.utility.Vector3dVector(IJK)
        pcd.colors = o3d.utility.Vector3dVector(color)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        # o3d.visualization.draw_geometries([voxel_grid], top=150)
        o3d.visualization.draw_geometries([
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=int(min(L,W,H)*0.1), origin=[0,0,0]),
            pcd,
            out_bbox,  
            # *cam_frustrm_lst
        ], window_name='vis_depthprune', top=150)  

    if grid.use_background:
        grid.background_data.data[..., -1] = args.init_sigma_bg
        #  grid.background_data.data[..., :-1] = 0.5 / svox2.utils.SH_C0

    optim_basis_mlp = None

    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        grid.reinit_learned_bases(init_type='sh')
        #  grid.reinit_learned_bases(init_type='fourier')
        #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
        #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

    elif grid.basis_type == svox2.BASIS_TYPE_MLP:
        # MLP!
        optim_basis_mlp = torch.optim.Adam(
                        grid.basis_mlp.parameters(),
                        lr=args.lr_basis
                    )

    grid.requires_grad_(True)
    config_util.setup_render_opts(grid.opt, args)
    print('Render options', grid.opt)

gx_c, gy_c, gz_c = grid.reso[0]//2, grid.reso[1]//2, grid.reso[2]//2
'''
gx_slices = range(gx_c - grid.reso[0]//4, gx_c + grid.reso[0]//4)
gy_slices = range(gy_c - grid.reso[1]//4, gy_c + grid.reso[1]//4)
gz_slices = range(gz_c - grid.reso[2]//4, gz_c + grid.reso[2]//4)
links = grid.links[gx_slices, gy_slices, gz_slices].view(-1)
grid.density_data.data[links.long()] = 30
'''

#  grid.sh_data.data[:, 0] = 4.0
#  osh = grid.density_data.data.shape
#  den = grid.density_data.data.view(grid.links.shape)
#  #  den[:] = 0.00
#  #  den[:, :256, :] = 1e9
#  #  den[:, :, 0] = 1e9
#  grid.density_data.data = den.view(osh)

gstep_id_base = 0

resample_cameras = [
        svox2.Camera(c2w.to(device=device),
                     dset.intrins.get('fx', i),
                     dset.intrins.get('fy', i),
                     dset.intrins.get('cx', i),
                     dset.intrins.get('cy', i),
                     width=dset.get_image_size(i)[1],
                     height=dset.get_image_size(i)[0],
                     ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
    ]
ckpt_path = path.join(args.train_dir, 'ckpt.npz')

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                               args.lr_basis_delay_mult, args.lr_basis_decay_steps)
lr_sigma_bg_func = get_expon_lr_func(args.lr_sigma_bg, args.lr_sigma_bg_final, args.lr_sigma_bg_delay_steps,
                               args.lr_sigma_bg_delay_mult, args.lr_sigma_bg_decay_steps)
lr_color_bg_func = get_expon_lr_func(args.lr_color_bg, args.lr_color_bg_final, args.lr_color_bg_delay_steps,
                               args.lr_color_bg_delay_mult, args.lr_color_bg_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

last_upsamp_step = args.init_iters
last_upsamp_epoch = 0
last_prune_step = args.init_iters
last_prune_epoch = 0

if args.enable_random:
    warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

frames = []
epoch_id = -1

'''
if args.textOverlay or args.gaussian_noise:
    samples = None
    if args.gaussian_noise:
        noise_data = f'samples_gaussian_{args.N_noise_samples}.npy'
        tmp_name = f'tmp_gaussian_{args.N_noise_samples}'
    if args.textOverlay:
        noise_data = f'samples_textOverlay_{args.N_noise_samples}.npy'
        tmp_name = f'tmp_textOverlay_{args.N_noise_samples}'
    if path.isfile(path.join(args.data_dir, noise_data)):
        samples = np.load(path.join(args.data_dir, noise_data))
        samples = torch.from_numpy(samples)
        tmppath = path.join(args.data_dir, tmp_name)
        os.makedirs(tmppath, exist_ok=True)
        count = 0
'''

if args.sh_basis_incre:
    if args.dataset_type == "nerf":
        sh_incre_ids = [3, 6]
    elif args.dataset_type == "llff":
        sh_incre_ids = [0, 1]
    degree = int(math.sqrt(args.used_sh_dim))
    degrees = list(range(1, degree+1))
    degrees = [degree**2 for degree in degrees]
    if len(degrees) == 2:
        degrees.append(degrees[-1])



while True:
    '''
    if args.textOverlay:
        if samples != None:
            print(f"Load samples from samples_textOverlay_{args.N_noise_samples}.npy")
            dset.get_sample(samples, count)
            test_img = dset.rays_init.gt.reshape(-1, dset.h, dset.w, 3)[0]
            imageio.imwrite(path.join(tmppath, f'{epoch_id+1}.png'), (test_img.numpy()*255).astype(np.uint8))
            count += 1
            if count % args.N_noise_samples == 0:
                count = 0
        else:
            start = time.time()
            dset.add_text_noise(args.max_occupancy)
            print(f"Elapsed time (seconds) for text overlay: {time.time()-start}")
    if args.gaussian_noise:
        # if (epoch_id + 1) % args.N_noise_samples == 0:
            # torch.manual_seed(20200823)
            # np.random.seed(20200823)
            # random.seed(20200823)
        if samples != None:
            print(f"Load samples from samples_gaussian_{args.N_noise_samples}.npy")
            dset.get_sample(samples, count)
            test_img = dset.rays_init.gt.reshape(-1, dset.h, dset.w, 3)[0]
            imageio.imwrite(path.join(tmppath, f'{epoch_id+1}.png'), (test_img.numpy()*255).astype(np.uint8))
            count += 1
            if count % args.N_noise_samples == 0:
                count = 0     
        else:
            dset.add_gaussian_noise(args.std)
    '''
    dset.shuffle_rays()
    epoch_id += 1
    epoch_size = dset.rays.origins.size(0)
    batches_per_epoch = (epoch_size-1)//args.batch_size+1
    # Test
    def eval_step():
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}
            psnr_test = {}
            # Standard set
            N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            img_ids = range(0, dset_test.n_images, img_eval_interval)

            # Special 'very hard' specular + fuzz set
            #  img_ids = [2, 5, 7, 9, 21,
            #             44, 45, 47, 49, 56,
            #             80, 88, 99, 115, 120,
            #             154]
            #  img_save_interval = 1

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.get('fx', img_id),
                                   dset_test.intrins.get('fy', img_id),
                                   dset_test.intrins.get('cx', img_id),
                                   dset_test.intrins.get('cy', img_id),
                                   width=dset_test.get_image_size(img_id)[1],
                                   height=dset_test.get_image_size(img_id)[0],
                                   ndc_coeffs=dset_test.ndc_coeffs)
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)

                if False and epoch_id == 0:
                    im_gt = dset_test.gt[img_id].numpy()
                    im_gt = (im_gt * 255).astype(np.uint8)
                    eval_inputs_dir = path.join(args.train_dir, 'eval_inputs')
                    os.makedirs(eval_inputs_dir, exist_ok=True)
                    gt_img_path = path.join(eval_inputs_dir, f'{dset_test.ids[img_id]:04d}_gt.png')
                    imageio.imwrite(gt_img_path, im_gt)

                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)
                    summary_writer.add_image(f'test/image_{dset_test.ids[img_id]:04d}',
                            img_pred, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        summary_writer.add_image(f'test/mse_map_{dset_test.ids[img_id]:04d}',
                                mse_img, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_depth_map:
                        depth_img = grid.volume_render_depth_image(cam,
                                    args.log_depth_map_use_thresh if
                                    args.log_depth_map_use_thresh else None
                                )
                        depth_img = viridis_cmap(depth_img.cpu())
                        summary_writer.add_image(f'test/depth_map_{dset_test.ids[img_id]:04d}',
                                depth_img,
                                global_step=gstep_id_base, dataformats='HWC')

                rgb_pred_test = rgb_gt_test = None 
                mse_num : float = all_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_test['mse'] += mse_num
                stats_test['psnr'] += psnr
                if True and epoch_id + 1 >= args.n_epochs:
                    psnr_test[f'img_{dset_test.ids[img_id]}'] = psnr
                n_images_gen += 1

            if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE or \
               grid.basis_type == svox2.BASIS_TYPE_MLP:
                 # Add spherical map visualization
                EQ_RESO = 256
                eq_dirs = generate_dirs_equirect(EQ_RESO * 2, EQ_RESO)
                eq_dirs = torch.from_numpy(eq_dirs).to(device=device).view(-1, 3)

                if grid.basis_type == svox2.BASIS_TYPE_MLP:
                    sphfuncs = grid._eval_basis_mlp(eq_dirs)
                else:
                    sphfuncs = grid._eval_learned_bases(eq_dirs)
                sphfuncs = sphfuncs.view(EQ_RESO, EQ_RESO*2, -1).permute([2, 0, 1]).cpu().numpy()

                stats = [(sphfunc.min(), sphfunc.mean(), sphfunc.max())
                        for sphfunc in sphfuncs]
                sphfuncs_cmapped = [viridis_cmap(sphfunc) for sphfunc in sphfuncs]
                for im, (minv, meanv, maxv) in zip(sphfuncs_cmapped, stats):
                    cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                                0, 0.5, [255, 0, 0])
                sphfuncs_cmapped = np.concatenate(sphfuncs_cmapped, axis=0)
                summary_writer.add_image(f'test/spheric',
                        sphfuncs_cmapped, global_step=gstep_id_base, dataformats='HWC')
                # END add spherical map visualization

            stats_test['mse'] /= n_images_gen
            stats_test['psnr'] /= n_images_gen
            for stat_name in stats_test:
                summary_writer.add_scalar('test/' + stat_name,
                        stats_test[stat_name], global_step=gstep_id_base)
            summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base)
            if True and epoch_id + 1 >= args.n_epochs:
                print(psnr_test)
            print('eval stats:', stats_test)
    def eval_train_step():
        # Put in a function to avoid memory leak
        print('Eval train step')
        with torch.no_grad():
            stats_train = {'psnr' : 0.0, 'mse' : 0.0}

            # Standard set
            img_ids = range(0, dset.n_images)

            # Special 'very hard' specular + fuzz set
            #  img_ids = [2, 5, 7, 9, 21,
            #             44, 45, 47, 49, 56,
            #             80, 88, 99, 115, 120,
            #             154]
            #  img_save_interval = 1

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset.intrins.get('fx', img_id),
                                   dset.intrins.get('fy', img_id),
                                   dset.intrins.get('cx', img_id),
                                   dset.intrins.get('cy', img_id),
                                   width=dset.get_image_size(img_id)[1],
                                   height=dset.get_image_size(img_id)[0],
                                   ndc_coeffs=dset.ndc_coeffs)
                rgb_pred_train = grid.volume_render_image(cam, use_kernel=True)
                rgb_gt_train = dset.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_train - rgb_pred_train) ** 2).cpu()
                img_pred = rgb_pred_train.cpu()
                img_pred.clamp_max_(1.0)
                summary_writer.add_image(f'train/image_{dset.ids[img_id]:04d}',
                            img_pred, global_step=gstep_id_base, dataformats='HWC')
                if epoch_id == 0:
                    save_train_dir = os.path.join(args.train_dir, "init_train_renders")
                    os.makedirs(save_train_dir, exist_ok=True)
                    imageio.imwrite(os.path.join(save_train_dir, f"r_{dset.ids[i]}.png"), (img_pred.numpy()*255).astype(np.uint8))
                
                write_train_renders = False
                if write_train_renders and epoch_id > 0:
                    save_train_dir = os.path.join(args.train_dir, "vis_train_renders")
                    os.makedirs(save_train_dir, exist_ok=True)
                    imageio.imwrite(os.path.join(save_train_dir, f"r_{dset.ids[i]}_epoch{epoch_id}.png"), (img_pred.numpy()*255).astype(np.uint8))
                    
                    depth_img = grid.volume_render_depth_image(cam,
                                        args.log_depth_map_use_thresh if
                                        args.log_depth_map_use_thresh else None
                                    )
                    depth_img = viridis_cmap(depth_img.cpu().numpy())
                    depth_img = (depth_img * 255).astype(np.uint8)
                    depth_path = os.path.join(save_train_dir, f'depth_{dset.ids[img_id]}_epoch{epoch_id}.png')
                    depth_img = imageio.imwrite(depth_path, depth_img)
                    

                rgb_pred_train = rgb_gt_train = None 
                mse_num : float = all_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_train['mse'] += mse_num
                stats_train['psnr'] += psnr
                n_images_gen += 1

            stats_train['mse'] /= n_images_gen
            stats_train['psnr'] /= n_images_gen
            for stat_name in stats_train:
                summary_writer.add_scalar('train/' + stat_name,
                        stats_train[stat_name], global_step=gstep_id_base)
            print('eval train stats:', stats_train)
    if epoch_id % max(factor, args.eval_every) == 0: #and (epoch_id > 0 or not args.tune_mode):
        # NOTE: we do an eval sanity check, if not in tune_mode
        eval_step()
        if args.n_train != None:
            eval_train_step()
        gc.collect()

    def train_step():
        print('Train step')
        pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
        stats = {"mse" : 0.0, "psnr" : 0.0, "psnr_clean": 0.0, "invsqr_mse" : 0.0}
        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base
            if args.lr_fg_begin_step > 0 and gstep_id == args.lr_fg_begin_step:
                grid.density_data.data[:] = args.init_sigma
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_sh = args.lr_sh * lr_sh_factor
                lr_basis = args.lr_basis * lr_basis_factor

            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_origins = dset.rays.origins[batch_begin: batch_end]
            batch_dirs = dset.rays.dirs[batch_begin: batch_end]
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            # rgb_gt = dset.rays.gt[batch_begin: batch_end].clone() # match the folloing comments
            '''
            if args.textOverlay or args.gaussian_noise:
                rgb_gt_clean = dset.rays.gt_c[batch_begin: batch_end]
            '''

            '''
            if args.noise_aug:
                # rgb_gt_clean = rgb_gt.clone()
                rgb_gt_clean = dset.rays.gt[batch_begin: batch_end]
                if args.noise_type == 'impulse':
                    p = random.uniform(0, args.max_p)
                    N = math.floor(len(rgb_gt)*p)
                    # pos = random.choices(range(len(rgb_gt)), k=N) # with replacement
                    pos = np.random.choice(len(rgb_gt), N, replace=False).tolist()
                    cols = torch.rand(N, 3).to(device=device)
                    rgb_gt[pos] = cols
                elif args.noise_type == 'gaussian':
                    std = random.uniform(0, args.max_s)
                    noise = torch.normal(0, std, size=rgb_gt.shape).to(device=device)
                    # rgb_gt = torch.clamp((rgb_gt + noise), 0., 1.)
                    rgb_gt = rgb_gt + noise
            '''

            '''
            if args.gaussian_noise:
                rgb_gt_clean = rgb_gt.clone()
                std = random.uniform(0, args.std)
                noise = torch.normal(0, std, size=rgb_gt_clean.shape, device=device)
                rgb_gt = torch.clamp((rgb_gt + noise), 0., 1.)
            '''

            rays = svox2.Rays(batch_origins, batch_dirs)

            if args.annealing and gstep_id <= args.Nt-1:
                eta = min(max(gstep_id/(args.Nt-1), args.Ps), 1)
                # eta = 0.8
                gx_s = max(gx_c - math.floor(eta * (grid.reso[0]//2)), 0)
                gx_e = min(gx_c + math.floor(eta * (grid.reso[0]//2)), grid.reso[0]-1)
                gy_s = max(gy_c - math.floor(eta * (grid.reso[1]//2)), 0)
                gy_e = min(gy_c + math.floor(eta * (grid.reso[1]//2)), grid.reso[1]-1)
                gz_s = max(gz_c - math.floor(eta * (grid.reso[2]//2)), 0) 
                gz_e = min(gz_c + math.floor(eta * (grid.reso[2]//2)), grid.reso[2]-1)
                init_links = torch.asarray(grid.init_links, copy=True)
                init_links[:gx_s,:,:] = -1
                init_links[gx_e:,:,:] = -1
                init_links[:,:gy_s,:] = -1
                init_links[:,gy_e:,:] = -1
                init_links[:,:,:gz_s] = -1
                init_links[:,:,gz_e:] = -1
                grid.links = init_links
                
            # if args.textOverlay and args.loss == 'l1':
            if args.loss == 'l1':
                rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                        beta_loss=args.lambda_beta,
                        sparsity_loss=args.lambda_sparsity,
                        randomize=args.enable_random,
                        is_rgb_gt = False)
            else:
                #  with Timing("volrend_fused"):
                rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                        beta_loss=args.lambda_beta,
                        sparsity_loss=args.lambda_sparsity,
                        randomize=args.enable_random)
                

            # 只对光度损失后向传播的梯度进行缩放
            if args.use_gradient_scaling:

                if args.dataset_type == "nerf":
                    IJK = (grid.links >= 0).nonzero()
                    gs_factors = grid.gs_grid[IJK[:,0], IJK[:,1], IJK[:,2]].unsqueeze(-1)
                    grid.density_data.grad[grid.links[IJK[:,0], IJK[:,1], IJK[:,2]].to(torch.long)] *= gs_factors
                    grid.sh_data.grad[grid.links[IJK[:,0], IJK[:,1], IJK[:,2]].to(torch.long)] *= gs_factors
                elif args.dataset_type == "llff" and  gstep_id < args.Ns:
                # elif args.dataset_type == "llff":
                    
                    IJK = (grid.links >= 0).nonzero()
                    gs_factors = grid.gs_grid[IJK[:,0], IJK[:,1], IJK[:,2]].unsqueeze(-1)
                    # grid.density_data.grad[grid.links[IJK[:,0], IJK[:,1], IJK[:,2]].to(torch.long)] *= gs_factors
                    grid.sh_data.grad[grid.links[IJK[:,0], IJK[:,1], IJK[:,2]].to(torch.long)] *= gs_factors
                    
                    
                    sigma_grid = torch.zeros_like(grid.gs_grid)
                    sigma_grid[grid.links>=0] = grid.density_data.data[grid.links[grid.links>=0].to(torch.long)].squeeze()
                    sigma_grid[sigma_grid<0] = 0.0
                    gs_grid = torch.cumsum(sigma_grid, dim=-1)
                    gs_grid_tran = torch.exp(-gs_grid*(1/(grid._scaling[-1]*grid.reso[-1])))
                    # gs_grid_alpha = 1. - torch.exp(-sigma_grid*(1/(grid._scaling[-1]*grid.reso[-1])))
                    # gs_grid[..., 1:] = (gs_grid_tran[..., :-1] * gs_grid_alpha[...,1:]) / (gs_grid_alpha[...,0:1] + 1e-6)
                    gs_grid[..., 1:] = gs_grid_tran[..., :-1]
                    gs_grid[..., 0] = 1.
                    grid.gs_grid = 1. / gs_grid
                    # gs_grid[..., 1:] = gs_grid_tran[..., :-1] - gs_grid_tran[..., 1:]
                    # gs_grid[..., 0] = 1. - gs_grid_tran[..., 0]
                    # gs_grid = gs_grid[..., :1] / gs_grid
                    
                    '''
                    with open(os.path.join(args.train_dir, 'max_gs.txt'), 'a') as max_gs_file:
                        max_gs_file.write(f"{epoch_id}|{gstep_id}|{grid.gs_grid.max().cpu().item():.3f}\n")
                    '''

            #  with Timing("loss_comp"):
            mse = F.mse_loss(rgb_gt, rgb_pred)


            # Stats
            mse_num : float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            
            # if args.gaussian_noise or args.textOverlay:
            '''
            if args.noise_aug:
                mse_clean = ((rgb_gt_clean - rgb_pred.detach())**2).mean()
                psnr_clean = -10.0 * math.log10(mse_clean)
                stats['psnr_clean'] += psnr_clean
            '''
            

            '''
            if args.vis_training and gstep_id % args.vis_training_every == 0:
                with torch.no_grad():
                    img_id = args.vis_img_id
                    c2w = dset.c2w[img_id].to(device=device)
                    cam = svox2.Camera(c2w,
                                    dset.intrins.get('fx', img_id),
                                    dset.intrins.get('fy', img_id),
                                    dset.intrins.get('cx', img_id),
                                    dset.intrins.get('cy', img_id),
                                    width=dset.get_image_size(img_id)[1],
                                    height=dset.get_image_size(img_id)[0],
                                    ndc_coeffs=dset.ndc_coeffs)
                    rgb_pred_train0 = grid.volume_render_image(cam, use_kernel=True)
                    rgb_pred_train0.clamp_(0.0, 1.0)
                    rgb_gt_train0 = dset.gt[img_id].to(device=device)

                    mse1 = (rgb_pred_train0 - rgb_gt_train0) ** 2
                    mse1_num : float = mse1.mean().item()
                    psnr = -10.0 * math.log10(mse1_num)
                    ssim = compute_ssim(rgb_gt_train0, rgb_pred_train0).item()

                    
                    rgb_pred_train0 = rgb_pred_train0.cpu().numpy()
                    cv2.putText(rgb_pred_train0, f"{img_id=:d}", (5, 20), 0, 0.6, [1, 0, 0])
                    cv2.putText(rgb_pred_train0, f"steps={gstep_id}", (5, 40), 0, 0.6, [1, 0, 0])
                    cv2.putText(rgb_pred_train0, f"{psnr=:.2f}", (5, 60), 0, 0.6, [1, 0, 0])
                    cv2.putText(rgb_pred_train0, f"{ssim=:.3f}", (5, 80), 0, 0.6, [1, 0, 0])
                    depth_pred_train0 = grid.volume_render_depth_image(cam)
                    depth_pred_train00 = depth_pred_train0.cpu().numpy()
                    depth_pred_train00 = viridis_cmap(depth_pred_train00)
                    rgb_depth = np.concatenate([rgb_pred_train0, depth_pred_train00], axis=1)
                    rgb_depth = (rgb_depth * 255).astype(np.uint8)
                    frames.append(rgb_depth)
            '''
            

            if (iter_id + 1) % args.print_every == 0:
                # Print averaged stats
                # if args.gaussian_noise or args.textOverlay:
                # if args.noise_aug:
                    # pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f} psnr_clean={psnr_clean:.2f}')
                # else:
                pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f}')
                for stat_name in stats:
                    stat_val = stats[stat_name] / args.print_every
                    summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    stats[stat_name] = 0.0
                #  if args.lambda_tv > 0.0:
                #      with torch.no_grad():
                #          tv = grid.tv(logalpha=args.tv_logalpha, ndc_coeffs=dset.ndc_coeffs)
                #      summary_writer.add_scalar("loss_tv", tv, global_step=gstep_id)
                #  if args.lambda_tv_sh > 0.0:
                #      with torch.no_grad():
                #          tv_sh = grid.tv_color()
                #      summary_writer.add_scalar("loss_tv_sh", tv_sh, global_step=gstep_id)
                #  with torch.no_grad():
                #      tv_basis = grid.tv_basis() #  summary_writer.add_scalar("loss_tv_basis", tv_basis, global_step=gstep_id)
                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                if grid.use_background:
                    summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                    summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

                if args.weight_decay_sh < 1.0:
                    grid.sh_data.data *= args.weight_decay_sigma
                if args.weight_decay_sigma < 1.0:
                    grid.density_data.data *= args.weight_decay_sh

            ## For outputting the % sparsity of the gradient
            if args.log_grad_sparsity:
                indexer = grid.sparse_grad_indexer
                if indexer is not None:
                    if indexer.dtype == torch.bool:
                        nz = torch.count_nonzero(indexer)
                    else:
                        nz = indexer.size()
                    with open(os.path.join(args.train_dir, 'sigma_grad_sparsity.txt'), 'a') as sparsity_file:
                        sparsity_file.write(f"{epoch_id} {gstep_id} {nz} {grid.density_data.data.size(0)}\n")
                indexer = grid.sparse_sh_grad_indexer
                if indexer is not None:
                    if indexer.dtype == torch.bool:
                        nz = torch.count_nonzero(indexer)
                    else:
                        nz = indexer.size()
                    with open(os.path.join(args.train_dir, 'sh_grad_sparsity.txt'), 'a') as sparsity_file:
                        sparsity_file.write(f"{epoch_id} {gstep_id} {nz} {grid.sh_data.data.size(0)}\n")
                
            if args.log_maxmin_density:
                max_density, max_pointer = grid.density_data.data.max(dim=0)
                min_density, min_pointer = grid.density_data.data.min(dim=0)
                max_indice = (grid.links == max_pointer).nonzero().to(torch.int32).squeeze().cpu().numpy().tolist()
                min_indice = (grid.links == min_pointer).nonzero().to(torch.int32).squeeze().cpu().numpy().tolist()
                with open(os.path.join(args.train_dir, 'max_density.txt'), 'a') as max_density_file:
                    max_density_file.write(f"{epoch_id}|{gstep_id}|{max_density.item():.3f}|{max_indice}\n")
                with open(os.path.join(args.train_dir, 'min_density.txt'), 'a') as min_density_file:
                    min_density_file.write(f"{epoch_id}|{gstep_id}|{min_density.item():.3f}|{min_indice}\n")
                



            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(grid.density_data.grad,
                        scaling=args.lambda_tv,
                        sparse_frac=args.tv_sparsity,
                        logalpha=args.tv_logalpha,
                        ndc_coeffs=dset.ndc_coeffs,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
                grid.inplace_tv_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_sh,
                        sparse_frac=args.tv_sh_sparsity,
                        ndc_coeffs=dset.ndc_coeffs,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_lumisphere > 0.0:
                grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_lumisphere,
                        dir_factor=args.tv_lumisphere_dir_factor,
                        sparse_frac=args.tv_lumisphere_sparsity,
                        ndc_coeffs=dset.ndc_coeffs)
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_l2_sh)
            if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                grid.inplace_tv_background_grad(grid.background_data.grad,
                        scaling=args.lambda_tv_background_color,
                        scaling_density=args.lambda_tv_background_sigma,
                        sparse_frac=args.tv_background_sparsity,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()
            #  print('nz density', torch.count_nonzero(grid.sparse_grad_indexer).item(),
            #        ' sh', torch.count_nonzero(grid.sparse_sh_grad_indexer).item())

            if args.sh_basis_incre:
                
                sh_basis_used = min(max(math.floor((gstep_id/args.Nf)*args.sh_dim), 1), args.sh_dim)
                sh_basis_used = max(math.floor(math.sqrt(sh_basis_used)), 1) ** 2
                summary_writer.add_scalar("sh_basis_num", sh_basis_used, global_step=gstep_id)
                if sh_basis_used < args.sh_dim:
                    grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,sh_basis_used:] = 0.0
                
                
                # grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,args.used_sh_dim:] = 0.0
                
                '''
                if epoch_id <= sh_incre_ids[0]:
                    grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,degrees[0]:] = 0.0
                elif epoch_id <= sh_incre_ids[1]:
                    # grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,:degrees[0]] = 0.0
                    grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,degrees[1]:] = 0.0
                # elif epoch_id <= 9:
                    # grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,:degrees[0]] = 0.0
                    # grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,degrees[2]:] = 0.0
                '''  


            # like coarse stage in DVGO
            if args.low_freq_init:
                if epoch_id < args.N_init:
                    grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,1:] = 0.0
                else:
                    grid.sh_data.grad.reshape(-1,3,args.sh_dim)[...,0:1] = 0.0
                    grid.density_data.grad[:] = 0.0

            
            
            # Manual SGD/rmsprop step
            if gstep_id >= args.lr_fg_begin_step:
                grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            if grid.use_background:
                grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
            if gstep_id >= args.lr_basis_begin_step:
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                    optim_basis_mlp.step()
                    optim_basis_mlp.zero_grad()

    # links = torch.asarray(grid.links, copy=True)
    train_step()
    # if torch.all(grid.links == links):
        # print(f"links not changed after {epoch_id}")
    # else:
        # print(f"links changed after {epoch_id}")
        # break
    gc.collect()
    gstep_id_base += batches_per_epoch

    #  ckpt_path = path.join(args.train_dir, f'ckpt_{epoch_id:05d}.npz')
    # Overwrite prev checkpoints since they are very huge
    if args.save_every > 0 and (epoch_id + 1) % max(
            factor, args.save_every) == 0 and not args.tune_mode:
        print('Saving', ckpt_path)
        grid.save(ckpt_path)
    '''
    if args.low_freq_init and epoch_id == args.N_init-1:
        ckpt_sh1_path = path.join(args.train_dir, 'ckpt_sh1.npz')
        print('Saving', ckpt_sh1_path)
        grid.save(ckpt_sh1_path)
    
    if epoch_id == 14:
        ckpt_sh1_path = path.join(args.train_dir, 'ckpt_epoch15.npz')
        print('Saving', ckpt_sh1_path)
        grid.save(ckpt_sh1_path)
    '''
    if (epoch_id + 1 - last_prune_epoch) >= args.prune_every and args.use_prune:
        last_prune_epoch = epoch_id + 1
        print("Eval before pruning")
        eval_step()
        if args.n_train != None:
            eval_train_step()
        gc.collect()
        print("* Pruning...")
        use_sparsify = True
        grid.resample(reso=grid.reso,
                    sigma_thresh=args.density_thresh,
                    # weight_thresh=args.weight_thresh / grid.reso[2] if use_sparsify else 0.0,
                    weight_thresh=args.weight_thresh if use_sparsify else 0.0,
                    dilate=args.dilate, #use_sparsify,
                    cameras=resample_cameras if args.thresh_type == 'weight' else None,
                    accelerate = not args.noacc,
                    max_elements=args.max_grid_elements,
                    save_dir = args.train_dir,
                    step = gstep_id_base)
        # only prune once
        args.use_prune = False

    # if (gstep_id_base - last_upsamp_step) >= args.upsamp_every and args.use_upsample:
        # last_upsamp_step = gstep_id_base
    if (epoch_id + 1 - last_upsamp_epoch) >= args.upsamp_every and args.use_upsample:
        # last_upsamp_step = gstep_id_base
        last_upsamp_epoch = epoch_id + 1
        if reso_id < len(reso_list) - 1:
            print("Eval before upsampling")
            eval_step()
            if args.n_train != None:
                eval_train_step()
            gc.collect()
            # print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
            print(f'* Upsampling from \033[96m{reso_list[reso_id]}\033[0m to \033[96m{reso_list[reso_id + 1]}\033[0m')
            if args.tv_early_only > 0:
                print('turning off TV regularization')
                args.lambda_tv = 0.0
                args.lambda_tv_sh = 0.0
            elif args.tv_decay != 1.0:
                args.lambda_tv *= args.tv_decay
                args.lambda_tv_sh *= args.tv_decay

            reso_id += 1
            use_sparsify = True
            z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
            
            if args.upsample_no_prune:
                grid.upsample(reso=reso_list[reso_id],
                          use_sphere_bound = False,
                          accelerate = not args.noacc,
                          max_elements = args.max_grid_elements)
            else:
                grid.resample(reso=reso_list[reso_id],
                    sigma_thresh=args.density_thresh,
                    weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                    # weight_thresh=args.weight_thresh if use_sparsify else 0.0,
                    dilate=0 if args.first_upsample_dilate and reso_id>1 else args.dilate, #use_sparsify,
                    cameras=resample_cameras if args.thresh_type == 'weight' else None,
                    accelerate = not args.noacc,
                    max_elements=args.max_grid_elements,
                    save_dir = args.train_dir,
                    step = gstep_id_base)
            

            if grid.use_background and reso_id <= 1:
                grid.sparsify_background(args.background_density_thresh)

            if args.upsample_density_add:
                grid.density_data.data[:] += args.upsample_density_add

        if factor > 1 and reso_id < len(reso_list) - 1:
            print('* Using higher resolution images due to large grid; new factor', factor)
            factor //= 2
            dset.gen_rays(factor=factor)
            dset.shuffle_rays()

    # if gstep_id_base >= args.n_iters:
    if epoch_id + 1 >= args.n_epochs:
        print('* Final eval and save')
        eval_step()
        if args.n_train != None:
            epoch_id = epoch_id + 1
            eval_train_step()
        if args.vis_training and args.vis_training_every:
            vid_path = os.path.join(args.train_dir, 'vis_train.mp4')
            imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)
        if args.log_grad_sparsity:
            filepath = os.path.join(args.train_dir, 'sh_grad_sparsity.txt')
            with open(filepath, 'r') as fp:
                lines = fp.read().split('\n')[:-1]
                print(f'The total lines is {len(lines)}')
                data = []
                for line in lines:
                    id, n, N = line.split(' ')[1:]
                    data.append([int(id)+1, (int(n)/int(N))*100])
                data = np.array(data)
                xdata = data[:,0]
                ydata = data[:,1]
                fig, ax = plt.subplots()
                ax.plot(xdata, ydata)
                ax.set_xlabel('step_id')
                ax.set_xscale('log')
                ax.set_ylabel('sparsity gradient (% Voxels)')
                ax.grid(visible=True)
                fig.savefig(os.path.join(args.train_dir, 'grad_sparsity.png'))
                # plt.show()

        if args.log_maxmin_density:
            filepath1 = os.path.join(args.train_dir, 'max_density.txt')
            filepath2 = os.path.join(args.train_dir, 'min_density.txt')
            fp1 = open(filepath1, 'r')
            fp2 = open(filepath2, 'r')
            lines1 = fp1.read().split('\n')[:-1]
            lines2 = fp2.read().split('\n')[:-1]
            # print(f'The total lines is {len(lines)}')
            max_densities = []
            min_densities = []
            for line1, line2 in zip(lines1, lines2):
                id1, max_density = line1.split('|')[1:-1]
                id2, min_density = line2.split('|')[1:-1]
                max_densities.append([int(id1)+1, float(max_density)])
                min_densities.append([int(id2)+1, float(min_density)])
            fig0, ax0 = plt.subplots()
            ax0.plot(np.array(max_densities)[:,0], np.array(max_densities)[:,1])
            ax0.set_xlabel('step_id')
            ax0.set_ylabel('max_density')
            fig0.savefig(os.path.join(args.train_dir, 'max_density.png'))
            fig1, ax1 = plt.subplots()
            ax1.plot(np.array(min_densities)[:,0], np.array(min_densities)[:,1])
            ax1.set_xlabel('step_id')
            ax1.set_ylabel('min_density')
            fig1.savefig(os.path.join(args.train_dir, 'min_density.png'))
            fp1.close()
            fp2.close()

        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
        timings_file.write(f"{secs / 60}\n")
        timings_file.close()
        if not args.tune_nosave:
            grid.save(ckpt_path)
        break
