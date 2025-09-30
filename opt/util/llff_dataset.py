# LLFF-format Forward-facing dataset loader
# Please use the LLFF code to run COLMAP & convert 
#
# Adapted from NeX data loading code (NOT using their hand picked bounds)
# Entry point: LLFFDataset
#
# Original:
# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License

#from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import struct
import json
import glob
import copy

import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import imageio
import cv2
import pyexr
from .util import Rays, Intrin
from .dataset_base import DatasetBase
from .load_llff import load_llff_data
from typing import Union, Optional, List

from svox2.utils import convert_to_ndc

import util.myutils

class LLFFDataset(DatasetBase):
    """
    LLFF dataset loader adapted from NeX code
    Some arguments are inherited from them and not super useful in our case
    """
    def __init__(
        self,
        root : str,
        split : str,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        permutation: bool = True,
        factor: int = 1,
        ref_img: str="",
        scale : Optional[float]=1.0/4.0,  # 4x downsample
        dmin : float=-1,
        dmax : int=-1,
        invz : int= 0,
        transform=None,
        render_style="",
        hold_every=8,
        depth_source="",
        offset=250,
        hardcode_train_views: List[int] = None,
        n_images: int=None,
        **kwargs
    ):
        super().__init__()
        if scale is None:
            scale = 1.0 / 4.0  # Default 1/4 size for LLFF data since it's huge
        self.scale = scale
        self.dataset = root
        self.epoch_size = epoch_size
        self.device = device
        self.permutation = permutation
        self.split = split
        self.transform = transform
        self.sfm = SfMData(
            root,
            split,
            ref_img=ref_img,
            dmin=dmin,
            dmax=dmax,
            invz=invz,
            scale=scale,
            render_style=render_style,
            offset=offset,
            hold_every=hold_every,
            depth_source=depth_source,
        )

        assert len(self.sfm.cams) == 1, \
                "Currently assuming 1 camera for simplicity, " \
                "please feel free to extend"

        self.imgs = []
        is_train_split = split.endswith('train')
        for i, ind in enumerate(self.sfm.imgs):
            img = self.sfm.imgs[ind]
            img_train_split = ind % hold_every > 0
            if is_train_split == img_train_split:
                self.imgs.append(img)
        self.is_train_split = is_train_split
        if self.is_train_split:
            all_ids = set(range(0, len(self.sfm.imgs)))
            test_ids = set(range(0,len(self.sfm.imgs),hold_every))
            self.ids = sorted(list(all_ids - test_ids))
            if hardcode_train_views != None:
                imgs = []
                for id in hardcode_train_views:
                    imgs.append(self.sfm.imgs[id])
                self.imgs = imgs
                self.ids = hardcode_train_views
            elif n_images != None and n_images < len(self.ids):
                rngstate = np.random.get_state()
                np.random.seed(0)
                ids = np.random.choice(self.ids, n_images, replace=False)
                print(f'Sparse inputs idx \033[96m{ids}\033[0m') # 对所有图像进行编号
                imgs = []
                for id in ids:
                    # imgs.append(self.imgs[id-id//8-1]) 
                    imgs.append(self.sfm.imgs[id])
                self.imgs = imgs
                self.ids = ids # cams改了吗？ cams就包含在self.imgs里
                np.random.set_state(rngstate)
        else:
            self.ids = range(0,len(self.sfm.imgs),hold_every)

        self.alpha_c = False
        self._load_images()

        '''
        os.makedirs('inputs', exist_ok=True)
        for id in range(len(self.ids)):
            imageio.imwrite(f'inputs\\img_{self.ids[id]}.png', (self.gt[id]*255).numpy().astype(np.uint8))
            pyexr.write(f'inputs\\depth_{self.ids[id]}.exr', self.d[id].numpy())
        '''
        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        assert self.h_full == self.sfm.ref_cam["height"]
        assert self.w_full == self.sfm.ref_cam["width"]

        self.intrins_full = Intrin(self.sfm.ref_cam['fx'],
                                   self.sfm.ref_cam['fy'],
                                   self.sfm.ref_cam['px'],
                                   self.sfm.ref_cam['py'])
        
        '''
        if depth_source == 'depthAnything':
            if self.d.shape[0] != 0:
                super().gen_rays(factor)
                points = self.rays.origins + self.rays.dirs*self.rays.depths
                z_depths = points[...,2]
                sc = 1.0 / (z_depths.min() * 0.75)
                self.d *= sc
                # self.c2w *= sc # 对底下的self.gen_rays有影响 这行代码有误
                self.c2w[:, :3, 3] *= sc
            print(f'Bounds from estimated depths from {depth_source} after depth scaling : \033[96m[{self.d.min()}, {self.d.max()}]\033[0m')
        '''

        self.ndc_coeffs = (2 * self.intrins_full.fx / self.w_full,
                           2 * self.intrins_full.fy / self.h_full)
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins = self.intrins_full
        self.should_use_background = False  # Give warning  # no need of background for LLFF

    def z2abs_depth(self, z_depth):
        yy, xx = torch.meshgrid(
            torch.arange(z_depth.shape[0], dtype=torch.float32) + 0.5,
            torch.arange(z_depth.shape[1], dtype=torch.float32) + 0.5,
        )
        xx = (xx - self.sfm.ref_cam['px']) / self.sfm.ref_cam['fx']
        yy = (yy - self.sfm.ref_cam['py']) / self.sfm.ref_cam['fy']
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True) # directions are normalized
        abs_depth = z_depth / dirs[...,-1:]

        return abs_depth.numpy()

    
    def _load_images(self):
        scale = self.scale

        all_gt = []
        all_depths = []
        all_masks = []
        all_c2w = []
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        global_w2rc = np.concatenate([self.sfm.ref_img['r'], self.sfm.ref_img['t']], axis=1) # world to reference camera
        global_w2rc = np.concatenate([global_w2rc, bottom], axis=0).astype(np.float64)
        for idx in tqdm(range(len(self.imgs))):
            R = self.imgs[idx]["R"].astype(np.float64)
            t = self.imgs[idx]["center"].astype(np.float64)
            c2w = np.concatenate([R, t], axis=1)
            c2w = np.concatenate([c2w, bottom], axis=0)
            #  c2w = global_w2rc @ c2w
            all_c2w.append(torch.from_numpy(c2w.astype(np.float32)))

            if 'path' in self.imgs[idx]:
                img_path = self.imgs[idx]["path"]
                img_path = os.path.join(self.dataset, img_path)
                if not os.path.isfile(img_path):
                    path_noext = os.path.splitext(img_path)[0]
                    # Hack: also try png
                    if os.path.exists(path_noext + '.png'):
                        img_path = path_noext + '.png'
                img = imageio.imread(img_path)
                if scale != 1 and not self.sfm.use_integral_scaling:
                    h, w = img.shape[:2]
                    if self.sfm.dataset_type == "deepview":
                        newh = int(h * scale)  # always floor down height
                        neww = round(w * scale)
                    else:
                        newh = round(h * scale)
                        neww = round(w * scale)
                    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
                all_gt.append(torch.from_numpy(img))

            if 'd_path' in self.imgs[idx]:
                depth_path = self.imgs[idx]["d_path"]
                depth_path = os.path.join(self.dataset, depth_path)
                if self.sfm.depth_source == 'depthAnything':
                    depth = pyexr.read(depth_path) * 10. # 与原先的单位一致(分米)
                    depth = self.z2abs_depth(depth)
                    depth *= self.sfm.scaler
                    mask = depth > 0.
                    # print(f'Bounds from estimated depths after scaling: \033[96m[{depth.min()}, {depth.max()}]\033[0m')
                elif self.sfm.depth_source in ['DKMv3', 'VGGT', 'DUSt3R', 'colmap', 'depthAnything(aligned)']: # 稀疏深度
                    depth = pyexr.read(depth_path) # 这里的深度是缩放场景之后的深度，除了colmap
                    if self.sfm.depth_source == 'colmap' or self.sfm.depth_source == 'depthAnything(aligned)':
                        depth = self.z2abs_depth(depth) # z深度, depthAnything用colmap深度来对齐
                        depth = depth * self.sfm.scaler # colmap的深度是缩放场景之前的深度，单位为分米
                    mask = depth > 0.
                    depth[depth<0.] = 2. # 这里设置的值是任意的，后续会用掩码屏蔽掉
                elif self.sfm.depth_source == 'denseViews':
                     depth = pyexr.read(depth_path)

                if scale !=  1 and not self.sfm.use_integral_scaling_d:
                    h, w = depth.shape[:2]
                    if self.sfm_dataset_type == "deepview":
                        newh = int(h * scale)
                        neww = round(w * scale)
                    else:
                        newh = round(h * scale)
                        neww = round(w * scale)
                    if self.sfm.depth_source in ['DKMv3', 'VGGT', 'DUSt3R', 'colmap']:
                        # 稀疏深度用最邻近插值
                        depth = cv2.resize(depth, (neww, newh), interpolation=cv2.INTER_NEAREST)
                        mask = cv2.resize(mask, (neww, newh), interpolation=cv2.INTER_NEAREST)
                    else:
                        depth = cv2.resize(depth, (neww, newh), interpolation=cv2.INTER_AREA)
                all_depths.append(torch.from_numpy(depth))
                all_masks.append(torch.from_numpy(mask))

        self.gt = torch.stack(all_gt).float() / 255.0
        if self.gt.size(-1) == 4:
            # Apply alpha channel
            self.alpha_c = True
            self.alphas = self.gt[..., -1:]
            # the same as background_brightness
            self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
        if all_depths != []:
            self.d = torch.stack(all_depths)
            if self.sfm.depth_source == 'depthAnything':
                print(f'Bounds from estimated depths after bound scaling: \033[96m[{depth.min()}, {depth.max()}]\033[0m')
            # if self.sfm.depth_source in ['DKMv3', 'VGGT', 'DUSt3R', 'colmap']:
            if self.sfm.depth_source in ['depthAnything', 'DKMv3', 'VGGT', 'DUSt3R', 'colmap', 'depthAnything(aligned)']:
                self.masks = torch.stack(all_masks)
            else:
                self.masks = torch.tensor([])
        else:
            self.d = torch.tensor([])
            self.masks = torch.tensor([])

        self.c2w = torch.stack(all_c2w)
        bds_scale = 1.0
        # bds_scale = 1.0 if all_depths == [] else 1.0 / (self.d.min() / 0.75)
        self.z_bounds = [self.sfm.dmin * bds_scale, self.sfm.dmax * bds_scale]
        if bds_scale != 1.0:
            self.c2w[:, :3, 3] *= bds_scale
            if self.sfm.depth_source in ['depthAnything', 'DKMv3', 'VGGT', 'DUSt3R', 'colmap', 'depthAnything(aligned)']:
                self.d *= bds_scale
        HWF = [self.sfm.ref_cam['height'], self.sfm.ref_cam['width'], self.sfm.ref_cam['fx']]
        # util.myutils.vis_cams(self.c2w[:,:3].cpu().numpy(), HWF, 'svox2')
        # for pose in self.c2w:
            # util.myutils.vis_viewtransform(pose[:3,:], HWF)

        if not self.is_train_split:
            render_c2w = []
            for idx in tqdm(range(len(self.sfm.render_poses))):
                R = self.sfm.render_poses[idx]["R"].astype(np.float64)
                t = self.sfm.render_poses[idx]["center"].astype(np.float64)
                c2w = np.concatenate([R, t], axis=1)
                c2w = np.concatenate([c2w, bottom], axis=0)
                render_c2w.append(torch.from_numpy(c2w.astype(np.float32)))
            self.render_c2w = torch.stack(render_c2w)
            if bds_scale != 1.0:
                self.render_c2w[:, :3, 3] *= bds_scale

        fx = self.sfm.ref_cam['fx']
        fy = self.sfm.ref_cam['fy']
        width = self.sfm.ref_cam['width']
        height = self.sfm.ref_cam['height']

        print('z_bounds from LLFF:', self.z_bounds, '(not used)')

        # Padded bounds
        radx = 1 + 2 * self.sfm.offset / self.gt.size(2) # 默认半径为1
        rady = 1 + 2 * self.sfm.offset / self.gt.size(1)
        radz = 1.0
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [radx, rady, radz]
        print('scene_radius', self.scene_radius)
        self.use_sphere_bound = False

    def depth_cam2ndc(self, dir_mod):
        near = 1.0
        t = (near - self.rays.origins[Ellipsis, 2]) / self.rays.dirs[Ellipsis, 2]
        self.rays.origins += t[Ellipsis, None] * self.rays.dirs

        self.rays_init.depths -= t[Ellipsis, None] # 这里的深度被当做绝对深度（是否合理？）深度已经被绝对化
        self.rays_init.depths = 1. - self.rays.origins[Ellipsis, 2] / (self.rays.origins[Ellipsis, 2] + self.rays_init.depths.squeeze()*self.rays.dirs[Ellipsis, 2])

        self.rays_init.depths *= dir_mod.squeeze() # 这里乘以方向的模是因为方向被归一化
        self.rays_init.depths = self.rays_init.depths[Ellipsis, None]
    
    def depth_ndc2cam(self):
        # dirs = self.rays.dirs * self.dirs_mod
        depths_ndc = (self.rays.depths / self.dir_mod).squeeze()

        near = 1.0
        t = (near - self.rays_beforeNDC.origins[Ellipsis, 2]) / self.rays_beforeNDC.dirs[Ellipsis, 2]
        origins = self.rays_beforeNDC.origins + t[Ellipsis, None] * self.rays_beforeNDC.dirs

        depth_cam = origins[Ellipsis, 2] / self.rays_beforeNDC.dirs[Ellipsis, 2] * (1/(1-depths_ndc)-1) + t

        return depth_cam
        
    def gen_rays(self, factor=1):
        super().gen_rays(factor)
        # To NDC (currently, we are normalizing these rays unlike NeRF,
        #  may not be ideal)

        
        self.rays_beforeNDC = Rays(origins=self.rays.origins, dirs=self.rays.dirs, gt=self.rays.gt)

        origins, dirs = convert_to_ndc(
                self.rays.origins,
                self.rays.dirs,
                self.ndc_coeffs)
        
        dir_mod = torch.norm(dirs, dim=-1, keepdim=True)
        self.dir_mod = dir_mod
        dirs /= dir_mod
        
        if self.d.shape[0] != 0:
            if self.sfm.depth_source in ['depthAnything', 'DKMv3', 'VGGT', 'DUSt3R', 'colmap', 'depthAnything(aligned)']:
                self.depth_cam2ndc(dir_mod)


        self.rays_init = Rays(origins=origins, dirs=dirs, gt=self.rays.gt,
                              depths=self.rays_init.depths if self.d.shape[0] != 0 else None,
                              masks=self.rays_init.masks if self.masks.shape[0] != 0 else None,
                              alphas=self.rays_init.alphas if self.alpha_c else None)
        self.rays = self.rays_init



class SfMData:
    def __init__(
        self,
        root,
        split,
        ref_img="",
        scale=1,
        dmin=0,
        dmax=0,
        invz=0,
        render_style="",
        offset=200,
        hold_every=8,
        depth_source='',
    ):
        self.split = split
        self.scale = scale
        self.ref_cam = None
        self.ref_img = None
        self.render_poses = None
        self.dmin = dmin
        self.dmax = dmax
        self.invz = invz
        self.dataset = root
        self.dataset_type = "unknown"
        self.render_style = render_style
        self.hold_every = hold_every
        self.depth_source = depth_source
        self.white_background = False  # change background to white if transparent.
        self.index_split = []  # use for split dataset in blender
        self.offset = offset
        # Detect dataset type
        can_hanle = (
            self.readDeepview(root)  # support DeepView dataset
            or self.readLLFF(root, ref_img)
            or self.readColmap(root)  # support Colmap dataset
        )
        if not can_hanle:
            raise Exception("Unknow dataset type")
        # Dataset processing
        self.cleanImgs()
        self.selectRef(ref_img)
        self.scaleAll(scale)
        self.selectDepth(dmin, dmax, offset)

    def cleanImgs(self):
        """
        Remove non exist image from self.imgs
        """
        todel = []
        for image in self.imgs:
            img_path = self.dataset + "/" + self.imgs[image]["path"]
            if "center" not in self.imgs[image] or not os.path.exists(img_path):
                todel.append(image)
        for it in todel:
            del self.imgs[it]

    def selectRef(self, ref_img):
        """
        Select Reference image
        """
        if ref_img == "" and self.ref_cam is not None and self.ref_img is not None:
            return
        for img_id, img in self.imgs.items():
            if ref_img in img["path"]: # specify reference image
                self.ref_img = img
                self.ref_cam = self.cams[img["camera_id"]]
                return
        raise Exception("reference view not found")

    def selectDepth(self, dmin, dmax, offset):
        """
        Select dmin/dmax from planes.txt / bound.txt / argparse
        """
        if self.dmin < 0 or self.dmax < 0:
            if os.path.exists(self.dataset + "/bounds.txt"):
                with open(self.dataset + "/bounds.txt", "r") as fi:
                    data = [
                        np.reshape(np.matrix([float(y) for y in x.split(" ")]), [3, 1])
                        for x in fi.readlines()[3:]
                    ]
                ls = []
                for d in data:
                    v = self.ref_img["r"] * d + self.ref_img["t"]
                    ls.append(v[2])
                self.dmin = np.min(ls)
                self.dmax = np.max(ls)
                self.invz = 0

            elif os.path.exists(self.dataset + "/planes.txt"):
                with open(self.dataset + "/planes.txt", "r") as fi:
                    data = [float(x) for x in fi.readline().split(" ")]
                    if len(data) == 3:
                        self.dmin, self.dmax, self.invz = data
                    elif len(data) == 2:
                        self.dmin, self.dmax = data
                    elif len(data) == 4:
                        self.dmin, self.dmax, self.invz, self.offset = data
                        self.offset = int(self.offset)
                        print(f"Read offset from planes.txt: {self.offset}")
                    else:
                        raise Exception("Malform planes.txt")
            else:
                print("no planes.txt or bounds.txt found")
        if dmin > 0:
            print("Overriding dmin %f-> %f" % (self.dmin, dmin))
            self.dmin = dmin
        if dmax > 0:
            print("Overriding dmax %f-> %f" % (self.dmax, dmax))
            self.dmax = dmax
        if offset != 200:
            print(f"Overriding offset {self.offset}-> {offset}")
            self.offset = offset
        print(
            "dmin = %f, dmax = %f, invz = %d, offset = %d"
            % (self.dmin, self.dmax, self.invz, self.offset)
        )

    def readLLFF(self, dataset, ref_img=""):
        """
        Read LLFF
        Parameters:
          dataset (str): path to datasets
          ref_img (str): ref_image file name
        Returns:
          bool: return True if successful load LLFF data
        """
        if not os.path.exists(os.path.join(dataset, "poses_bounds.npy")):
            return False
        image_dir = os.path.join(dataset, "images")
        if not os.path.exists(image_dir) and not os.path.isdir(image_dir):
            return False
        # depth_dir = os.path.join(dataset, "depths")
        # has_depth = False
        # if os.path.exists(depth_dir):
            # has_depth = True

        self.use_integral_scaling = False
        self.use_integral_scaling_d = False
        scaled_img_dir = ''
        scale = self.scale
        has_depth = False
        if scale != 1 and abs((1.0 / scale) - round(1.0 / scale)) < 1e-9:
            # Integral scaling
            scaled_img_dir = "images_" + str(round(1.0 / scale))
            if os.path.isdir(os.path.join(self.dataset, scaled_img_dir)):
                self.use_integral_scaling = True
                image_dir = os.path.join(self.dataset, scaled_img_dir)
                print('Using pre-scaled images from', image_dir)
            else:
                scaled_img_dir = "images"

            scaled_depth_dir = "depths_" + str(round(1.0 / scale)) + (f'_{self.depth_source}' if self.depth_source else '')
            if os.path.isdir(os.path.join(self.dataset, scaled_depth_dir)):
                has_depth = True
                self.use_integral_scaling_d = True
                depth_dir = os.path.join(self.dataset, scaled_depth_dir)
                # self.depth_dir = depth_dir
                print('Using pre-scaled depths from', depth_dir)
                
        # load R,T
        (
            reference_depth,
            reference_view_id,
            render_poses,
            poses,
            intrinsic,
            scaler
        ) = load_llff_data(
            dataset, factor=None, split_train_val=self.hold_every,
            render_style=self.render_style
        )
        self.scaler = scaler

        # NSVF-compatible sort key
        def nsvf_sort_key(x):
            if len(x) > 2 and x[1] == '_':
                return x[2:]
            else:
                return x
        def keep_images(x):
            exts = ['.png', '.jpg', '.jpeg', '.exr']
            return [y for y in x if not y.startswith('.') and any((y.lower().endswith(ext) for ext in exts))] 

        # get all image of this dataset
        images_path = [os.path.join(scaled_img_dir, f) for f in sorted(keep_images(os.listdir(image_dir)), key=nsvf_sort_key)]
        if has_depth and self.split == 'train':
            if scaled_depth_dir.split('_')[-1] in ['DKMv3', 'VGGT', 'DUSt3R', 'colmap', 'depthAnything(aligned)']:
                depths_path = [os.path.join(scaled_depth_dir, f'depth_{i:03d}.exr') for i in range(len(images_path))]
            else:
                depths_path = [os.path.join(scaled_depth_dir, f) for f in sorted(keep_images(os.listdir(depth_dir)), key=nsvf_sort_key)]
        else:
            depths_path = None

        # LLFF dataset has only single camera in dataset
        if len(intrinsic) == 3:
            H, W, f = intrinsic
            cx = W / 2.0
            cy = H / 2.0
            fx = f
            fy = f
        else:
            H, W, fx, fy, cx, cy = intrinsic

        self.cams = {0: buildCamera(W, H, fx, fy, cx, cy)}

        # create render_poses for video render
        self.render_poses = buildNerfPoses(render_poses)

        # create imgs pytorch dataset
        # we store train and validation together
        # but it will sperate later by pytorch dataloader
        self.imgs = buildNerfPoses(poses, images_path, depths_path)

        # if not set ref_cam, use LLFF ref_cam
        if ref_img == "":
            # restore image id back from reference_view_id
            # by adding missing validation index
            image_id = reference_view_id + 1  # index 0 alway in validation set
            image_id = image_id + (image_id // self.hold_every)  # every 8 will be validation set
            self.ref_cam = self.cams[0] # reference intrinsic

            self.ref_img = self.imgs[image_id]  # here is reference view from train set

        # if not set dmin/dmax, use LLFF dmin/dmax
        if (self.dmin < 0 or self.dmax < 0) and (
            not os.path.exists(dataset + "/planes.txt")
        ):
            self.dmin = reference_depth[0]
            self.dmax = reference_depth[1]
        self.dataset_type = "llff"
        return True

    def scaleAll(self, scale):
        self.ocams = copy.deepcopy(self.cams)  # original camera
        for cam_id in self.cams.keys():
            cam = self.cams[cam_id]
            ocam = self.ocams[cam_id]

            nw = round(ocam["width"] * scale)
            nh = round(ocam["height"] * scale)
            sw = nw / ocam["width"]
            sh = nh / ocam["height"]
            cam["fx"] = ocam["fx"] * sw
            cam["fy"] = ocam["fy"] * sh
            # TODO: What is the correct way?
            #  cam["px"] = (ocam["px"] + 0.5) * sw - 0.5
            #  cam["py"] = (ocam["py"] + 0.5) * sh - 0.5
            cam["px"] = ocam["px"] * sw
            cam["py"] = ocam["py"] * sh
            cam["width"] = nw
            cam["height"] = nh

    def readDeepview(self, dataset):
        if not os.path.exists(os.path.join(dataset, "models.json")):
            return False

        self.cams, self.imgs = readCameraDeepview(dataset)
        self.dataset_type = "deepview"
        return True

    def readColmap(self, dataset):
        sparse_folder = dataset + "/dense/sparse/"
        image_folder = dataset + "/dense/images/"
        if (not os.path.exists(image_folder)) or (not os.path.exists(sparse_folder)):
            return False

        self.imgs = readImagesBinary(os.path.join(sparse_folder, "images.bin"))
        self.cams = readCamerasBinary(sparse_folder + "/cameras.bin")
        self.dataset_type = "colmap"
        return True


def readCameraDeepview(dataset):
    cams = {}
    imgs = {}
    with open(os.path.join(dataset, "models.json"), "r") as fi:
        js = json.load(fi)
        for i, cam in enumerate(js):
            for j, cam_info in enumerate(cam):
                img_id = cam_info["relative_path"]
                cam_id = img_id.split("/")[0]

                rotation = (
                    Rotation.from_rotvec(np.float32(cam_info["orientation"]))
                    .as_matrix()
                    .astype(np.float32)
                )
                position = np.array([cam_info["position"]], dtype="f").reshape(3, 1)

                if i == 0:
                    cams[cam_id] = {
                        "width": int(cam_info["width"]),
                        "height": int(cam_info["height"]),
                        "fx": cam_info["focal_length"],
                        "fy": cam_info["focal_length"] * cam_info["pixel_aspect_ratio"],
                        "px": cam_info["principal_point"][0],
                        "py": cam_info["principal_point"][1],
                    }
                imgs[img_id] = {
                    "camera_id": cam_id,
                    "r": rotation,
                    "t": -np.matmul(rotation, position),
                    "R": rotation.transpose(),
                    "center": position,
                    "path": cam_info["relative_path"],
                }
    return cams, imgs


def readImagesBinary(path):
    images = {}
    f = open(path, "rb")
    num_reg_images = struct.unpack("Q", f.read(8))[0]
    for i in range(num_reg_images):
        image_id = struct.unpack("I", f.read(4))[0]
        qv = np.fromfile(f, np.double, 4)

        tv = np.fromfile(f, np.double, 3)
        camera_id = struct.unpack("I", f.read(4))[0]

        name = ""
        name_char = -1
        while name_char != b"\x00":
            name_char = f.read(1)
            if name_char != b"\x00":
                name += name_char.decode("ascii")

        num_points2D = struct.unpack("Q", f.read(8))[0]

        for i in range(num_points2D):
            f.read(8 * 2)  # for x and y
            f.read(8)  # for point3d Iid

        r = Rotation.from_quat([qv[1], qv[2], qv[3], qv[0]]).as_dcm().astype(np.float32)
        t = tv.astype(np.float32).reshape(3, 1)

        R = np.transpose(r)
        center = -R @ t
        # storage is scalar first, from_quat takes scalar last.
        images[image_id] = {
            "camera_id": camera_id,
            "r": r,
            "t": t,
            "R": R,
            "center": center,
            "path": "dense/images/" + name,
        }

    f.close()
    return images


def readCamerasBinary(path):
    cams = {}
    f = open(path, "rb")
    num_cameras = struct.unpack("Q", f.read(8))[0]

    # becomes pinhole camera model , 4 parameters
    for i in range(num_cameras):
        camera_id = struct.unpack("I", f.read(4))[0]
        model_id = struct.unpack("i", f.read(4))[0]

        width = struct.unpack("Q", f.read(8))[0]
        height = struct.unpack("Q", f.read(8))[0]

        fx = struct.unpack("d", f.read(8))[0]
        fy = struct.unpack("d", f.read(8))[0]
        px = struct.unpack("d", f.read(8))[0]
        py = struct.unpack("d", f.read(8))[0]

        cams[camera_id] = {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "px": px,
            "py": py,
        }
        # fx, fy, cx, cy
    f.close()
    return cams


def nerf_pose_to_ours(cam):
    # 以下变换等价于
    # 1）先右乘旋转矩阵R将照相机模式从OpenGL变换为OpenCV
    # 2) 再左乘[R|center]变换世界坐标系
    R = cam[:3, :3]
    center = cam[:3, 3].reshape([3, 1])
    center[1:] *= -1
    R[1:, 0] *= -1
    R[0, 1:] *= -1

    r = np.transpose(R) # 对于旋转变换矩阵，转置与求逆等价
    t = -r @ center   # c2w --> w2c
    return R, center, r, t


def buildCamera(W, H, fx, fy, cx, cy):
    return {
        "width": int(W),
        "height": int(H),
        "fx": float(fx),
        "fy": float(fy),
        "px": float(cx),
        "py": float(cy),
    }


def buildNerfPoses(poses, images_path=None, depths_path=None):
    output = {}
    for poses_id in range(poses.shape[0]):
        R, center, r, t = nerf_pose_to_ours(poses[poses_id].astype(np.float32))
        output[poses_id] = {"camera_id": 0, "r": r, "t": t, "R": R, "center": center}
        if images_path is not None:
            output[poses_id]["path"] = images_path[poses_id]
        if depths_path is not None:
            output[poses_id]["d_path"] = depths_path[poses_id]

    return output
