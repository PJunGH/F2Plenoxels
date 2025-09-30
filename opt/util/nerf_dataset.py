# Standard NeRF Blender dataset loader
from .util import Rays, Intrin, select_or_shuffle_rays
from .dataset_base import DatasetBase
import torch
import torch.nn.functional as F
from typing import NamedTuple, List, Optional, Union
from os import path
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np
import pyexr
import os


class NeRFDataset(DatasetBase):
    """
    NeRF dataset loader

    WARNING: this is only intended for use with NeRF Blender data!!!!
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,
        factor: int = 1,
        scale : Optional[float] = None, # downsampling
        permutation: bool = True,
        white_bkgd: bool = True,
        grey_bkgd: bool = False,
        hardcode_train_views: List[int] = None,
        n_images = None,
        skip = 1,
        depth_source = None,
        use_vggt_intri = False,
        use_zdepth = False,
        extra_pruned_rays = False,
        prune_factor = 0.2,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 2/3 # scale scene by putting camera near or far
        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        all_c2w = []
        all_gt = []
        all_d = []

        split_name = split if split != "test_train" else "train"
        data_path = path.join(root, split_name)
        data_json = path.join(root, "transforms_" + split_name + ".json")

        print("LOAD DATA", data_path)
        print("WARNING: This data loader is ONLY intended for use with NeRF-synthetic Blender data!!!!")
        print("If you want to try running this code on Instant-NGP data please use scripts/ingp2nsvf.py")

        j = json.load(open(data_json, "r"))

        # OpenGL -> OpenCV
        cam_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        for frame in tqdm(j["frames"]):
            fpath = path.join(data_path, path.basename(frame["file_path"]) + ".png")
            if 'depth_path' in frame.keys():
                dpath = path.join(data_path, path.basename(frame["depth_path"]) + ".exr")
            else:
                dpath = ''
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            c2w = c2w @ cam_trans  # To OpenCV

            im_gt = imageio.imread(fpath)
            if dpath:
                im_d = pyexr.read(dpath)

            if scale < 1.0:
                full_size = list(im_gt.shape[:2])
                rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
                im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA) # the same as dietNerf
                if dpath:
                    im_d = cv2.resize(im_d, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

            all_c2w.append(c2w)
            all_gt.append(torch.from_numpy(im_gt))
            if dpath:
                all_d.append(torch.from_numpy(im_d))
                
        focal = float(
            0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"])
        ) #  1250.0 if scene in ['drums', 'materials'] else 1111.11

        if depth_source is not None and use_vggt_intri:
            vggt_intri_path = path.join(root, "vggt_intri.txt")
            fxy = open(vggt_intri_path, 'r').read().strip().split(',')
            for f in fxy:
                key, value = f.strip().split(' ')
                if key == 'fx':
                    fx = float(value)
                if key == 'fy':
                    fy = float(value)
            

        self.c2w = torch.stack(all_c2w)
        self.c2w[:, :3, 3] *= scene_scale # scale scene by putting camera near or far

        self.gt = torch.stack(all_gt).float() / 255.0
        if split == 'train':
            half_name = None
            if path.isfile(path.join(root, 'mean_half.npy')):
                half_name = 'mean_half.npy'
            elif path.isfile(path.join(root, 'median_half.npy')):
                half_name = 'median_half.npy'

            if half_name != None:
                print(f'Load images from {half_name}')
                gt_half = np.load(path.join(root, f'{half_name}'))
                self.gt = torch.from_numpy(gt_half)

        if all_d:
            self.d = torch.stack(all_d)
            assert self.d.shape[0] == self.gt.shape[0], "The number of depth images is unmatched with that of groundtruths"
        else:
            self.d = torch.tensor([])
        self.ids = np.arange(len(self.gt))
        self.alpha_c = False
        if self.gt.size(-1) == 4:
            self.alpha_c = True
            self.alphas = self.gt[..., -1:]
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                if grey_bkgd:
                    self.gt = self.gt[..., :3] * self.gt[..., 3:] + 0.5 * (1.0 - self.gt[..., 3:])
                else:
                    self.gt = self.gt[..., :3]

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        # Choose a subset of training images
        
        self.masks = torch.tensor([])
        if n_images is not None:
            if n_images > self.n_images:
                print(f'using {self.n_images} available training views instead of the requested {n_images}.')
                n_images = self.n_images
            self.n_images = n_images
            rngstate = np.random.get_state()
            # the same as dietnerf
            if hardcode_train_views != None:
                ids = np.asarray(hardcode_train_views)
            else:
                if n_images == 8:
                    np.random.seed(0)
                    ids = np.random.choice(len(self.gt), self.n_images, replace=False)
                # the same as Infonerf
                if n_images == 4:
                    np.random.seed(0)
                    ids = np.random.choice(len(self.gt), self.n_images, replace=False)
            print(f'Use images with IDs \033[96m{ids}\033[0m (counting from 0) for few-shot training.')
            self.ids = ids
            self.gt = self.gt[ids,...]
            self.c2w = self.c2w[ids,...]
            if self.d.shape[0] != 0:
                self.d = self.d[ids,...]
            if depth_source != None:
                depth_sources = depth_source.split('+')
                if depth_sources[0] in ["DKMv3", "VGGT"]:
                    all_d = []
                    all_masks = []
                    for id in ids:
                        if depth_sources[0] == "VGGT" and use_vggt_intri:
                            # depth = pyexr.read(os.path.join(root, f"depth_{depth_sources[0]}", f"depth_{id:03d}_oriIntri.exr"))
                            depth = pyexr.read(os.path.join(root, f"depth_{depth_sources[0]}", f"depth_{id:03d}.exr"))
                        else:
                            if use_zdepth:
                                zdepth = pyexr.read(os.path.join(root, f"depth_{depth_sources[0]}", f"zdepth_{id:03d}.exr"))
                                zdepth = zdepth.squeeze()
                                zmask = zdepth > 0.
                                x, y = np.meshgrid(np.arange(self.w_full), np.arange(self.h_full))
                                x = (x - self.w_full / 2) / focal
                                y = (y - self.h_full / 2) / focal
                                points = np.stack((np.multiply(x, zdepth), np.multiply(y, zdepth), zdepth), axis=-1) # OpenCV convention
                                depth = np.linalg.norm(points, axis=-1)
                                depth[~zmask] = -1.
                            else:
                                depth = pyexr.read(os.path.join(root, f"depth_{depth_sources[0]}", f"depth_{id:03d}.exr"))
                        mask = depth > 0.
                        depth = scene_scale * depth
                        all_d.append(torch.from_numpy(depth))
                        all_masks.append(torch.from_numpy(mask))
                    self.d = torch.stack(all_d)
                    self.masks = torch.stack(all_masks)
                    if len(depth_sources) == 2:
                        if depth_sources[1] in ["depthAnything_aligned", "VGGT_aligned"]:
                            for id in ids:
                                depth = pyexr.read(os.path.join(root, f"depth_{depth_sources[1]}", f"r_{id}_aligned.exr"))
                                depth = scene_scale * depth
                                depth = torch.from_numpy(depth)
                                if self.alpha_c:
                                    mask = torch.logical_and((self.alphas[id] == 1.0), ~self.masks[id])
                                else:
                                    mask = ~self.masks[id]
                                self.d[id][mask] = depth[mask]
                        else:
                            raise NotImplementedError
            if self.alpha_c:
                self.alphas = self.alphas[ids,...]
            np.random.set_state(rngstate)
            # self.gt = self.gt[0:n_images,...]
            # self.c2w = self.c2w[0:n_images,...]

        # debugging
        # input_dir = '.\\inputs'
        # os.makedirs(input_dir, exist_ok=True)
        # for img_id in range(self.gt.shape[0]):
            # img = (self.gt[img_id].numpy()*255).astype(np.uint8)
            # imageio.imwrite(input_dir+'\\'+f'{self.ids[img_id]}.png', img)
            # depth = self.d[img_id].numpy()
            # pyexr.write(input_dir+'\\'+f'{self.ids[img_id]}.exr', depth)

        if split == 'test' and skip != 1:
            print(f'Select images for test every {skip}.')
            self.ids = np.arange(len(self.gt))[::skip]
            self.gt = self.gt[::skip]
            self.c2w = self.c2w[::skip]
            self.n_images = self.gt.size(0)
        
        if depth_source is not None and use_vggt_intri:
            self.intrins_full : Intrin = Intrin(fx, fy,
                                            self.w_full * 0.5,
                                            self.h_full * 0.5)
        else:
            self.intrins_full : Intrin = Intrin(focal, focal,
                                            self.w_full * 0.5,
                                            self.h_full * 0.5)

        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor, extra_pruned_rays=extra_pruned_rays, prune_factor=prune_factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

        self.should_use_background = False  # Give warning; No background model for nerf-synthetic dataset

