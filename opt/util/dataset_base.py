import torch
import torch.nn.functional as F
from typing import Union, Optional, List
from .util import select_or_shuffle_rays, Rays, Intrin
import numpy as np
from sys import platform
from PIL import Image, ImageDraw, ImageFont
import random
from string import ascii_letters
from torchvision.transforms import PILToTensor, ToPILImage


class DatasetBase:
    split: str
    permutation: bool
    epoch_size: Optional[int]
    n_images: int
    h_full: int
    w_full: int
    intrins_full: Intrin
    c2w: torch.Tensor  # C2W OpenCV poses
    gt: Union[torch.Tensor, List[torch.Tensor]]   # RGB images
    device : Union[str, torch.device]

    def __init__(self):
        self.ndc_coeffs = (-1, -1)
        self.use_sphere_bound = False
        self.should_use_background = True # a hint
        self.use_sphere_bound = True
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [1.0, 1.0, 1.0]
        self.permutation = False

    def _add_text_overlay(self, img, max_occupancy, seed=0):
        """Adds text overlay to images."""

        assert max_occupancy < 1, 'Text parameter is an occupancy probability'

        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            # serif = 'Times New Roman.ttf'
            serif = 'times.ttf'
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        # w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
        if seed:
            random.seed(seed)
            max_occupancy = max_occupancy
        else:
            max_occupancy = np.random.uniform(0, max_occupancy)
        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            # random text size
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            # random text length
            length = np.random.randint(10, 25)
            # random text 
            chars = ''.join(random.choice(ascii_letters) for i in range(length))
            # random color
            color = tuple(np.random.randint(0, 255, c))
            # random position
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break
        # imageio.imwrite('./data/train/crops/text_overlay.jpg', text_img)
        return text_img
    
    def add_text_noise(self, max_occupancy):
        if self.rays_init.gt_c == None:
            self.rays_init.gt_c = self.rays_init.gt.clone() 
        # imgs_tensor = self.rays_init.gt.reshape(-1, self.h, self.w, 3).permute(0,3,1,2)
        imgs_tensor = self.rays_init.gt_c.reshape(-1, self.h, self.w, 3).permute(0,3,1,2)
        to_pil_img = ToPILImage()
        to_tensor = PILToTensor()
        img_tensor_list = []
        for img_tensor in imgs_tensor:
            img_pil = to_pil_img(img_tensor)
            img_textOverlay = self._add_text_overlay(img_pil, max_occupancy)
            img_textOverlay_tensor = to_tensor(img_textOverlay).permute(1,2,0)
            img_tensor_list.append(img_textOverlay_tensor/255)
        self.rays_init.gt = torch.stack(img_tensor_list, dim=0).reshape(-1,3)

    def add_gaussian_noise(self, std):
        if self.rays_init.gt_c == None:
            self.rays_init.gt_c = self.rays_init.gt.clone() 
        imgs_tensor = self.rays_init.gt_c.reshape(-1, self.h, self.w, 3)
        rgb_noise_list = []
        for rgb_gt in imgs_tensor:
            s = random.uniform(0, std)
            noise = torch.normal(0, s, size=rgb_gt.shape)
            rgb_noise = torch.clamp((rgb_gt + noise), 0., 1.)
            rgb_noise_list.append(rgb_noise)
        self.rays_init.gt = torch.stack(rgb_noise_list, dim=0).reshape(-1,3)
        
    def get_sample(self, samples, count):
        if self.rays_init.gt_c == None:
            self.rays_init.gt_c = self.rays_init.gt.clone() 
        noisy_samples = samples[count]
        self.rays_init.gt = noisy_samples.reshape(-1,3)

    def shuffle_rays(self):
        """
        Shuffle all rays
        """
        if self.split == "train":
            del self.rays
            self.rays = select_or_shuffle_rays(self.rays_init, self.permutation,
                                               self.epoch_size, self.device)

    def gen_rays(self, factor=1, extra_pruned_rays=False, prune_factor=0.2):
        print(" Generating rays, scaling factor", factor)
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins = self.intrins_full.scale(1.0 / true_factor)
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
        )
        xx = (xx - self.intrins.cx) / self.intrins.fx
        yy = (yy - self.intrins.cy) / self.intrins.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True) # directions are normalized
        dirs = dirs.reshape(1, -1, 3, 1)
        del xx, yy, zz
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0] # transform directions from OpenCV to World

        if factor != 1:
            gt = F.interpolate(
                self.gt.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="area"
            ).permute([0, 2, 3, 1])
            gt = gt.reshape(self.n_images, -1, 3)
            if self.d.shape[0] != 0:
                if self.sfm.depth_source == 'DKMv3':
                    d = F.interpolate(
                        self.d.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="nearest"
                        ).permute([0, 2, 3, 1])
                else:
                    d = F.interpolate(
                        self.d.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="area"
                        ).permute([0, 2, 3, 1])
                d = d.reshape(self.n_images, -1, 1)
            if self.masks.shape[0] != 0:
                masks = F.interpolate(
                        self.masks.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="nearest"
                        ).permute([0, 2, 3, 1])
            if self.alpha_c:
                alphas = F.interpolate(
                    self.alphas.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="nearest")
                alphas = alphas.reshape(self.n_images, -1, 1)
        else:
            gt = self.gt.reshape(self.n_images, -1, 3)
            if self.d.shape[0] != 0:
                d = self.d.reshape(self.n_images, -1, 1)
            if self.masks.shape[0] != 0:
                masks = self.masks.reshape(self.n_images, -1, 1)
            if self.alpha_c:
                alphas = self.alphas.reshape(self.n_images, -1, 1)
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous()
        
        if extra_pruned_rays:
            prune_h = int(self.h * (1 + prune_factor))
            prune_w = int(self.w * (1 + prune_factor))
            prune_yy, prune_xx = torch.meshgrid(
            torch.arange(prune_h, dtype=torch.float32) + 0.5,
            torch.arange(prune_w, dtype=torch.float32) + 0.5,
            )
            prune_xx = (prune_xx - prune_w*0.5) / self.intrins.fx
            prune_yy = (prune_yy - prune_h*0.5) / self.intrins.fy
            prune_zz = torch.ones_like(prune_xx)
            prune_dirs = torch.stack((prune_xx, prune_yy, prune_zz), dim=-1)  # OpenCV convention
            prune_dirs /= torch.norm(prune_dirs, dim=-1, keepdim=True) # directions are normalized
            prune_dirs = prune_dirs.reshape(1, -1, 3, 1)
            del prune_xx, prune_yy, prune_zz
            prune_dirs = (self.c2w[:, None, :3, :3] @ prune_dirs)[..., 0].reshape(-1, prune_h, prune_w, 3) # transform directions from OpenCV to World
            prune_origins = self.c2w[:, None, :3, 3].expand(-1, prune_h * prune_w, -1).reshape(-1, prune_h, prune_w, 3)
            prune_idx = torch.ones(prune_h, prune_w) == 1.0
            h_start = (prune_h - self.h) // 2
            w_start = (prune_w - self.w) // 2
            prune_idx[h_start:h_start+self.h, w_start:w_start+self.w] = False
            prune_dirs = prune_dirs[:,prune_idx].reshape(-1,3).contiguous()
            prune_origins = prune_origins[:,prune_idx].reshape(-1,3).contiguous()
            
        
        if self.split == "train":
            origins = origins.view(-1, 3)
            dirs = dirs.view(-1, 3)
            gt = gt.reshape(-1, 3)
            if self.d.shape[0] != 0:
                d = d.reshape(-1, 1)
            if self.masks.shape[0] != 0:
                masks = masks.reshape(-1, 1)
            if self.alpha_c:
                alphas = alphas.reshape(-1, 1)

        self.rays_init = Rays(origins=origins, dirs=dirs, gt=gt, 
                              depths=d if self.d.shape[0] != 0 else None,
                              masks=masks if self.masks.shape[0] != 0 else None,
                              alphas=alphas if self.alpha_c else None,
                              prune_origins=prune_origins if extra_pruned_rays else None,
                              prune_dirs=prune_dirs if extra_pruned_rays else None,)
        self.rays = self.rays_init

    def get_image_size(self, i : int):
        # H, W
        if hasattr(self, 'image_size'):
            return tuple(self.image_size[i])
        else:
            return self.h, self.w

