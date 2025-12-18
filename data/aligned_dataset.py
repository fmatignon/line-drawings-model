import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert self.opt.load_size >= self.opt.crop_size  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain (with Canny edge as 4th channel)
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # Compute Canny edges from A (before transforms for better edge detection)
        A_np = np.array(A)
        A_gray = cv2.cvtColor(A_np, cv2.COLOR_RGB2GRAY)
        # Canny edge detection with thresholds
        canny_edges = cv2.Canny(A_gray, threshold1=50, threshold2=150)
        # Normalize to [0, 1] and convert to PIL Image
        canny_edges = Image.fromarray(canny_edges).convert("L")

        # apply the same transform to both A, B, and Canny edges
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=False)  # Keep RGB, use BICUBIC for photos
        # Use NEAREST interpolation for B (line drawings) to avoid anti-aliasing
        B_transform = get_transform(
            self.opt, transform_params, grayscale=(self.output_nc == 1),
            method=transforms.InterpolationMode.NEAREST
        )
        # Use NEAREST for Canny edges too (they're binary)
        canny_transform = get_transform(
            self.opt, transform_params, grayscale=True,
            method=transforms.InterpolationMode.NEAREST
        )

        A = A_transform(A)  # [3, H, W] in range [-1, 1]
        B = B_transform(B)
        canny = canny_transform(canny_edges)  # [1, H, W] in range [-1, 1]

        # Concatenate A (3 channels) with Canny (1 channel) to get 4 channels
        A = torch.cat([A, canny], dim=0)  # [4, H, W]

        return {"A": A, "B": B, "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
