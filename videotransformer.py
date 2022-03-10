import numpy as np
import numbers

import torch
from torchvision.transforms import functional as F
    

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (T, C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return tensor.sub_(self.mean).div_(self.std)
            
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (cv2 Image): Image to be cropped.
        Returns:
            cv2 Image: Cropped image.
        """
        t, c, h, w = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, :, i:i+th, j:j+tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
