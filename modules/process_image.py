import torch
import numpy as np


def crop_image(top, left, new_h, new_w, img=None):
    b, c, h, w = img.shape
    tmp_img = img[:, :, top: top + new_h, left: left + new_w]
    return tmp_img


class RandCrop(object):
    def __init__(self, patch_size, num_crop):
        self.patch_size = patch_size
        self.num_crop = num_crop

    def __call__(self, sample):
        # r_img : C x H x W
        r_img = sample['ref_data']
        d_img = sample['file_data']

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        ret_r_img = torch.zeros((c, self.patch_size, self.patch_size))
        ret_d_img = torch.zeros((c, self.patch_size, self.patch_size))
        for _ in range(self.num_crop):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            tmp_r_img = r_img[:, top: top + new_h, left: left + new_w]
            tmp_d_img = d_img[:, top: top + new_h, left: left + new_w]
            ret_r_img = ret_r_img + tmp_r_img
            ret_d_img = ret_d_img + tmp_d_img
        ret_r_img /= self.num_crop
        ret_d_img /= self.num_crop

        sample['ref_data'] = ret_r_img
        sample['file_data'] = ret_d_img

        return sample


def five_point_crop(idx, d_img, r_img, crop_size):
    new_h = crop_size
    new_w = crop_size
    if len(d_img.shape) == 3:
        c, h, w = d_img.shape
    else:
        b, c, h, w = d_img.shape
    center_h = h // 2
    center_w = w // 2
    if idx == 0:
        top = 0
        left = 0
    elif idx == 1:
        top = 0
        left = w - new_w
    elif idx == 2:
        top = h - new_h
        left = 0
    elif idx == 3:
        top = h - new_h
        left = w - new_w
    elif idx == 4:
        top = center_h - new_h // 2
        left = center_w - new_w // 2
    elif idx == 5:
        left = 0
        top = center_h - new_h // 2
    elif idx == 6:
        left = w - new_w
        top = center_h - new_h // 2
    elif idx == 7:
        top = 0
        left = center_w - new_w // 2
    elif idx == 8:
        top = h - new_h
        left = center_w - new_w // 2

    if len(d_img.shape) == 3:
        d_img_org = d_img[:, top: top + new_h, left: left + new_w]
        r_img_org = r_img[:, top: top + new_h, left: left + new_w]
    else:
        d_img_org = d_img[:, :, top: top + new_h, left: left + new_w]
        r_img_org = r_img[:, :, top: top + new_h, left: left + new_w]
    return d_img_org, r_img_org


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        r_img = sample['ref_data']
        d_img = sample['file_data']

        prob_lr = np.random.random()
        # np.fliplr needs HxWxC
        if prob_lr > 0.5:
            d_img = torch.flip(d_img, dims=[-1]).clone()
            r_img = torch.flip(r_img, dims=[-1]).clone()

        sample['ref_data'] = r_img
        sample['file_data'] = d_img
        return sample
