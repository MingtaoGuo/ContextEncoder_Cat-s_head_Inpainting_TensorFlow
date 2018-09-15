from config import *
import numpy as np
from PIL import Image

def get_mask():
    Y = np.random.randint(0, MASK_W + 1)
    X = np.random.randint(0, MASK_H + 1)
    patch = np.ones([MASK_H, MASK_W])
    mask = np.zeros([IMG_H, IMG_W])
    mask[X:X+MASK_H, Y:Y+MASK_W] = patch
    if IMG_C == 3:
        mask = np.dstack((mask, mask, mask))
    return mask, X, Y

def get_patchs(batch, X, Y):
    return batch[:, X:X+MASK_H, Y:Y+MASK_W, :]

def read_img_and_crop(path):
    img = np.array(Image.open(path))
    if np.shape(img).__len__() < 3:
        img = np.dstack((img, img, img))
    h = img.shape[0]
    w = img.shape[1]
    if h > w:
        diff = h - w
        random_y = np.random.randint(0, diff)
        return img[random_y:random_y+w, :]
    elif h < w:
        diff = w - h
        random_x = np.random.randint(0, diff)
        return img[:, random_x:random_x+h]
    else:
        return img
