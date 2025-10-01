import cv2
import torch.nn.functional as F 
from torchvision.transforms import Resize
from torch import tensor
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import json
# from random import shuffle, choice, sample


# with open('classes.txt') as f:
#     class_names= f.read().split("\n")

class_names = ['background', 'bati', 'line']

id_to_label= {i: cls for i, cls in enumerate(class_names)}
N_LABELS= len(class_names)



def channel_to_contours(channel):
    kernel = np.ones((3,3), np.uint8)  # 3x3 square kernel
    binary= cv2.erode(channel.copy(), kernel=kernel, iterations=4)
    binary= cv2.dilate(binary.copy(), kernel=kernel, iterations=4)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    return contours

def contour_to_vgg_region(contour, label):
    Xs, Ys= contour.squeeze().transpose(1,0)
    region= {
            'region_attributes': 
                {
                    'label': label
                },
            'shape_attributes': 
                {
                    'name': 'polygon',
                    "all_points_x": Xs.tolist(),
                    'all_points_y': Ys.tolist()
                }
        }
    return region


def mask_to_channels(mask, n_classes= 12,  imsize= 1024):
    resizer= Resize(imsize)
    channels= resizer(F.one_hot(tensor(mask).long(), n_classes).permute(2,0,1)).cpu().numpy().astype('uint8')
    return channels


def mask_to_vgg(mask, fn, id_to_label, n_classes= 12, imsize=1024):
    channels= mask_to_channels(mask, n_classes=n_classes, imsize=imsize)
    file= {
        'fileref': '',
        'filename': fn,
        'size': '',
        'base64_img_data': '',
        'file_attributes': {},
        'regions': {}
    }
    k=0
    for i, channel in enumerate(channels):
        if i == 0:
            continue
        contours= channel_to_contours(channel)
        label= id_to_label[i]
        
        for contour in contours:
            file['regions'][f'{k}']= contour_to_vgg_region(contour, label)
            k+= 1
    return file

