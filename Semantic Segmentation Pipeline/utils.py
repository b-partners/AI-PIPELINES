import hashlib
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os, glob
from random import choice
import matplotlib.patches as mpatches
from collections import defaultdict


def int_to_hex6(num: int) -> str:
    """Hash an integer ≥ 0 and return a 6-digit hexadecimal string."""
    if num < 0:
        raise ValueError("Number must be ≥ 0")

    # Convert integer to a SHA256 hash, then take the first 3 bytes (6 hex digits)
    hex_hash = hashlib.sha256(str(num).encode()).hexdigest()[:6]
    
    return hex_hash.upper()  # Return uppercase hex (optional)

def hex6_to_rgb(hex_str: str) -> tuple:
    """Convert a 6-digit hexadecimal string to an RGB tuple."""
    if len(hex_str) != 6:
        raise ValueError("Hex string must be exactly 6 characters long")
    
    # Convert hex pairs to integers (R, G, B)
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    
    return np.array((r, g, b))

# Example usage:
def id_to_color(cls_id: int):
    hex_color= int_to_hex6(cls_id)
    return hex6_to_rgb(hex_color)

with open('all_autres_classes.txt') as f:
    classes= f.read().split('\n')
id2cls= {i: cls for i, cls in enumerate(classes)}
cls2id= {v:k for k,v in id2cls.items()}

colors = {
    i: id_to_color(i) for i in id2cls.keys()
}

complements = ['pv', 'velux', 'cheminee', 'obstacle']


def contains_any(text: str, words: list[str]) -> bool:
    return any(text.__contains__(word) for word in words)


roofs_names = [cls for cls in classes if cls.__contains__('roof')]
autres_roofs = [cls for cls in roofs_names if not contains_any(cls, ['ardoise', 'tuiles'])]
roofs_ids = {cls: [] for cls in roofs_names}

for cls, i in cls2id.items():
    if cls in ['cheminee', 'pv', 'obstacle', 'velux'] or contains_any(cls, ['couleur', 'usure']):
        for v in roof_ids.values():
            v.append(i)

    elif cls.__contains__('ardoise'):
        roofs_ids['ardoise'].append(i)
    elif cls.__contains__('autres'):
        for rf in autres_roofs:
            roofs_ids[rf].append(i)
    elif cls.__contains__('tuiles'):
        roofs_ids['tuiles'].append(i)

def preprocess_mask(mask):

    mask_3d= np.array([mask, mask, mask]).transpose(1,2,0)
    
    for cls_id, color in colors.items():
        mask_3d= np.where(mask_3d == [cls_id]*3, color, mask_3d).astype(np.uint8)

    return mask_3d

def plot_masks(gt_mask, dt_mask, pr_mask):
    plt.figure(figsize=(16, 6))
    classes_in_mask= list(set(np.unique(pr_mask).tolist() + np.unique(gt_mask).tolist() ))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(
        preprocess_mask(gt_mask)
    )  # Convert CHW to HWC for plotting
    plt.title("Ground Truth")
    plt.axis("off")

    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.imshow(preprocess_mask(dt_mask))  # Visualize predicted mask
    plt.title("Predicted")
    plt.axis("off")

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.imshow(preprocess_mask(pr_mask))  # Visualize predicted mask
    plt.title("Processed")
    plt.axis("off")


    plt.subplots_adjust(left=0.05, right=0.75, wspace=0.1)

    patches = [mpatches.Patch(color=colors[idx]/255, label=id2cls[idx]) for idx in classes_in_mask]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Show the figure
    plt.savefig('test.png')

def mask_to_roofs(mask, cls2id, roofs_ids):
    ardoise= roofs_ids['ardoise']
    autres= roofs_ids['autres']
    tuiles= roofs_ids['tuiles']
    
    ardois= np.where(np.isin(mask, ardoise), cls2id['roof_ardoise'], 0).astype('uint8')
    autre= np.where(np.isin(mask, autres), cls2id['roof_autres'], 0).astype('uint8')
    tuile= np.where(np.isin(mask, tuiles), cls2id['roof_tuiles'], 0).astype('uint8')
    
    return {'ardoise_mask': ardois, 'autres_mask': autre, 'tuiles_mask': tuile}

