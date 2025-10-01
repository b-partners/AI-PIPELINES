import time
t1= time.time()
print('start')
import warnings
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore') 
from torchvision.transforms import Resize, InterpolationMode


import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp

import glob, cv2, os
import numpy as np
import pandas as pd

import torch, json
from torchmetrics import ConfusionMatrix, F1Score, Recall, Precision, AveragePrecision
from torchmetrics.segmentation import MeanIoU
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from Model import DeepLabV3PlusModel3B
# from SegmentationDataset import SegmentationDataset
from mask_to_vgg import mask_to_vgg
import hashlib
from preprocess import NLMDenoise
from postprocess import process_mask
from utils import mask_to_roofs, roofs_ids, cls2id

def plot_confusion_matrix(cm, acc, classes, title, save_dir, post=False):
    """Plots and saves the confusion matrix."""

    cm= cm / (cm.sum(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{title}\naccuracy: {acc}")
    pr= ""
    if post:
        pr= "post_"
    plt.savefig(f"{save_dir}/{pr}confusion_matrix.png")
    

nlm= NLMDenoise(h=8, hColor= 8, templateWindowSize=5, searchWindowSize=15)

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

# MODEL LOADING
# model = smp.from_pretrained('./deeplabV3R34-4c')





with open('alter_labels.txt') as f:
        classes= f.read().split("\n")

pairs= [(2,3), (3,2), (4,7), (7,4), (13,15), (15,13)]
# classes= ['background', 'bati', 'voirie']
# print(f"{output= }")

test_images_fps= sorted(glob.glob('alter-dataset/test/images/*.jpg'))

print(len(test_images_fps))

def postprocess_mask(mask):
    return mask.cpu().numpy().astype('uint8')

N_CHANNELS= 3
INC_ROOFS= True
SPEC_ROOFS= True
IM_SIZE= 512
resizer= Resize((IM_SIZE, IM_SIZE), interpolation=InterpolationMode.NEAREST)


def get_roof_path(img_fp):
    return img_fp.replace('images', 'roofs').replace('.jpg', '.png')

images= []
pre_images= []
gt_masks= []
roofs= []
vgg= {}

def add_channel_4(img, c4_msk):
    r,g,b= cv2.split(img)
    c4_img= cv2.merge([r,g,b, c4_mask])
    return c4_img

def prepare_img(img, im_size= 512):
    im= cv2.resize(img.copy(), (im_size, im_size), interpolation=cv2.INTER_LINEAR)
    im = im/255.
    im= (im - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return im.transpose(2, 0, 1).astype('float')

for i, img_path in enumerate(test_images_fps):
    img= cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # img= nlm(image= img)['image']
    msk= cv2.imread(img_path.replace('images', 'masks').replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
    rf= cv2.imread(img_path.replace('images', 'roofs').replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)

    gt_masks.append(msk)
    roofs.append(rf)
    images.append(img.copy())

    pre_images.append(prepare_img(img.copy()))


class_names = {
    i: cls for i, cls in enumerate(classes)
}

colors = {
    i: id_to_color(i) for i in class_names.keys()
}

n_classes= len(class_names)

def postprocess_preds(pr_mask, gt_mask):
    mask= np.where(gt_mask != 0, 1, 0).astype('uint8')
    ppr_mask= pr_mask * cv2.resize(mask, (IM_SIZE, IM_SIZE))
    return ppr_mask.astype('uint8') 

def preprocess_mask(mask):

    mask_3d= np.array([mask, mask, mask]).transpose(1,2,0)
    
    for cls_id, color in colors.items():
        mask_3d= np.where(mask_3d == [cls_id]*3, color, mask_3d).astype(np.uint8)

    return mask_3d

models_pths= ['alter_200eps_wloss1_model3r_se_resnext50_32x4d_512.pth']
print(models_pths)
print(f"{classes= }")
for model_pth in models_pths:
    print(model_pth)
    vgg= {}
    model_dir= model_pth.split(".")[0]

    model = torch.load(f"EXPERIMENTS/{model_dir}/weights/{model_pth}", weights_only=False)

    model_dir= model_pth.split(".")[0]
    print(f"warming up time is: {time.time() - t1: .3f}s")
    t1= time.time()
    print(f"{pre_images[0].shape = }")
    with torch.no_grad():
        model.eval()
        logits = model(torch.tensor(pre_images).float())
    print(f"{model_pth} inference  time of {len(test_images_fps)} images:  {time.time() - t1: .3f}s")
    # Get raw logits from the model
    t1= time.time()
    
    os.makedirs(f'EXPERIMENTS/{model_dir}/preds/figs', exist_ok=True)
    os.makedirs(f'EXPERIMENTS/{model_dir}/masks-preds', exist_ok=True)


    pr_masks = logits.softmax(dim=1)
    

    
    # Convert class probabilities to predicted class labels
    pr_masks = pr_masks.argmax(dim=1) # Shape: [batch_size, H, W]
    # os.makedirs('test-batch-res', exist_ok=True)

    
    

    r_gtmasks = resizer(torch.tensor(gt_masks).long())
    r_roofs = resizer(torch.tensor(roofs).long())
    print(f"{r_gtmasks.shape = }")
    print(f"{pr_masks.shape = }")

    y_true = r_gtmasks.view(-1)
    y_pred = pr_masks.view(-1)

    conf_mat= ConfusionMatrix(task="multiclass", num_classes=len(classes))
    f1_score= F1Score(task="multiclass", num_classes=len(classes), average='macro')
    recall= Recall(task="multiclass", num_classes=len(classes), average='macro')
    precision= Precision(task="multiclass", num_classes=len(classes), average='macro')
    avg_prc= AveragePrecision(task="multiclass", num_classes=len(classes), average='macro')
    MIoU= MeanIoU(num_classes=len(classes), input_format="index")
    
    f1s= f1_score(y_pred, y_true)
    r= recall(y_pred, y_true)
    p= precision(y_pred, y_true)
    ap= avg_prc(logits.softmax(dim=1), r_gtmasks)
    miou= MIoU(pr_masks, r_gtmasks)
    
    print("########## Metrics Avant Post Processing ##########")
    print(f"Recall = {r.item(): >30}\nPrecision = {p.item(): >30}\nAverage Precision = {ap.item(): >30}\nF1 Score = {f1s.item(): >30}\nMean IoU = {miou.item(): >30}")

    cm= conf_mat(y_pred, y_true)
    acc= cm.diag().sum()/cm.sum()
    plot_confusion_matrix(cm.cpu().numpy(), acc, classes, model_dir, f"EXPERIMENTS/{model_dir}/preds",False)


    t0= time.time()
    post_masks= []
    for idx, dt_mask in enumerate(pr_masks):
        fn= os.path.basename(test_images_fps[idx])
        # rf_mask= cv2.resize(cv2.imread(f"roofs-masks/{fn.replace('.jpg', '.png')}", cv2.IMREAD_GRAYSCALE), (IM_SIZE, IM_SIZE), interpolation= cv2.INTER_NEAREST)
        roofs_masks= mask_to_roofs(r_roofs[idx], cls2id, roofs_ids)
        post_masks.append(process_mask(dt_mask.cpu().numpy().astype('uint8'), roofs_masks, roofs_ids))
    print(f"post processed masks in {time.time() - t0: .3f}s")

    post_masks= torch.tensor(post_masks).long()

    print("Unique preds:", torch.unique(post_masks))
    print("Unique target:", torch.unique(y_true))



    f1s= f1_score(post_masks.view(-1), y_true)
    r= recall(post_masks.view(-1), y_true)
    p= precision(post_masks.view(-1), y_true)
    miou= MIoU(post_masks, r_gtmasks)
    print("########## Metrics Apres Post Processing ##########")
    print(f"Recall = {r.item(): >30}\nPrecision = {p.item(): >30}\nF1 Score = {f1s.item(): >30}\nMean IoU = {miou.item(): >30}")

    cm2= conf_mat(post_masks.view(-1), y_true)
    acc2= cm2.diag().sum()/cm2.sum()
    plot_confusion_matrix(cm2.cpu().numpy(), acc2, classes, model_dir, f"EXPERIMENTS/{model_dir}/preds", True)

    print(f"calculating metrics in {time.time() - t1: .3f}s")

    t1= time.time()

    for idx, (image, gt_mask, pr_mask, post_mask) in enumerate(zip(images, gt_masks, pr_masks, post_masks.cpu().numpy().astype('uint8'))):

        fn= os.path.basename(test_images_fps[idx])
        
        pr_mask= postprocess_mask(pr_mask, )
        vgg[fn]= mask_to_vgg(post_mask.copy(), fn, class_names, n_classes, 1024)

        try:
            save_mask= cv2.resize(post_mask, (1024,1024), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"EXPERIMENTS/{model_dir}/masks-preds/{fn.replace('.jpg', '.png')}", save_mask)
        except:
            print("erreur")
            print(f"pr_mask is None= {pr_mask is None}, {pr_mask.shape= }")

        plt.figure(figsize=(20, 8))
        classes_in_mask= np.unique(gt_mask).tolist() + np.unique(post_mask).tolist()
        
        # Original Image
        plt.subplot(1, 4, 1)
        plt.imshow(
            cv2.resize(image, pr_mask.shape, interpolation=cv2.INTER_LINEAR)
        )  # Convert CHW to HWC for plotting
        plt.title("Image")
        plt.axis("off")
        
        # Predicted Mask
        plt.subplot(1, 4, 2)
        plt.imshow(preprocess_mask(gt_mask))  # Visualize predicted mask
        plt.title("True")
        plt.axis("off")

        # Predicted Mask
        plt.subplot(1, 4, 3)
        plt.imshow(preprocess_mask(pr_mask))  # Visualize predicted mask
        plt.title("Prediction")
        plt.axis("off")

        # Predicted Mask
        plt.subplot(1, 4, 4)
        plt.imshow(preprocess_mask(post_mask))  # Visualize predicted mask
        plt.title("processed")
        plt.axis("off")

        plt.subplots_adjust(left=0.05, right=0.75, wspace=0.1)

        patches = [mpatches.Patch(color=colors[idx]/255, label=class_names[idx]) for idx in classes_in_mask]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # Show the figure
        plt.tight_layout()
        plt.savefig(f"EXPERIMENTS/{model_dir}/preds/figs/{fn.replace('.jpg', '.png')}")
    
    print(f"finished plotting in {time.time() - t1: .3f}")
    with open(f'EXPERIMENTS/{model_dir}/preds/{model_dir}.json', 'w') as f:
        json.dump(vgg, f)
        print('saved vgg json file')
    