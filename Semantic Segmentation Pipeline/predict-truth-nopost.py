import os
import cv2
import glob
import json
import time
import hashlib
import warnings
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix, F1Score, Recall, Precision, AveragePrecision
from torchmetrics.segmentation import MeanIoU
from torchvision.transforms import Resize, InterpolationMode

# from preprocess import NLMDenoise
from mask_to_vgg import mask_to_vgg


# === Utils === #
def compute_class_iou(conf_matrix, class_names=None, weights=None):
    """Compute per-class IoU from a confusion matrix."""
    num_classes = conf_matrix.shape[0]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP
    denom = TP + FP + FN

    ious = TP / denom
    ious[denom == 0] = np.nan
    iou_dict = {class_names[i]: ious[i].item() for i in range(num_classes)}

    if weights is not None:
        iou_dict["weighted IoU"] = ious.dot(weights).item()

    return iou_dict


def plot_confusion_matrix(cm, acc, classes, title, save_dir, post=False):
    """Plots and saves the confusion matrix."""
    cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{title}\naccuracy: {acc}")
    pr = "post_" if post else ""
    plt.savefig(f"{save_dir}/{pr}confusion_matrix.png")
    plt.close()


def int_to_hex6(num: int) -> str:
    """Hash an integer ≥ 0 and return a 6-digit hex string."""
    if num < 0:
        raise ValueError("Number must be ≥ 0")
    return hashlib.sha256(str(num).encode()).hexdigest()[:6].upper()


def hex6_to_rgb(hex_str: str) -> np.ndarray:
    """Convert a 6-digit hexadecimal string to an RGB tuple."""
    if len(hex_str) != 6:
        raise ValueError("Hex string must be exactly 6 characters")
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)])


def id_to_color(cls_id: int) -> np.ndarray:
    return hex6_to_rgb(int_to_hex6(cls_id))


def preprocess_mask(mask, colors, class_names):
    """Convert integer mask to RGB mask using class colors."""
    mask_3d = np.array([mask, mask, mask]).transpose(1, 2, 0)
    for cls_id, color in colors.items():
        mask_3d = np.where(mask_3d == [cls_id] * 3, color, mask_3d).astype(np.uint8)
    return mask_3d


def prepare_img(img, im_size=256):
    """Resize and normalize image for model input."""
    im = cv2.resize(img.copy(), (im_size, im_size), interpolation=cv2.INTER_LINEAR)
    im = im / 255.0
    im = (im - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return im.transpose(2, 0, 1).astype("float32")


# === Main pipeline === #
def main(model_paths: list[str]):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    print("Start inference pipeline...")

    # === Load classes === #
    with open("toiture_damages_classes.txt") as f:
        classes = f.read().split("\n")

    # Build dataset
    xp = "damages_kept_classes_no-intensity_all-roofed_complemented"
    with open(f"VGG/{xp}.json") as f:
        vgg = json.load(f)

    classes = sorted({reg["region_attributes"]["label"] for file in vgg.values() for reg in file["regions"].values()})
    classes.insert(0, "background")
    class_names = {i: cls for i, cls in enumerate(classes)}
    colors = {i: id_to_color(i) for i in class_names.keys()}

    test_images_fps = sorted(glob.glob(f"new-dataset-toiture/{xp}_dataset/test/images/*.jpg"))
    print(f"Found {len(test_images_fps)} test images")

    # === Preprocess images and masks === #
    gt_masks, images, pre_images = [], [], []
    for img_path in test_images_fps:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(img_path.replace("images", "masks").replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)
        if msk.max() == 255:
            msk = (msk / 255).astype("uint8")
        gt_masks.append(msk)
        images.append(img.copy())
        pre_images.append(prepare_img(img))

    resizer = Resize((256, 256), interpolation=InterpolationMode.NEAREST)
    r_gtmasks = resizer(torch.tensor(gt_masks).long())

    # === Run models === #
    for model_pth in model_paths:
        model_dir = os.path.splitext(os.path.basename(model_pth))[0]
        print(f"\nLoading model {model_pth} ...")

        model = torch.load(model_pth, weights_only=False)
        model.eval()

        with torch.no_grad():
            logits = model(torch.tensor(pre_images).float())
            pr_masks = logits.softmax(dim=1).argmax(dim=1)

        # Metrics
        y_true = r_gtmasks.view(-1)
        y_pred = pr_masks.view(-1)
        weights = F.one_hot(r_gtmasks, len(class_names)).permute(0, 3, 1, 2).cpu().numpy().sum(axis=(0, 2, 3))
        weights = weights / weights.sum()

        conf_mat = ConfusionMatrix(task="multiclass", num_classes=len(classes))
        f1 = F1Score(task="multiclass", num_classes=len(classes), average="macro")(y_pred, y_true)
        recall = Recall(task="multiclass", num_classes=len(classes), average="macro")(y_pred, y_true)
        precision = Precision(task="multiclass", num_classes=len(classes), average="macro")(y_pred, y_true)
        miou = MeanIoU(num_classes=len(classes), input_format="index", per_class=True)(pr_masks, r_gtmasks)

        per_class_iou = {class_names[i]: mi.item() for i, mi in enumerate(miou)}
        cm = conf_mat(y_pred, y_true)
        acc = cm.diag().sum() / cm.sum()

        print(f"\n{' ' + model_dir + ' ':#^80}")
        print(f"Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")
        pprint(per_class_iou)
        pprint(compute_class_iou(cm.numpy(), classes, weights))

        # Save confusion matrix
        os.makedirs(f"EXPERIMENTS/{model_dir}/preds", exist_ok=True)
        plot_confusion_matrix(cm.cpu().numpy(), acc, classes, model_dir, f"EXPERIMENTS/{model_dir}/preds")

        # Save predictions
        os.makedirs(f"EXPERIMENTS/{model_dir}/preds/figs", exist_ok=True)
        for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, r_gtmasks.cpu().numpy(), pr_masks)):
            fn = os.path.basename(test_images_fps[idx])
            plt.figure(figsize=(20, 8))
            plt.subplot(1, 3, 1); plt.imshow(image); plt.title("Image"); plt.axis("off")
            plt.subplot(1, 3, 2); plt.imshow(preprocess_mask(gt_mask, colors, class_names)); plt.title("True"); plt.axis("off")
            plt.subplot(1, 3, 3); plt.imshow(preprocess_mask(pr_mask, colors, class_names)); plt.title("Prediction"); plt.axis("off")
            patches = [mpatches.Patch(color=colors[cid]/255, label=class_names[cid]) for cid in np.unique([*gt_mask.ravel(), *pr_mask.ravel()])]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"EXPERIMENTS/{model_dir}/preds/figs/{fn.replace('.jpg', '.png')}")
            plt.close()

    print(f"\n✅ Done in {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    models_pths = [
        "EXPERIMENTS/damages_tt_200eps_ablossg15_model3r_mit_b0_384/weights/damages_tt_200eps_ablossg15_model3r_mit_b0_384.pth"
    ]
    main(models_pths)
