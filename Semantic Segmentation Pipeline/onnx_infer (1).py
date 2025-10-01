import os
import glob
import json
import time
import hashlib
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import onnxruntime as ort
from torchvision.transforms import Resize, InterpolationMode

from mask_to_vgg import mask_to_vgg
# from preprocess import NLMDenoise


# ==============================
# CONSTANTS
# ==============================
# NLM = NLMDenoise(h=8, hColor=8, templateWindowSize=5, searchWindowSize=15)
IM_SIZE = 512
IM_RESIZER = Resize((IM_SIZE, IM_SIZE))
MSK_RESIZER = Resize((IM_SIZE, IM_SIZE), interpolation=InterpolationMode.NEAREST)


# ==============================
# UTILS
# ==============================
def int_to_hex6(num: int) -> str:
    if num < 0:
        raise ValueError("Number must be ≥ 0")
    return hashlib.sha256(str(num).encode()).hexdigest()[:6].upper()


def hex6_to_rgb(hex_str: str) -> np.ndarray:
    if len(hex_str) != 6:
        raise ValueError("Hex string must be exactly 6 characters long")
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)])


def id_to_color(cls_id: int) -> np.ndarray:
    return hex6_to_rgb(int_to_hex6(cls_id))


def get_exp_name_from_model_path(model_path: str) -> str:
    """Extract experiment name from ONNX model path."""
    return os.path.splitext(os.path.basename(model_path))[0]


# ==============================
# POSTPROCESSING
# ==============================
def apply_dense_crf(image: np.ndarray, prob_map: np.ndarray, n_iters: int = 10) -> np.ndarray:
    H, W, _ = image.shape
    n_labels = prob_map.shape[0]
    d = dcrf.DenseCRF2D(W, H, n_labels)
    d.setUnaryEnergy(utils.unary_from_softmax(prob_map))
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=image, compat=10)
    Q = d.inference(n_iters)
    return np.argmax(Q, axis=0).reshape((H, W))


def preprocess_mask(mask: np.ndarray, colors: Dict[int, np.ndarray]) -> np.ndarray:
    mask_3d = np.repeat(mask[..., None], 3, axis=-1)
    for cls_id, color in colors.items():
        mask_3d = np.where(mask_3d == [cls_id] * 3, color, mask_3d).astype(np.uint8)
    return mask_3d


# ==============================
# INFERENCE
# ==============================
def infer(
    ort_session: ort.InferenceSession,
    images: np.ndarray,
    preprocess: bool = False,
    normalize: bool = False,
    crf: bool = False,
) -> np.ndarray:
    raw_images = np.array(images) if isinstance(images, list) else images.copy()
    if raw_images.ndim == 3:
        raw_images = np.expand_dims(raw_images, axis=0)

    # if preprocess:
    #     for i, img in enumerate(raw_images):
    #         raw_images[i] = NLM(image=img)["image"]

    if normalize:
        raw_images = (raw_images / 255.0).astype("float32")

    pre_images = raw_images.transpose(0, 3, 1, 2).astype("uint8")
    ort_inputs = {"input": pre_images}
    logits = torch.from_numpy(ort_session.run(None, ort_inputs)[0])
    print("Model logits computed.")

    if crf:
        softmax = logits.softmax(dim=1).numpy()
        dt_masks = []
        for img, prob_map in zip(
            IM_RESIZER(torch.from_numpy(raw_images).permute(0, 3, 1, 2))
            .numpy()
            .transpose(0, 2, 3, 1)
            .astype("uint8"),
            softmax,
        ):
            dt_masks.append(apply_dense_crf(np.ascontiguousarray(img), prob_map))
        dt_masks = np.array(dt_masks, dtype="uint8")
    else:
        dt_masks = logits.softmax(dim=1).argmax(dim=1).cpu().numpy().astype("uint8")

    return dt_masks


# ==============================
# MAIN PIPELINE
# ==============================
def run_inference_pipeline(model_path: str, images_dir: str, suffix: str = "p") -> None:
    # Load classes
    with open("classes.txt") as f:
        classes = f.read().splitlines()
    class_names = {i: cls for i, cls in enumerate(classes)}
    colors = {i: id_to_color(i) for i in class_names.keys()}

    # Collect test images
    test_images_fps = sorted(glob.glob(f"{images_dir}/*.jpg"))
    images, gt_masks = [], []
    for img_path in test_images_fps:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(
            img_path.replace("images", "masks").replace(".jpg", ".png"),
            cv2.IMREAD_GRAYSCALE,
        )
        images.append(img)
        gt_masks.append(msk)

    print(f"Loaded {len(test_images_fps)} test images.")

    # Init ONNX model
    ort_session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
    )
    exp_name = get_exp_name_from_model_path(model_path)

    # Run inference
    t1 = time.time()
    pr_masks = infer(ort_session, np.array(images), crf=False, preprocess=False)
    print(f"{exp_name} inference on {len(test_images_fps)} images: {time.time() - t1:.3f}s")

    # Eval
    tp, fp, fn, tn = smp.metrics.get_stats(
        torch.tensor(pr_masks).long(),
        MSK_RESIZER(torch.tensor(gt_masks).long()),
        mode="multiclass",
        num_classes=len(classes),
    )
    classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=1).mean(axis=0)
    mIoU = float(np.mean(classwise_iou.cpu().numpy()))

    # Save results
    out_dir = os.path.join(os.path.dirname(model_path), f"preds{suffix}")
    os.makedirs(f"{out_dir}/figs", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)

    pd.Series({cls: classwise_iou[i].item() for i, cls in enumerate(classes)}).plot(kind="bar")
    plt.title(f"{exp_name} | mIoU= {mIoU:.3f}")
    plt.savefig(f"{out_dir}/{exp_name}.png")

    # Save masks and VGG JSON
    vgg = {}
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, gt_masks, pr_masks)):
        fn = os.path.basename(test_images_fps[idx])
        vgg[fn] = mask_to_vgg(pr_mask.copy(), fn, class_names, len(classes), 1024)
        cv2.imwrite(f"{out_dir}/masks/{fn.replace('.jpg', '.png')}",
                    cv2.resize(pr_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST))

        # Plot comparison
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 3, 1); plt.imshow(image); plt.title("Image"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(preprocess_mask(gt_mask, colors)); plt.title("True"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(preprocess_mask(pr_mask, colors)); plt.title("Prediction"); plt.axis("off")
        patches = [mpatches.Patch(color=colors[c]/255, label=class_names[c]) for c in np.unique(pr_mask)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.savefig(f"{out_dir}/figs/{fn.replace('.jpg', '.png')}")
        plt.close()

    with open(f"{out_dir}/{exp_name}.json", "w") as f:
        json.dump(vgg, f)
    print("✅ Saved predictions and VGG JSON.")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    MODEL_PATH = "EXPERIMENTS/comp_200eps_model3b_3c_resnext50_32x4d_512/onnx/comp_200eps_model3b_3c_resnext50_32x4d_512.onnx"
    IMAGES_DIR = "new-dataset/test/images"
    run_inference_pipeline(MODEL_PATH, IMAGES_DIR, suffix="p")
