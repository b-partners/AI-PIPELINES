import os
import glob
import time
import json
import hashlib
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.transforms import Resize, InterpolationMode
from Model import DeepLabV3PlusModel3B


# ==============================
# Configuration (variables)
# ==============================
ALL_CLASSES_PATH = "all_classes.txt"
TEST_IMAGES_DIR = "test-zooms/zoom20-vendee18-tiff"
MASKS_DIR = os.path.join(TEST_IMAGES_DIR, "masks-preds")
PREDS_DIR = os.path.join(TEST_IMAGES_DIR, "preds")
EXPERIMENTS_DIR = "EXPERIMENTS"
IM_SIZE = 1024

# Put your models here (list for reusability)
MODEL_PATHS = [
    "original_tt_200eps_ablossg2_model3r_se_resnext50_32x4d_512.pth",
]


# ==============================
# Utilities
# ==============================
def load_class_names(file_path: str) -> dict[int, str]:
    """Load class names from file and map index -> class name."""
    with open(file_path, "r") as f:
        classes = f.read().splitlines()
    return {i: cls for i, cls in enumerate(classes)}


def int_to_hex6(num: int) -> str:
    return hashlib.sha256(str(num).encode()).hexdigest()[:6].upper()


def hex6_to_rgb(hex_str: str) -> np.ndarray:
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)])


def build_color_map(class_names: dict[int, str]) -> dict[int, np.ndarray]:
    """Generate a deterministic color for each class id."""
    return {i: hex6_to_rgb(int_to_hex6(i)) for i in class_names.keys()}


def preprocess_mask(mask: np.ndarray, colors: dict[int, np.ndarray]) -> np.ndarray:
    """Map each class id in mask to its corresponding RGB color."""
    mask_3d = np.stack([mask] * 3, axis=-1)
    for cls_id, color in colors.items():
        mask_3d[mask == cls_id] = color
    return mask_3d.astype(np.uint8)


def load_images(image_dir: str) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load and preprocess test images."""
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    images, pre_images = [], []

    for img_path in img_paths:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        images.append(img)
        pre_images.append(img.transpose(2, 0, 1))  # CHW format

    return images, pre_images, img_paths


# ==============================
# Inference + Saving
# ==============================
def run_inference(model_path: str, pre_images: list[np.ndarray]) -> torch.Tensor:
    """Load model and run inference on preprocessed images."""
    model_name = os.path.splitext(model_path)[0]
    model = torch.load(os.path.join(EXPERIMENTS_DIR, model_name, "weights", model_path), weights_only=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(pre_images).float())
        return logits.softmax(dim=1).argmax(dim=1)


def save_predictions(
    images: list[np.ndarray],
    pr_masks: torch.Tensor,
    img_paths: list[str],
    colors: dict[int, np.ndarray],
    class_names: dict[int, str],
) -> None:
    """Save prediction masks and visualization plots."""
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

    for idx, (image, pr_mask) in enumerate(zip(images, pr_masks)):
        fn = os.path.basename(img_paths[idx])
        mask_np = pr_mask.cpu().numpy().astype("uint8")

        # Save raw mask
        cv2.imwrite(os.path.join(MASKS_DIR, fn.replace(".jpg", ".png")), mask_np)

        # Save visualization
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(preprocess_mask(mask_np, colors))
        plt.title("Prediction")
        plt.axis("off")

        unique_classes = np.unique(mask_np).tolist()
        patches = [mpatches.Patch(color=colors[i] / 255, label=class_names[i]) for i in unique_classes]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(os.path.join(PREDS_DIR, fn.replace(".jpg", ".png")))
        plt.close()


# ==============================
# Main Entry Point
# ==============================
def main():
    t0 = time.time()

    # Load classes + color map
    class_names = load_class_names(ALL_CLASSES_PATH)
    colors = build_color_map(class_names)

    # Load test images
    images, pre_images, img_paths = load_images(TEST_IMAGES_DIR)
    print(f"Found {len(images)} test images")

    # Run inference for each model
    for model_path in MODEL_PATHS:
        print(f"\nLoading model {model_path}...")
        pr_masks = run_inference(model_path, pre_images)
        print("Saving masks and visualizations...")
        save_predictions(images, pr_masks, img_paths, colors, class_names)

    print(f"\nAll done in {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
