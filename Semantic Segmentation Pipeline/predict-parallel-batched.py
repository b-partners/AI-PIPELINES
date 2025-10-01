import os
import glob
import time
import json
import hashlib
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Model import DeepLabV3PlusModel3B  # Adjust if needed
from mask_to_vgg import mask_to_vgg


# ==============================
# Configuration
# ==============================
BATCH_SIZE = 16
NUM_WORKERS = 4
IM_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEF_DIR = "spot-2024/tiles"
MODEL_PATHS = ["spot.pth"]
PROFILE_PATH = "target_profile_2015.npz"
ALL_CLASSES = ["background", "bati", "line"]  # Override list if needed


# ==============================
# Class Names & Colors
# ==============================
class_names = {i: cls for i, cls in enumerate(ALL_CLASSES)}


def int_to_hex6(num: int) -> str:
    return hashlib.sha256(str(num).encode()).hexdigest()[:6].upper()


def hex6_to_rgb(hex_str: str) -> np.ndarray:
    return np.array([int(hex_str[i:i + 2], 16) for i in (0, 2, 4)])


def id_to_color(cls_id: int) -> np.ndarray:
    return hex6_to_rgb(int_to_hex6(cls_id))


colors = {i: id_to_color(i) for i in class_names.keys()}


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Convert mask (H,W) of class ids to an RGB mask using deterministic colors."""
    mask_3d = np.stack([mask] * 3, axis=-1)
    for cls_id, color in colors.items():
        mask_3d[mask == cls_id] = color
    return mask_3d.astype(np.uint8)


# ==============================
# Dataset
# ==============================
class ImageDataset(Dataset):
    def __init__(self, image_paths, profile=None):
        self.image_paths = image_paths
        self.profile = profile
        if profile is not None:
            self.LAB_means = profile["lab_means"]
            self.LAB_stds = profile["lab_stds"]

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_paths)

    def color_transfer(self, source, BGR=False):
        """Transfer LAB color stats from target profile to source image."""
        src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        if BGR:
            src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)

        l_src, a_src, b_src = cv2.split(src_lab)

        src_mean, a_src_mean, b_src_mean = src_lab.mean(axis=(0, 1))
        src_std, a_src_std, b_src_std = src_lab.std(axis=(0, 1))

        tar_mean, a_tar_mean, b_tar_mean = self.LAB_means
        tar_std, a_tar_std, b_tar_std = self.LAB_stds

        l_new = (l_src - src_mean) * (tar_std / src_std) + tar_mean
        a_new = (a_src - a_src_mean) * (a_tar_std / a_src_std) + a_tar_mean
        b_new = (b_src - b_src_mean) * (b_tar_std / b_src_std) + b_tar_mean

        l_new = np.clip(l_new, 0, 255)
        a_new = np.clip(a_new, 0, 255)
        b_new = np.clip(b_new, 0, 255)

        transfer_lab = cv2.merge([l_new, a_new, b_new]).astype(np.uint8)

        if BGR:
            return cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2RGB)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        if self.profile is not None:
            img = self.color_transfer(img.copy())

        img1 = img / 255.0
        img1 = (img1 - self.mean) / self.std
        img_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float()
        return img_tensor, img.astype("uint8"), os.path.basename(path)


# ==============================
# Visualization Helpers
# ==============================
class_colors = {
    0: (0, 0, 0),      # Background
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
}


def overlay_mask_on_image(image, mask, class_colors):
    """Overlay a multiclass mask on an RGB image."""
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have matching height and width.")

    overlaid = image.copy()
    for class_id, color in class_colors.items():
        if class_id == 0:
            continue
        class_mask = (mask == class_id)
        alpha = 0.5
        for c in range(3):
            overlaid[:, :, c] = np.where(
                class_mask,
                (alpha * color[c] + (1 - alpha) * overlaid[:, :, c]).astype(np.uint8),
                overlaid[:, :, c],
            )
    return overlaid


# ==============================
# Main Pipeline
# ==============================
def run_inference():
    os.makedirs(f"{DEF_DIR}/preds", exist_ok=True)
    os.makedirs(f"{DEF_DIR}/masks-preds", exist_ok=True)

    test_images_fps = sorted(glob.glob(f"{DEF_DIR}/*.jpg"))
    print(f"Found {len(test_images_fps)} test images")

    profile = np.load(PROFILE_PATH)
    dataset = ImageDataset(test_images_fps, profile=profile)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    vgg = {}
    t0 = time.time()

    for model_pth in MODEL_PATHS:
        model_dir = os.path.splitext(model_pth)[0]
        print(f"\nLoading model {model_pth}...")

        model = torch.load(f"EXPERIMENTS/{model_dir}/weights/{model_pth}", weights_only=False)
        model = model.to(DEVICE)
        model = torch.nn.DataParallel(model)
        model.eval()

        print("Running batched inference...")
        with torch.no_grad():
            for imgs_tensor, imgs_np, fnames in dataloader:
                imgs_tensor = imgs_tensor.to(DEVICE)
                logits = model(imgs_tensor)
                pr_masks = logits.softmax(dim=1).argmax(dim=1).cpu().numpy()

                for i, fn in enumerate(fnames):
                    mask_np = pr_masks[i].astype("uint8")
                    vgg[fn] = mask_to_vgg(
                        mask_np, fn, class_names, n_classes=len(class_names), imsize=IM_SIZE
                    )

                    image_np = imgs_np[i].cpu().numpy()
                    cv2.imwrite(f"{DEF_DIR}/masks-preds/{fn.replace('.jpg', '.png')}", mask_np)

                    # Visualization
                    plt.figure(figsize=(16, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image_np)
                    plt.title("Original Image")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(overlay_mask_on_image(image_np, mask_np, class_colors))
                    plt.title("Prediction")
                    plt.axis("off")

                    unique_classes = np.unique(mask_np).tolist()
                    patches = [mpatches.Patch(color=colors[i] / 255, label=class_names[i]) for i in unique_classes]
                    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.tight_layout()

                    plt.savefig(f"{DEF_DIR}/preds/{fn.replace('.jpg', '.png')}")
                    plt.close()

    print(f"\n✅ All done in {time.time() - t0:.2f} seconds")

    with open(f"{DEF_DIR}/preds/vgg.json", "w") as f:
        json.dump(vgg, f)


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    run_inference()
