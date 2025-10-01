import os
import cv2
import numpy as np
from skimage import exposure
from tqdm import tqdm

# ---------- Helpers ----------

def compute_cdf(img_gray):
    """Compute normalized CDF for grayscale image."""
    hist, bins = np.histogram(img_gray.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # normalize
    return cdf

def match_cdf(src_gray, target_cdf):
    """Histogram specification: match src_gray to target_cdf."""
    src_hist, bins = np.histogram(src_gray.flatten(), 256, [0,256])
    src_cdf = src_hist.cumsum()
    src_cdf = src_cdf / src_cdf[-1]

    # build LUT by matching closest CDF values
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(src_cdf[i] - target_cdf)
        lut[i] = np.argmin(diff)

    return cv2.LUT(src_gray, lut)

def apply_texture(img, texture=0.3):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Texture enhancement (local contrast)
    L = lab[:,:,0].astype(np.uint8)
    blurred = cv2.GaussianBlur(L, (0,0), 3)
    L_tex = cv2.addWeighted(L, 1 + texture, blurred, -texture, 0)
    lab[:,:,0] = np.clip(L_tex, 0, 255)
    
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out

    
## APPLYING

def apply_profile(img, profile_file="profile.npz", texture=.1):
    prof = np.load(profile_file)

    lab_means = prof["lab_means"]
    lab_stds = prof["lab_stds"]
    cdf_mean = prof["cdf_mean"]


    # --- Color transfer (LAB mean/std) ---
    src_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean = src_lab.mean(axis=(0,1))
    src_std = src_lab.std(axis=(0,1))

    l_new = (src_lab[:,:,0] - src_mean[0]) * (lab_stds[0]/src_std[0]) + lab_means[0]
    a_new = (src_lab[:,:,1] - src_mean[1]) * (lab_stds[1]/src_std[1]) + lab_means[1]
    b_new = (src_lab[:,:,2] - src_mean[2]) * (lab_stds[2]/src_std[2]) + lab_means[2]

    l_new = np.clip(l_new, 0, 255).astype(np.uint8)
    a_new = np.clip(a_new, 0, 255).astype(np.uint8)
    b_new = np.clip(b_new, 0, 255).astype(np.uint8)

    lab_new = cv2.merge([l_new, a_new, b_new])
    img_colored = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

    # --- Exposure match (histogram specification of L channel) ---
    lab2 = cv2.cvtColor(img_colored, cv2.COLOR_BGR2LAB)
    L = lab2[:,:,0]
    L_matched = match_cdf(L, cdf_mean)
    lab2[:,:,0] = L_matched
    img_exposure = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # --- Texture match (sharpness) ---
    final_img = apply_texture(img_exposure, texture=texture)

    return final_img

if __name__ == "__main__":
    target_folder = 'Fusion_data_Place_Stanislas_256X256_Z25'
    for fn in tqdm(os.listdir(target_folder), unit="image(s)", desc="preprocessing images"):
        img = cv2.imread(f"{target_folder}/{fn}")
        n_img = apply_profile(img, "stanislas_profile.npz", texture=.2)
        cv2.imwrite(f"{target_folder}/{fn}", n_img)