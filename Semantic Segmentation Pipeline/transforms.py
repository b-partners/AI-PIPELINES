import cv2, os
import numpy as np
import albumentations as A
from sklearn.decomposition import PCA

class Roll(A.DualTransform):
    def __init__(self, x_limit: float = 1, y_limit: float = 1, always_apply=False, p=1.0):
        """
        Albumentations transform that shifts and rolls (wraps around) an image/mask.

        Parameters:
        - x_limit: Maximum shift fraction of image width (0 to 1).
        - y_limit: Maximum shift fraction of image height (0 to 1).
        """
        super().__init__(always_apply=always_apply, p=p)
        self.x_limit = min(x_limit, 1)  # Ensure limit is at most 1
        self.y_limit = min(y_limit, 1)

    def get_params(self):
        """Generate random shift values within the defined limits."""
        return {
            "shift_x": np.random.uniform(-self.x_limit, self.x_limit),
            "shift_y": np.random.uniform(-self.y_limit, self.y_limit)
        }

    def apply(self, image, shift_x=1, shift_y=1, **params):
        return self.shift_and_roll(image, shift_x, shift_y)

    def apply_to_mask(self, mask, shift_x=1, shift_y=1, **params):
        return self.shift_and_roll(mask, shift_x, shift_y)

    def shift_and_roll(self, image, shift_x, shift_y):
        """Shifts and wraps (rolls) an image circularly in both x and y directions."""
        try:
            h, w, _ = image.shape  # Image case (RGB)
        except ValueError:
            h, w = image.shape  # Mask case (grayscale)

        shift_x = int(w * shift_x)
        shift_y = int(h * shift_y)

        rolled_image = np.roll(image, shift=shift_x, axis=1)  # Shift along width (X-axis)
        rolled_image = np.roll(rolled_image, shift=shift_y, axis=0)  # Shift along height (Y-axis)
        return rolled_image

class NLMDenoise(A.DualTransform):
    def __init__(self, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21, 
                 always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.h = h
        self.hColor = hColor
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize

    def apply(self, image, **params):
        return self.apply_denoising(image)

    def apply_to_mask(self, mask, **params):
        return mask  # Mask remains unchanged

    def apply_denoising(self, image):
        # Apply PCA denoising
      
        # Apply Non-Local Means denoising on top of PCA
        return cv2.fastNlMeansDenoisingColored(
            image, None, self.h, self.hColor, self.templateWindowSize, self.searchWindowSize
        )

class NLMPCADenoise(A.DualTransform):
    def __init__(self, components_ratio=0.95, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21, 
                 always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.components_ratio = components_ratio
        self.h = h
        self.hColor = hColor
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize

    def apply(self, image, **params):
        return self.apply_denoising(image)

    def apply_to_mask(self, mask, **params):
        return mask  # Mask remains unchanged

    def pca_denoise(self, image):
        h, w, c = image.shape
        img_reshaped = image.reshape((-1, c))

        # Determine valid number of components based on available samples
        max_components = min(img_reshaped.shape)
        n_components = max(1, int(max_components * self.components_ratio))  # At least 1 component

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(img_reshaped)
        denoised = pca.inverse_transform(transformed)

        return denoised.reshape((h, w, c)).astype(np.uint8)

    def apply_denoising(self, image):
        # Apply PCA denoising
        pca_denoised = self.pca_denoise(image)

        # Apply Non-Local Means denoising on top of PCA
        return cv2.fastNlMeansDenoisingColored(
            pca_denoised, None, self.h, self.hColor, self.templateWindowSize, self.searchWindowSize
        )

class AutoBrightness(A.DualTransform):
    def __init__(self, target_brightness=128, always_apply=True, p=1.0):
        """
        Albumentations transform that automatically adjusts brightness.

        Parameters:
        - target_brightness: Desired mean brightness level (0-255).
        """
        super().__init__(always_apply, p)
        self.target_brightness = target_brightness

    def apply(self, image, **params):
        return self.adjust_brightness(image)
    
    def apply_to_mask(self, mask, **params):
        return mask

    def adjust_brightness(self, image):
        """Automatically adjusts brightness to match the target mean brightness."""
        # Convert image to float32 for precision
        image = image.astype(np.float32)

        # Convert to grayscale and compute mean brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        # min_brightness= np.min(gray)

        # Compute brightness scaling factor
        if mean_brightness > 0:
            scale = self.target_brightness/mean_brightness
            scale= min(scale, 1.5)
        else:
            scale = 1.0  # Avoid division by zero

        # Apply brightness correction: scale pixel values
        adjusted = np.clip(image.transpose(2,0,1) * scale, 0, 255).astype(np.uint8).transpose(1,2,0)

        return adjusted


class AutoContrast(A.ImageOnlyTransform):
    def __init__(self, target_contrast=64, always_apply=True, p=1.0):
        """
        Albumentations transform that automatically adjusts contrast.

        Parameters:
        - target_contrast: Desired standard deviation of pixel intensities.
        """
        super().__init__(always_apply, p)
        self.target_contrast = target_contrast

    def apply(self, image, **params):
        return self.adjust_contrast(image)

    def adjust_contrast(self, image):
        """Automatically adjusts contrast to match the target level."""
        # Convert image to float32 for precision
        image = image.astype(np.float32)

        # Convert to grayscale to calculate contrast (std deviation)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)

        # Compute contrast scale factor
        if contrast > 0:
            contrast_scale = self.target_contrast / contrast
            contrast_scale= min(contrast_scale, 1.5)
        else:
            contrast_scale = 1.0  # Avoid division by zero

        # Apply contrast adjustment
        mean_intensity = np.mean(image)
        adjusted = np.clip((image - mean_intensity) * contrast_scale + mean_intensity, 0, 255).astype(np.uint8)

        return adjusted

class AutoSaturation(A.DualTransform):
    def __init__(self, target_saturation=100, max_scale=1.5, always_apply=True, p=1.0):
        """
        Albumentations transform that automatically adjusts saturation.

        Parameters:
        - target_saturation: Desired mean saturation level (0-255).
        - max_scale: Maximum allowed saturation adjustment factor to prevent over-enhancement.
        """
        super().__init__(always_apply, p)
        self.target_saturation = target_saturation
        self.max_scale = max_scale

    def apply(self, image, **params):
        return self.adjust_saturation(image)
    
    def apply_to_mask(self, mask, *args, **params):
        return mask

    def adjust_saturation(self, image):
        """Automatically adjusts saturation while keeping it within safe limits."""
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Extract saturation channel
        saturation = hsv[:, :, 1]

        # Compute mean saturation
        mean_saturation = np.mean(saturation)

        # Compute scaling factor
        if mean_saturation > 0:
            saturation_scale = self.target_saturation / mean_saturation
            saturation_scale = min(saturation_scale, self.max_scale)  # Clamp to max_scale
        else:
            saturation_scale = 1.0  # Avoid division by zero

        # Adjust saturation
        hsv[:, :, 1] = np.clip(saturation * saturation_scale, 0, 255)

        # Convert back to BGR
        adjusted_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return adjusted_image
