import warnings
warnings.filterwarnings('ignore') 
import boto3
from urllib.parse import urlparse

from torch.utils.data import Dataset as BaseDataset
import torch, os, cv2
import torch.nn.functional as F
import numpy as np
from random import choice

import s3fs



import albumentations as A


class ImageWithMaskPatchesDataset(BaseDataset):
    def __init__(self, image_paths, mask_paths, n_classes = 13, patch_size=256, overlap=128, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform  # Albumentations
        self.mean= np.array([0.485, 0.456, 0.406])
        self.std= np.array([0.229, 0.224, 0.225])
        self.n_classes = n_classes
        self.bad_ids = set()
        self.len_ = len(mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def extract_patches(self, image):
        """Return list of overlapping patches and patch coordinates"""
        h, w, c = image.shape
        ps = self.patch_size
        ov = self.overlap
        stride = ps - ov

        patches = []
        coords = []

        for y in range(0, h - ps + 1, stride):
            for x in range(0, w - ps + 1, stride):
                patch = image[y:y+ps, x:x+ps]
                patches.append(patch)
                coords.append((y, x))
        return patches, coords

    def __getitem__(self, idx):
        # Load image and mask
        
        
        in_idx = idx
        while in_idx in self.bad_ids:
            in_idx = choice(range(self.len_))
        found_mask = False
        while not found_mask:
            mask = cv2.imread(self.mask_paths[in_idx], cv2.IMREAD_GRAYSCALE)
            mask_in_range = np.all((mask >= 0) & (mask < self.n_classes))
            if mask_in_range:
                found_mask = True
                break
            print(f"bad id {in_idx =}")
            self.bad_ids.add(in_idx)
            in_idx = choice(range(self.len_))
            

        
        image = cv2.imread(self.image_paths[in_idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[in_idx], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image.copy(), (512, 512))
        mask = cv2.resize(mask.copy(), (512, 512), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image/255.
        image = (image - self.mean)/self.std
        
        patches, coords = self.extract_patches(image)

        # Convert patches to tensors
        patch_tensors = [torch.from_numpy(patch.transpose(2, 0, 1)).float() for patch in patches]

        # Full mask tensor (not patchified)
        mask_tensor = torch.from_numpy(mask).long()  # or .float() depending on use case

        return {
            'patches': torch.stack(patch_tensors),  # shape (n_patches, C, H, W)
            'mask': mask_tensor,                    # full mask (H, W) or (C, H, W)
            'coords': coords,                       # list of (y, x) for each patch
            'image_id': idx                         # for tracking
        }


class SegmentationDataset(BaseDataset):

    def __init__(self, images_dir, masks_dir, classes, augmentation=None, imsize= 512, **kwargs):
        self.ids = [idx for idx in os.listdir(images_dir) if idx.endswith('.jpg')]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('.jpg', '.png')) for image_id in self.ids]
        
        self.CLASSES= classes
            

        # Always map background ('unlabelled') to 0
        self.background_class = self.CLASSES.index("background")

        self.mean= np.array([0.485, 0.456, 0.406])
        self.std= np.array([0.229, 0.224, 0.225])
        self.imsize = (imsize, imsize)
        # If specific classes are provided, map them dynamically
        if classes:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))  # Default to all classes

        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 0, other classes will be remapped starting from 1.
        self.class_map = {self.background_class: 0}
        
        self.class_map.update(
            {
                v: i
                for i, v in enumerate(self.class_values)
                if v != self.background_class
            }
        )
            
        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        try: 
            image = cv2.imread(self.images_fps[i])
            # print(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Read the mask in grayscale mode
            mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
            
            if mask.max() == 255:
                mask = (mask/255).astype(np.uint8)
            mask_remap = np.zeros_like(mask)

            # Remap the mask according to the dynamically created class map
            for class_value, new_value in self.class_map.items():
                mask_remap[mask == class_value] = new_value


            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask_remap)
                image, mask_remap = sample["image"], sample["mask"]

            image= cv2.resize(image.copy(), self.imsize, interpolation=cv2.INTER_LINEAR)
            mask_remap= cv2.resize(mask_remap.copy(), self.imsize, interpolation=cv2.INTER_NEAREST)  
            
            image = image/255.
            image = (image - self.mean)/self.std
            image = image.transpose(2, 0, 1)

            # if self.augmentation:
            #     image2= cv2.resize(image2.copy(), self.imsize, interpolation=cv2.INTER_LINEAR)
            #     mask_remap2= cv2.resize(mask_remap2.copy(), self.imsize, interpolation=cv2.INTER_NEAREST)
                
            #     image2 = image2/255.
            #     image2 = (image2 - self.mean)/self.std
            #     image2 = image2.transpose(2, 0, 1)

            #     return np.stack([image, image2]).astype('float32'), np.stack([mask_remap, mask_remap2])
            
            return image.astype('float32'), mask_remap
           
        except:
            print(self.images_fps[i])
            pass

    def __len__(self):
        return len(self.ids)

class SpotSegmentationDataset(BaseDataset):

    def __init__(self, images_dir, vmasks_dir, bmasks_dir, classes, augmentation=None, **kwargs):
        self.ids = [idx for idx in os.listdir(images_dir) if idx.endswith('.jpg')]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.vmasks_fps = [os.path.join(vmasks_dir, image_id.replace('.jpg', '.png')) for image_id in self.ids]
        self.bmasks_fps = [os.path.join(bmasks_dir, image_id.replace('.jpg', '.png')) for image_id in self.ids]
        
        self.CLASSES= ['background', 'bati', 'voirie']

        # Always map background ('unlabelled') to 0
        self.background_class = self.CLASSES.index("background")
       

        # If specific classes are provided, map them dynamically
        if classes:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))  # Default to all classes

        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 0, other classes will be remapped starting from 1.
        self.class_map = {self.background_class: 0}
        
        self.class_map.update(
            {
                v: i
                for i, v in enumerate(self.class_values)
                if v != self.background_class
            }
        )
            
        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        try: 
            image = cv2.imread(self.images_fps[i])
            # print(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Read the mask in grayscale mode
            vmask = cv2.imread(self.vmasks_fps[i], cv2.IMREAD_GRAYSCALE)
            bmask = cv2.imread(self.bmasks_fps[i], cv2.IMREAD_GRAYSCALE)

            if vmask.max() == 255:
                vmask = (vmask/255).astype(np.uint8)
            if bmask.max() == 255:
                bmask = (bmask/255).astype(np.uint8)

            mask= np.where(bmask == 1, bmask, vmask*2).astype('uint8')
            
            mask_remap = np.zeros_like(mask)

            # Remap the mask according to the dynamically created class map
            for class_value, new_value in self.class_map.items():
                mask_remap[mask == class_value] = new_value


            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask_remap)
                image, mask_remap = sample["image"], sample["mask"]

            image = image.transpose(2, 0, 1)

            
            return image, mask_remap
           
        except:
            print(self.images_fps[i])
            pass

    def __len__(self):
        return len(self.ids)

class S3SegmentationDataset(BaseDataset):

    def __init__(self, images_dir, masks_dir, classes, augmentation=None, **kwargs):
        bucket, _ = self.parse_s3_uri(images_dir)
        self.ids = [idx for idx in self.list_s3_folder(images_dir) if idx.endswith('.jpg')]
        self.images_fps = [os.path.join(f"s3://{bucket}", image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(f"s3://{bucket}", image_id.replace('images', 'masks').replace('.jpg', '.png')) for image_id in self.ids]
       
        self.CLASSES= classes
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Always map background ('unlabelled') to 0
        self.background_class = self.CLASSES.index("background")
        

        # If specific classes are provided, map them dynamically
        if classes:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))  # Default to all classes

        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 0, other classes will be remapped starting from 1.
        self.class_map = {self.background_class: 0}

        self.class_map.update(
            {
                v: i
                for i, v in enumerate(self.class_values)
                if v != self.background_class
            }
        )

        self.augmentation = augmentation
        self.fs = s3fs.S3FileSystem()
    

    def parse_s3_uri(self, s3_uri):
        parsed = urlparse(s3_uri)
        if parsed.scheme != 's3':
            raise ValueError("Invalid S3 URI")
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        return bucket, prefix

    def list_s3_folder(self, s3_uri):
        bucket, prefix = self.parse_s3_uri(s3_uri)
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')

        # print(f"📂 Listing: s3://{bucket}/{prefix}")
        files = []

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if not key.endswith('/'):
                    # print(f"📄 {key}")
                    files.append(key)

        return files

    def read_image_from_s3_cv2(self, s3_uri, typ='image'):
        decoder = cv2.IMREAD_COLOR if typ == 'image' else cv2.IMREAD_GRAYSCALE
        fs = s3fs.S3FileSystem()
        with fs.open(s3_uri, 'rb') as f:
            img_bytes = f.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(img_array, decoder)  # or IMREAD_UNCHANGED
        return image

    def __getitem__(self, i):

        # Read the image
        try:
            
            image = self.read_image_from_s3_cv2(self.images_fps[i])
            
            # print(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
           
        except Exception as e:
            print(f'img not read: {self.images_fps[i]}')
            raise e
        
        try:
            # Read the mask in grayscale mode
            mask = self.read_image_from_s3_cv2(self.masks_fps[i], typ='mask')
        except Exception as e:
            print(f'msk not read: {self.masks_fps[i]}')
            raise e
            
        try:
            # Create a blank mask to remap the class values
            mask_remap = np.zeros_like(mask)

            # Remap the mask according to the dynamically created class map
            for class_value, new_value in self.class_map.items():
                mask_remap[mask == class_value] = new_value

            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask_remap)
                image, mask_remap = sample["image"], sample["mask"]

            image = (image - self.mean)/self.std
            image = image.astype('float16')/255.
            image = image.transpose(2, 0, 1)


            return image, mask_remap
           
        except Exception as e:
            print("overall error  ", self.images_fps[i])
            raise e

    def __len__(self):
        return len(self.ids)




