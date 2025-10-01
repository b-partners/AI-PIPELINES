import warnings
warnings.filterwarnings('ignore') 

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torch

import os, glob
import cv2, json
import numpy as np
import albumentations as A

import torch

from typing import List, Optional
from segmentation_models_pytorch.losses import DiceLoss, MULTICLASS_MODE, MULTILABEL_MODE, BINARY_MODE
from segmentation_models_pytorch.losses._functional import soft_tversky_score


from Model import DeepLabV3PlusModel, DeepLabV3PlusModel2, DeepLabV3PlusModel3B, DeepLabV3PlusModel3R, DeepLabV3PlusModel3RWithFreezeAndDropout
from SegmentationDataset import SegmentationDataset
from Loss import WeightedDiceLoss, IoULoss, WeightedDiceLoss3, WeightedLovaszLoss

if __name__== "__main__":

    xp = "augmented_2_damages_kept_classes_no-intensity_all-roofed_complemented"
    
    TRAIN_DIR= f'new-dataset-toiture/{xp}_dataset'
    DATA_DIR=  f'new-dataset-toiture/{xp}_dataset'
    INC_ROOFS= True
    SPEC_ROOFS= True
    N_CHANNEL= 3

    encoders_names= ["resnext50_32x4d", "se_resnext50_32x4d", "resnext101_32x4d", "resnext101_32x8d", 'resnet152', 'resnext101_32x16d', "resnet18", 'tu-seresnext26d_32x4d', 'tu-convnextv2_atto', 'tu-seresnext26ts', 'mit_b0']
    img_sizes= [256, 384, 512, 640, 1024]

    encoder_name= encoders_names[1]
    im_size= img_sizes[0]
    enc_weights= 'imagenet'

    
    prep= True
    batch_sizes= {1024: 8,
              640: 8,
              512: 12,
              384: 16,
              256: 32}

    sf = ['weights', 'preds', 'masks-preds', 'scores']

    x_train_dir = os.path.join(TRAIN_DIR, "train", "images")
    y_train_dir = os.path.join(TRAIN_DIR, "train", "masks")

    x_valid_dir = os.path.join(TRAIN_DIR, "valid", "images")
    y_valid_dir = os.path.join(TRAIN_DIR, "valid", "masks")

    x_test_dir = os.path.join(DATA_DIR, "test", "images")
    y_test_dir = os.path.join(DATA_DIR, "test", "masks")

    print(f"{torch.cuda.is_available()= }")

    # Définition des transformations
    train_transform = A.Compose([
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=.8),
        # A.RandomGridShuffle(p=.2),
        A.GridDistortion(p=0.2),
        A.Rotate(crop_border=False, rotate_method='largest_box', p=0.2, keep_size=True),
        A.CLAHE(p=.2),
        A.OneOf(
            [
                A.RandomRain(p=.1),
                A.Blur(blur_limit=3, p=.1),
                A.MotionBlur(blur_limit=3, p=.1),
            ],
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
        A.ISONoise(p=0.1),
        # A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_LINEAR, p=1)
        ], p=1)

    train_transform2 = A.Compose([
        A.ToGray(p=0.005),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=.2),
        A.RandomGridShuffle(p=.1),
        A.Rotate(limit=20, crop_border=False, rotate_method='largest_box', p=0.1, keep_size=True),
        A.CLAHE(p=0),
        A.OneOf(
            [
                A.RandomRain(p=.05),
                A.Blur(blur_limit=3, p=.05),
                A.MotionBlur(blur_limit=3, p=.05),
            ],
            p=0.2,
        ),
        # A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_LINEAR, p=1)
        ], p=.5)

    # valid_trans = A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_LINEAR, p=1)
    
    # with open('toiture_damages_classes.txt') as f:
    #     classes = f.read().split("\n")

    

    with open(f'VGG/{xp}.json') as f:
        vgg= json.load(f)

    classes= sorted(set([reg['region_attributes']['label'] for file in vgg.values() for reg in file['regions'].values()]))
    classes.insert(0, 'background')
    
    nbr_classes = len(classes)


    train_set = SegmentationDataset(
        images_dir=x_train_dir,  
        masks_dir=y_train_dir,
        # c4_masks_dir= 'c4-masks',
        classes=classes,
        augmentation=train_transform,
        channels= N_CHANNEL,
        specify_roofs= SPEC_ROOFS,
        include_roofs= INC_ROOFS,
        imsize= im_size,
        preprocess=prep
    )

    valid_set= SegmentationDataset(
        images_dir=x_valid_dir,
        masks_dir= y_valid_dir,
        # c4_masks_dir= 'c4-masks',
        classes= classes,
        # augmentation=valid_trans,
        imsize= im_size,
        channels= N_CHANNEL,
        specify_roofs= SPEC_ROOFS,
        include_roofs= INC_ROOFS,
        preprocess=prep)

    test_set= SegmentationDataset(
        images_dir=x_test_dir,
        masks_dir= y_test_dir,
        # c4_masks_dir= 'c4-masks',
        classes= classes,
        channels= N_CHANNEL,
        # augmentation= valid_trans,
        imsize= im_size,
        specify_roofs= SPEC_ROOFS,
        include_roofs= INC_ROOFS,
        preprocess=prep)

    # Diviser en Train / Validation
   

    # Dataset de validation (optionnel)



    train_loader = DataLoader(train_set, batch_size=batch_sizes[im_size], shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_set, batch_size=batch_sizes[im_size], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_sizes[im_size], shuffle=False, num_workers=4)

    weights = None
    loss_fn = True
    # thresh = None
    # ign_ind = -1
    # lambda_conf = 0

    if weights is not None:
        masks= []
        for mask in glob.glob(f"{y_train_dir}/*.png"):
            masks.append(cv2.imread(mask, cv2.IMREAD_GRAYSCALE))

        _, c= np.unique(masks, return_counts=True)
        c= c/c.sum()

        weights= 1/(np.log(1.02) + c)
        print(_)
        print(f"{weights.shape=}")

    if weights is not None and not loss_fn:
        loss_fn = WeightedDiceLoss(  mode= 'multiclass',
                                    from_logits=True,
                                    alpha= 0.5,
                                    beta= 0.5,
                                    gamma= 1,
                                    smooth= 0.2,
                                    ignore_index= None,
                                    thresh= None,
                                    lambda_conf= 0,
                                    class_weights= torch.from_numpy(weights).view(-1) if weights is not None else None)
    if loss_fn:
        # loss_fn = WeightedDiceLoss3(
        #     mode="multiclass",
        #     # classes = list(range(len(classes))),
        #     from_logits=True,
        #     alpha= 0.7,
        #     beta= 0.3,
        #     gamma= 1.5,
        #     ignore_index=None,
        #     log_loss = False,
        #     present=True,
        #     smooth=0.2,
        #     class_weights= torch.from_numpy(weights).view(-1) if weights is not None else None,
        #     dynamic_weighting=True,
        #     focal_weighting=False,
        #     exclude_background_from_mean=False,
        #     lambda_conf=0.005
        # )
        loss_fn = WeightedDiceLoss(  mode= 'multiclass',
                                    from_logits=True,
                                    alpha=2,
                                    beta= 2,
                                    log_loss=True,
                                    gamma= 1.5,
                                    smooth= 0.2,
                                    ignore_index= None,
                                    thresh= None,
                                    lambda_conf= 0.00,
                                    class_weights= torch.from_numpy(weights).view(-1) if weights is not None else None)
       
        # loss_fn = WeightedLovaszLoss(
        #     mode=MULTICLASS_MODE,
        #     from_logits=True,
        #     per_image=True,
        #     class_weights=None,  # optional
        #     dynamic_weighting=True,
        #     focal_weighting=True,
        #     gamma=1.5,
        #     lambda_conf=0.01,
        # )



    exp_name = f"test2"

    
    for d in sf:
        os.makedirs(f"EXPERIMENTS/{exp_name}/{d}", exist_ok=True)
    
        # loss_fn= IoULoss(weights= torch.from_numpy(weights))

    
    nbr_classes= len(classes)

    model = DeepLabV3PlusModel3R( encoder_name=encoder_name, 
                                in_channels=N_CHANNEL, 
                                out_classes=classes, 
                                enc_weights= enc_weights,
                                class_weights=weights, 
                                im_size=im_size, 
                                exp_name=exp_name, 
                                loss_fn= loss_fn, 
                                sf = False)

    # model = DeepLabV3PlusModel3RWithFreezeAndDropout(
    #                             encoder_name=encoder_name, 
    #                             in_channels=N_CHANNEL, 
    #                             out_classes=classes, 
    #                             enc_weights= enc_weights,
    #                             class_weights=weights, 
    #                             im_size=im_size, 
    #                             exp_name=exp_name, 
    #                             loss_fn= loss_fn,
    #                             freeze_encoder_stages=['layer0', 'layer1', 'layer2',],
    #                             encoder_dropout=0.2,
    #                             decoder_dropout=0.2,
    #                         )

    
    early_stop_callback = EarlyStopping(monitor="valid_LOSS", patience=15, mode="min")
    
    # checkpoint_loss = ModelCheckpoint(
    #                                     dirpath=f"lightning_logs/{exp_name}/checkpoints",
    #                                     filename="best_loss_{epoch:03d}",
    #                                     monitor="valid_LOSS",
    #                                     mode="min",  # Minimize validation loss
    #                                     save_top_k=1,  # Keep only the best model
    #                                     verbose=False
    #                                 )

    # Save the best model based on validation accuracy (higher is better)
    # checkpoint_iou = ModelCheckpoint(
    #                                     dirpath=f"lightning_logs/{exp_name}/checkpoints",
    #                                     filename="best_iou_{epoch:03d}",|
    #                                     monitor="valid_ma_IoU",
    #                                     mode="max",  # Maximize validation accuracy
    #                                     save_top_k=1,  # Keep only the best model
    #                                     verbose=False
    #                                 )


    logger = CSVLogger("lightning_logs", name=exp_name)
    # Training
    trainer = pl.Trainer(precision=16, max_epochs=200, log_every_n_steps=1, default_root_dir=None, callbacks= [early_stop_callback], logger= logger)

    
    print(f"{exp_name =}")
    print(classes)
    history= trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Sauvegarder uniquement les poids du modèle
    model.model.save_pretrained(f"EXPERIMENTS/{exp_name}")
    torch.save(model, f'EXPERIMENTS/{exp_name}/weights/{exp_name}.pth')

    valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
    print('validation')
    print(valid_metrics)
    print('-------------------------------------')

    print('test metrics')
    test_metrics= trainer.test(model, dataloaders= test_loader, verbose=False)
    print(test_metrics)