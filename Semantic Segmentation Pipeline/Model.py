import warnings
warnings.filterwarnings('ignore') 

import torch.nn.functional as F
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch, csv
import gc
from torchmetrics import ConfusionMatrix
from collections import defaultdict
import numpy as np
from torchvision.transforms import Resize
from segmentation_models_pytorch.losses import MULTICLASS_MODE


class DeepLabV3PlusModel(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=len(out_classes),
            **kwargs,
        )


        self.mode= mode
        self.im_size= im_size
        self.in_resize= Resize((im_size, im_size))
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        image= self.in_resize(image)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        mask= self.in_resize(mask)
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class DeepLabV3PlusModel2(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels= 512,
            decoder_atrous_rates= (6,12,18),
            
            **kwargs,
        )


        self.mode= mode
        self.in_resize= Resize((im_size, im_size))
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=enc_weights)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        image= self.in_resize(image)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        mask= self.in_resize(mask)
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class SegFormer(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels= 512,
            decoder_atrous_rates= (6,12,18),
            
            **kwargs,
        )


        self.mode= mode
        self.in_resize= Resize((im_size, im_size))
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=enc_weights)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        image= self.in_resize(image)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        mask= self.in_resize(mask)
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

class DeepLabV3PlusModel3(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels= 512,
            decoder_atrous_rates= (6,15,24),
            
            **kwargs,
        )


        self.mode= mode
        self.in_resize= Resize((im_size, im_size))
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=enc_weights)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        image= self.in_resize(image)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        mask= self.in_resize(mask)
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'EXPERIMENTS/{self.exp_name}/scores/{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

class DeepLabV3PlusModel3B(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels= 512,
            decoder_atrous_rates= (6,15,24),
            
            **kwargs,
        )


        self.mode= mode
        self.im_size=im_size
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=enc_weights)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.register_buffer("norm", torch.tensor([255.]))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        
        image=  F.interpolate(image.float(), size=(self.im_size, self.im_size), mode="bilinear", align_corners=False)
        image= image/ self.norm.view(1, -1, 1, 1)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        mask= F.interpolate(mask.unsqueeze(1), size=(self.im_size, self.im_size), mode="nearest").squeeze()
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'EXPERIMENTS/{self.exp_name}/scores/{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

class DeepLabV3PlusModel3RP(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, exp_name= None, im_size= 512, loss_fn=None, sf: bool = False, nw = 4, bs = 4, **kwargs):
        super().__init__()
        if not sf:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=enc_weights,
                in_channels=in_channels,
                classes=len(out_classes),
                encoder_output_stride=8,
                decoder_channels= 512,
                decoder_atrous_rates= (6,12,18),
                **kwargs,
            )
        else:
            self.model = smp.Segformer(
                encoder_name=encoder_name,
                encoder_weights=enc_weights,
                in_channels=in_channels,
                classes=len(out_classes),
                decoder_channels= 256,
                # decoder_atrous_rates= (6,15,24),
                **kwargs,
            )


        self.mode= mode
        self.im_size=im_size
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
       
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.nw = nw
        self.bs = bs


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        patches = batch["patches"]            # (total_patches, C, H, W)
        coords = batch["coords"]              # list of (y, x)
        image_ids = batch["image_ids"]        # list of int (index of full image)
        masks = batch["masks"]                # shape: (batch_size, H, W)
        
        patches = patches.to(self.device)
        masks = masks.to(self.device)

        image_id_mask = {}
        i = 0
       
        for img_id in image_ids:
            if img_id not in image_id_mask.keys():
                image_id_mask[img_id] = i
                i += 1
            
        
        torch.autograd.set_detect_anomaly(True)
        logits_patches = self.forward(patches)   # shape: (total_patches, num_classes, H, W)
    
        # Group outputs and coords by image_id
        grouped_outputs = defaultdict(list)
        grouped_coords = defaultdict(list)
        grouped_masks = defaultdict(list)
    
        for i, img_id in enumerate(image_ids):
            grouped_outputs[img_id].append(logits_patches[i])
            grouped_coords[img_id].append(coords[i])
    
        losses = []
        tps, fps, fns, tns = [], [], [], []
    
        for img_id in grouped_outputs:
            # Get ground truth mask
            mask = masks[image_id_mask[img_id]]  # shape: (H, W)
            h_full, w_full = mask.shape[-2:]

            # Prepare accumulator (initialized to very low values so max works)
            # acc = torch.full((self.number_of_classes, h_full, w_full), fill_value=-1e9, device=self.device)
            
            # patch_size = logits_patches.shape[-1]
            
            # for patch, (y, x) in zip(grouped_outputs[img_id], grouped_coords[img_id]):
            #     acc[:, y:y+patch_size, x:x+patch_size] = torch.maximum(
            #         acc[:, y:y+patch_size, x:x+patch_size], patch
            #     )
            
            # pred_full = acc  # shape: (C, H, W), already max-fused

            
            # Prepare accumulators
            acc = torch.zeros((self.number_of_classes, h_full, w_full), device=self.device)
            count = torch.zeros((1, h_full, w_full), device=self.device)
    
            patch_size = logits_patches.shape[-1]
    
            for patch, (y, x) in zip(grouped_outputs[img_id], grouped_coords[img_id]):
                acc[:, y:y+patch_size, x:x+patch_size] += patch
                count[:, y:y+patch_size, x:x+patch_size] += 1
    
            count[count == 0] = 1  # prevent div-by-zero
            pred_full = acc / count  # shape: (C, H, W)
    
            # Apply loss
            pred_full = pred_full.unsqueeze(0)  # (1, C, H, W)
            gt = mask.unsqueeze(0)              # (1, H, W)
            loss = self.loss_fn(pred_full, gt)
            losses.append(loss)
    
            # Post-process for metrics
            prob_mask = pred_full.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)  # shape: (1, H, W)
    
            # Metrics
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, gt, mode=self.mode, num_classes=self.number_of_classes)
    
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            tns.append(tn)
    
        # Stack metrics
        tp = torch.cat(tps, dim=0)
        fp = torch.cat(fps, dim=0)
        fn = torch.cat(fns, dim=0)
        tn = torch.cat(tns, dim=0)
    
        return {
            "loss": torch.mean(torch.stack(losses)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }


    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'EXPERIMENTS/{self.exp_name}/scores/{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr = 5e-5
        weight_decay = 0.01
        betas = (0.9, 0.999)
        eps = 1e-8
    
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e3
            )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or "epoch" if you step it per epoch
                "frequency": 1,
            },
        }

class DeepLabV3PlusModel3R(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, exp_name= None, im_size= 512, loss_fn=None, sf: bool = False, **kwargs):
        super().__init__()
        if not sf:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=enc_weights,
                in_channels=in_channels,
                classes=len(out_classes),
                encoder_output_stride=8,
                decoder_channels= 512,
                decoder_atrous_rates= (6,12,18),
                **kwargs,
            )
        else:
            self.model = smp.Segformer(
                encoder_name=encoder_name,
                encoder_weights=enc_weights,
                in_channels=in_channels,
                classes=len(out_classes),
                decoder_channels= 256,
                # decoder_atrous_rates= (6,15,24),
                **kwargs,
            )


        self.mode= mode
        self.im_size=im_size
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
       
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'EXPERIMENTS/{self.exp_name}/scores/{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr = 5e-5
        weight_decay = 0.01
        betas = (0.9, 0.999)
        eps = 1e-8
    
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e3
            )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or "epoch" if you step it per epoch
                "frequency": 1,
            },
        }


class DeepLabV3PlusModel3C(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels= 512,
            decoder_atrous_rates= (6,15,24),
            
            **kwargs,
        )


        self.mode= mode
        self.im_size=im_size
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=enc_weights)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.register_buffer("norm", torch.tensor(255.))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        
        image=  F.interpolate(image.float(), size=(self.im_size, self.im_size), mode="bilinear", align_corners=False)
        image= image/ self.norm
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        mask= F.interpolate(mask.unsqueeze(1), size=(self.im_size, self.im_size), mode="nearest").squeeze()
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'EXPERIMENTS/{self.exp_name}/scores/{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class DeepLabV3PlusModel4(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels= 512,
            decoder_atrous_rates= (9,18,27),
            
            **kwargs,
        )


        self.mode= mode
        self.in_resize= Resize((im_size, im_size))
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=enc_weights)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        image= self.in_resize(image)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        mask= self.in_resize(mask)
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class DeepLabV3PlusModel5(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels= 512,
            decoder_atrous_rates= (6,15,24),
            
            **kwargs,
        )


        self.mode= mode
        self.in_resize= Resize((im_size, im_size))
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=enc_weights)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        image= self.in_resize(image)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        mask= self.in_resize(mask)
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            # mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()
        
        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask.to(torch.float32), mask.to(torch.float32))

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        # if self.mode == 'multilabel':
            # pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

class DeepLabV3PlusModel3P(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, class_weights, enc_weights='imagenet', mode= MULTICLASS_MODE, mean= 0, std= 1, exp_name= None, im_size= 256, loss_fn=None,**kwargs):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels= 512,
            decoder_atrous_rates= (6,15,24),
            
            **kwargs,
        )


        self.mode= mode
        self.im_size=im_size
        if exp_name is None:
            exp_name= f"{encoder_name}_{im_size}"
        
        self.exp_name= exp_name
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=enc_weights)
        self.classes= out_classes
        self.number_of_classes = len(out_classes)
        self.weights= class_weights
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.register_buffer("norm", torch.tensor(255.))
        if in_channels == 4:
            std= params["std"] + [std]
            mean= params["mean"] + [mean]
            self.register_buffer("std", torch.tensor(std).view(1, 4, 1, 1))
            self.register_buffer("mean", torch.tensor(mean).view(1, 4, 1, 1))

        # Loss function for multi-class segmentation
        # self.tversky_loss= smp.losses.TverskyLoss(mode= mode, from_logits=True, alpha=0.4, beta= 0.6, smooth= 0.1)
        # self.focal_loss= smp.losses.FocalLoss(mode= mode)
        # self.lovasz_loss= smp.losses.LovaszLoss(mode= mode, per_image=False)
        
        self.loss_fn= loss_fn
        if loss_fn is None:
            self.loss_fn= smp.losses.LovaszLoss(mode= mode, per_image=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    # def loss_fn(self, y_pred, y_true, alpha=0.3, beta= 0.5, gamma= 0.2):
    #     return alpha * self.tversky_loss(y_pred, y_true) + beta * self.lovasz_loss(y_pred, y_true) + gamma * self.focal_loss(y_pred, y_true)


    def forward(self, image):
        # Normalize image
        
       
        image= image/ self.norm
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
    
        # Ensure that image dimensions are correct
        assert image.ndim == 4, f"image ndim === {image.ndim}" # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        if self.mode == 'multilabel':
            mask= F.one_hot(mask, self.number_of_classes).permute(0,3,1,2)

            assert mask.ndim == 4, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]
        # Mask shape
        else:
            assert mask.ndim == 3, f'mask ndim == {mask.ndim}'  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask= F.one_hot(pred_mask, self.number_of_classes).permute(0,3,1,2)
        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        # metrics = self.compute_metrics(pred_mask, mask)

        

        return {
            "loss":loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # **metrics
        }

    def compute_metrics(self, pred_mask, mask):
        pred_mask_ = pred_mask.cpu().numpy().flatten()
        mask_ = mask.cpu().numpy().flatten()

        return {
            "accuracy": accuracy_score(pred_mask_, mask_),
            "recall": recall_score(pred_mask_, mask_, average="macro"),
            "precision": precision_score(pred_mask_, mask_, average="macro"),
            "f1": f1_score(pred_mask_, mask_, average="macro"),
            "IoU": jaccard_score(pred_mask_, mask_, average="macro")
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss= torch.tensor([x["loss"].item() for x in outputs])
        # accuracy= torch.tensor([x["accuracy"] for x in outputs])
        # recall= torch.tensor([x["recall"] for x in outputs])
        # precision= torch.tensor([x["precision"] for x in outputs])
        # IoU= torch.tensor([x["IoU"] for x in outputs])
        
        

        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        # weights= np.ones(self.number_of_classes).tolist()
        # weights[0]= 0
        mu_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU= smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # Per-image IoU and dataset IoU calculations
        
        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }

        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # L1 Regularization (Manual Addition)
        # lambda_l1 = 1e-5
        # l1_loss = self.l1_regularization(lambda_l1)
        # train_loss_info["loss"] += l1_loss  # Add L1 penalty

        self.training_step_outputs.append(train_loss_info)


        return train_loss_info

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        """Logs the class-wise IoU values to a CSV file."""
        # Compute class-wise IoU
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)

        # Prepare the data to write to the file
        classwise_iou_data = classwise_iou.cpu().numpy().tolist()

        # Specify the path for the log file
        log_file = f'EXPERIMENTS/{self.exp_name}/scores/{self.exp_name}_{stage}_classwise_iou.csv'

        # If the file doesn't exist, write the header; otherwise, append the data
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        # Write classwise IoU to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header with class IDs as column names
                writer.writerow(['loss'] + self.classes)

            # Write a single row with IoU values for each class
            writer.writerow([loss] + classwise_iou_data)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        lr= 1e-4
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # Initialize SWA model after optimizer is created
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import csv
from torch.optim.swa_utils import AveragedModel, SWALR
from typing import List, Optional
from segmentation_models_pytorch.losses import DiceLoss

MULTICLASS_MODE = "multiclass"

class DeepLabV3PlusModelSWA(pl.LightningModule):
    def __init__(
        self,
        encoder_name,
        in_channels,
        out_classes,
        class_weights=None,
        enc_weights='imagenet',
        mode=MULTICLASS_MODE,
        exp_name=None,
        im_size=512,
        loss_fn=None,
        swa_start_epoch=10,
        swa_lr=1e-4,
        **kwargs
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels=512,
            decoder_atrous_rates=(6, 12, 18),
            **kwargs,
        )

        self.mode = mode
        self.im_size = im_size
        self.exp_name = exp_name or f"{encoder_name}_{im_size}"
        self.classes = out_classes
        self.number_of_classes = len(out_classes)
        self.weights = class_weights
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.use_swa = True

        self.loss_fn = loss_fn or smp.losses.LovaszLoss(mode=mode, per_image=True)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.swa_model = None
        self.swa_scheduler = None

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage):
        image, mask = batch
        mask = mask.long()

        if self.mode == 'multilabel':
            mask = F.one_hot(mask, self.number_of_classes).permute(0, 3, 1, 2)
        
        logits_mask = self(image).contiguous()
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)

        if self.mode == 'multilabel':
            pred_mask = F.one_hot(pred_mask, self.number_of_classes).permute(0, 3, 1, 2)

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )

        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = torch.tensor([x["loss"].item() for x in outputs])

        mu_IoU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")

        metrics = {
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }
        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)

        # SWA model update
        if self.use_swa and self.current_epoch >= self.swa_start_epoch and self.swa_model:
            self.swa_model.update_parameters(self.model)

        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        lr = 3e-4
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=1e-2,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e3
        )

        # SWA wrapper and scheduler
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(optimizer, swa_lr=self.swa_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_end(self):
        if self.use_swa and self.swa_model:
            torch.optim.swa_utils.update_bn(self.trainer.datamodule.train_dataloader(), self.swa_model)
            self.model.load_state_dict(self.swa_model.module.state_dict())


import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import csv
from typing import List, Optional
from segmentation_models_pytorch.losses import MULTICLASS_MODE


class DeepLabV3PlusModel3RWithFreezeAndDropout(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        out_classes: List[str],
        class_weights,
        enc_weights: str = 'imagenet',
        mode: str = MULTICLASS_MODE,
        exp_name: Optional[str] = None,
        im_size: int = 512,
        loss_fn=None,
        freeze_encoder_stages: Optional[List[str]] = ['layer0', 'layer1', 'layer2'],
        encoder_dropout: float = 0.35,
        decoder_dropout: float = 0.25,
        **kwargs
    ):
        super().__init__()
        
        self.encoder_dropout = encoder_dropout

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=enc_weights,
            in_channels=in_channels,
            classes=len(out_classes),
            decoder_channels=512,
            decoder_atrous_rates=(6, 12, 24),
            **kwargs
        )

        # Freeze specific encoder stages
        if freeze_encoder_stages is not None:
            for name, module in self.model.encoder.named_children():
                if name in freeze_encoder_stages:
                    for param in module.parameters():
                        param.requires_grad = False

        # Add dropout after encoder output
        self.encoder_output_dropout = nn.Dropout2d(p=encoder_dropout) if encoder_dropout > 0 else nn.Identity()

        # Optionally add dropout after decoder output (before final classifier)
        if decoder_dropout > 0:
            old_segmentation_head = self.model.segmentation_head
            self.model.segmentation_head = nn.Sequential(
                nn.Dropout2d(p=decoder_dropout),
                old_segmentation_head
            )

        self.mode = mode
        self.im_size = im_size
        self.exp_name = exp_name or f"{encoder_name}_{im_size}"
        self.classes = out_classes
        self.number_of_classes = len(out_classes)
        self.weights = class_weights
        self.loss_fn = loss_fn or smp.losses.LovaszLoss(mode=mode, per_image=True)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        features = self.model.encoder(image)
        features[-1] = self.encoder_output_dropout(features[-1])
        decoder_output = self.model.decoder(*features)
        mask = self.model.segmentation_head(decoder_output)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch
        assert image.ndim == 4, f"image ndim === {image.ndim}"
        mask = mask.long()
        if self.mode == 'multilabel':
            mask = F.one_hot(mask, self.number_of_classes).permute(0, 3, 1, 2)
            assert mask.ndim == 4
        else:
            assert mask.ndim == 3

        logits_mask = self.forward(image)
        assert logits_mask.shape[1] == self.number_of_classes
        logits_mask = logits_mask.contiguous()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)
        if self.mode == 'multilabel':
            pred_mask = F.one_hot(pred_mask, self.number_of_classes).permute(0, 3, 1, 2)

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=self.mode, num_classes=self.number_of_classes
        )
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = torch.tensor([x["loss"].item() for x in outputs])
        self.log_classwise_IoU(stage, tp, fp, fn, tn, torch.mean(loss))

        mu_IoU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        ma_IoU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        self.log_dict({
            f"{stage}_LOSS": torch.mean(loss),
            f"{stage}_mu_IoU": mu_IoU,
            f"{stage}_ma_IoU": ma_IoU,
        }, prog_bar=True)

    def training_step(self, batch, batch_idx):
        result = self.shared_step(batch, "train")
        self.training_step_outputs.append(result)
        return result

    def validation_step(self, batch, batch_idx):
        result = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(result)
        return result

    def test_step(self, batch, batch_idx):
        result = self.shared_step(batch, "test")
        self.test_step_outputs.append(result)
        return result

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=3e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def log_classwise_IoU(self, stage, tp, fp, fn, tn, loss):
        classwise_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None, zero_division=0).mean(axis=0)
        log_file = f'EXPERIMENTS/{self.exp_name}/scores/{self.exp_name}_{stage}_classwise_iou.csv'
        file_exists = False
        try:
            with open(log_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['loss'] + self.classes)
            writer.writerow([loss] + classwise_iou.cpu().numpy().tolist())


