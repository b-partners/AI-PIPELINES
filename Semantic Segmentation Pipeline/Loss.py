from segmentation_models_pytorch.losses import DiceLoss, MULTICLASS_MODE, MULTILABEL_MODE, BINARY_MODE, LovaszLoss
from segmentation_models_pytorch.losses._functional import soft_tversky_score
from typing import List, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.modules.loss import _Loss

from segmentation_models_pytorch.losses.lovasz import _lovasz_softmax



class WeightedLovaszLoss2(LovaszLoss):
    def __init__(
        self,
        mode: str,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        from_logits: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        dynamic_weighting: bool = False,
        focal_weighting: bool = False,
        gamma: float = 1.0,
        lambda_conf: float = 0.0,
        eps: float = 1e-7,
    ):
        super().__init__(
            mode=mode,
            per_image=per_image,
            ignore_index=ignore_index,
            from_logits=from_logits,
        )

        self.class_weights = class_weights
        self.dynamic_weighting = dynamic_weighting
        self.focal_weighting = focal_weighting
        self.gamma = gamma
        self.lambda_conf = lambda_conf
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        if self.mode != MULTICLASS_MODE:
            raise NotImplementedError("WeightedLovaszLoss2 currently supports only multiclass mode.")

        # Dynamic or static weights
        weights = self.class_weights
        if self.dynamic_weighting:
            with torch.no_grad():
                flat_labels = y_true.view(-1)
                C = y_pred.shape[1]
                pixel_counts = torch.stack([(flat_labels == c).sum() for c in range(C)])
                freqs = pixel_counts.float() / pixel_counts.sum().clamp(min=1)
                weights = 1.0 / (torch.log(1.02 + freqs))
                weights = weights / weights.sum()

        # Flatten and compute base Lovasz loss
        probas, labels = self._flatten_multiclass_probas(y_pred, y_true, self.ignore_index)
        losses = []

        C = y_pred.shape[1]
        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            class_pred = probas[:, c]
            errors = (fg - class_pred).abs()

            # Focal weighting
            if self.focal_weighting:
                pt = 1.0 - errors
                focal_weight = (1 - pt).pow(self.gamma)
                errors = errors * focal_weight

            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            loss_c = torch.dot(errors_sorted, grad)

            if weights is not None:
                loss_c *= weights[c].to(loss_c.device)

            losses.append(loss_c)

        loss = torch.stack(losses).mean()

        # Confidence penalty
        if self.lambda_conf > 0:
            entropy = -y_pred.clamp(min=self.eps).log() * y_pred
            conf_penalty = entropy.sum(dim=1).mean()
            loss += self.lambda_conf * conf_penalty

        return loss

    @staticmethod
    def _flatten_multiclass_probas(probas: torch.Tensor, labels: torch.Tensor, ignore: Optional[int] = None):
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
        labels = labels.view(-1)
        if ignore is not None:
            valid = labels != ignore
            probas = probas[valid]
            labels = labels[valid]
        return probas, labels

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


class WeightedLovaszLoss(_Loss):
    def __init__(
        self,
        mode: str,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        from_logits: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        dynamic_weighting: bool = False,
        focal_weighting: bool = False,
        gamma: float = 1.0,
        lambda_conf: float = 0.0,
        eps: float = 1e-7,
    ):
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.from_logits = from_logits

        self.class_weights = class_weights
        self.dynamic_weighting = dynamic_weighting
        self.focal_weighting = focal_weighting
        self.gamma = gamma
        self.lambda_conf = lambda_conf
        self.eps = eps

    def forward(self, y_pred, y_true):
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        if self.mode == MULTICLASS_MODE:
            loss = self._weighted_lovasz_softmax(
                y_pred, y_true,
                per_image=self.per_image,
                ignore=self.ignore_index
            )
        else:
            raise NotImplementedError("This extended version only supports multiclass mode.")

        return loss

    def _weighted_lovasz_softmax(self, probas, labels, per_image=False, ignore=None):
        if per_image:
            losses = []
            for prob, lab in zip(probas, labels):
                loss = self._lovasz_softmax_flat_weighted(
                    *_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore)
                )
                losses.append(loss)
            return torch.stack(losses).mean()
        else:
            return self._lovasz_softmax_flat_weighted(
                *_flatten_probas(probas, labels, ignore)
            )

    def _lovasz_softmax_flat_weighted(self, probas, labels):
        """Compute lovasz softmax loss with optional weights/focal terms"""
        if probas.numel() == 0:
            return probas.sum() * 0.0

        C = probas.size(1)
        total_pixels = labels.numel()

        losses = []
        weights = self.class_weights
        if self.dynamic_weighting:
            with torch.no_grad():
                freqs = torch.stack([(labels == c).float().sum() for c in range(C)])
                freqs = freqs / freqs.sum().clamp(min=self.eps)
                weights = 1.0 / torch.log(1.02 + freqs)
                weights = weights / weights.sum()

        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            class_pred = probas[:, c]
            errors = (fg - class_pred).abs()

            # Focal scaling
            if self.focal_weighting:
                pt = 1.0 - errors  # pt = |1 - error| → high for correct prediction
                focal_weight = (1 - pt).pow(self.gamma)
                errors *= focal_weight

            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]

            grad = _lovasz_grad(fg_sorted)
            loss_c = torch.dot(errors_sorted, grad)

            if weights is not None:
                loss_c *= weights[c].to(loss_c.device)

            losses.append(loss_c)

        final_loss = torch.mean(torch.stack(losses))

        # Add confidence penalty (entropy of prediction)
        if self.lambda_conf > 0:
            entropy = -probas.clamp(min=self.eps).log() * probas
            conf_penalty = entropy.sum(dim=1).mean()  # mean over all pixels
            final_loss += self.lambda_conf * conf_penalty

        return final_loss


class WeightedDiceLoss(DiceLoss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        thresh: Optional[float] = None, 
        lambda_conf: float= 0.0,
        eps: float = 1e-7,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float= 0.5,
        beta: float= 0.5,
        gamma: float= 1.0,
    ):
        """
        Weighted Dice loss for image segmentation.

        Args:
            class_weights: A tensor of shape (C,) where C is the number of classes, 
                           specifying the weight for each class.
        """
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps,
        )

        self.alpha= alpha
        self.beta= beta
        self.gamma= gamma
        self.thresh= thresh
        self.lambda_conf= lambda_conf
        
        if class_weights is not None:
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch.Tensor"
            assert class_weights.dim() == 1, "class_weights must be a 1D tensor"
            
            self.class_weights = class_weights
        else:
            self.class_weights = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        if self.mode == MULTICLASS_MODE and self.thresh is not None:
                confs, _ = torch.max(y_pred, dim=1)
                # mask: 1 where confident, 0 where low confidence
                confident_mask = confs >= self.thresh
                # apply ignore_index where confidence is too low
                y_true = torch.where(confident_mask, y_true, torch.full_like(y_true, self.ignore_index))
        
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        if self.class_weights is not None:
            
            assert loss.shape == self.class_weights.shape, "Loss and class_weights shape mismatch"
            loss *= self.class_weights.to(loss.device)

        if self.lambda_conf > 0:
            # Avoid log(0) with clamp
            conf_penalty = y_pred * (y_pred.clamp_min(self.eps).log())
            conf_penalty = -conf_penalty.sum(dim=1)  # Entropy over classes
            conf_penalty = conf_penalty.mean()
            loss += self.lambda_conf * conf_penalty

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return soft_tversky_score(
            output, target, self.alpha, self.beta, smooth, eps, dims
        )





class WeightedDiceLoss2(DiceLoss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        thresh: Optional[float] = None,
        lambda_conf: float = 0.0,
        eps: float = 1e-7,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        dynamic_weighting: bool = False,
        focal_weighting: bool = False,
        exclude_background_from_mean: bool = False,
    ):
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps,
        )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.thresh = thresh
        self.lambda_conf = lambda_conf
        self.exclude_background_from_mean = exclude_background_from_mean
        self.dynamic_weighting = dynamic_weighting
        self.focal_weighting = focal_weighting

        if class_weights is not None:
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch.Tensor"
            assert class_weights.dim() == 1, "class_weights must be a 1D tensor"
            self.class_weights = class_weights
        else:
            self.class_weights = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.from_logits:
            if self.mode == 'multiclass':
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        if self.mode == 'multiclass' and self.thresh is not None:
            confs, _ = torch.max(y_pred, dim=1)
            confident_mask = confs >= self.thresh
            y_true = torch.where(confident_mask, y_true, torch.full_like(y_true, self.ignore_index))

        if self.mode == 'binary':
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        elif self.mode == 'multiclass':
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes).permute(0, 2, 1)

        elif self.mode == 'multilabel':
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        # Dynamic class weights from current batch
        weights = self.class_weights
        if self.dynamic_weighting:
            with torch.no_grad():
                pixel_per_class = y_true.sum(dim=(0, 2))
                total_pixels = pixel_per_class.sum()
                class_frequencies = pixel_per_class.float() / total_pixels.clamp(min=1)
                weights = 1.0 / (torch.log(1.02 + class_frequencies))
                weights = weights / weights.sum()
        
        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        # Select specific class subset if needed
        if self.classes is not None:
            loss = loss[self.classes]
            if weights is not None:
                weights = weights[self.classes]

        # Exclude background from mean (background is class 0)
        if self.exclude_background_from_mean:
            loss = loss[1:]
            if weights is not None:
                weights = weights[1:]

        # Apply class weights
        if weights is not None:
            weights = weights.to(loss.device)
            loss = loss * weights

        # Add optional confidence penalty
        if self.lambda_conf > 0:
            conf_penalty = y_pred * (y_pred.clamp_min(self.eps).log())
            conf_penalty = -conf_penalty.sum(dim=1).mean()
            loss += self.lambda_conf * conf_penalty
        
        # Apply focal-like power scaling
        if self.focal_weighting:
            loss = (loss ** self.gamma)
        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        if not self.focal_weighting:
            return loss.mean() ** self.gamma
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)


class WeightedDiceLoss3(DiceLoss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        thresh: Optional[float] = None,
        lambda_conf: float = 0.0,
        eps: float = 1e-7,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        dynamic_weighting: bool = False,
        focal_weighting: bool = False,
        exclude_background_from_mean: bool = False,
        present: bool = False,
    ):
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps,
        )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.thresh = thresh
        self.lambda_conf = lambda_conf
        self.exclude_background_from_mean = exclude_background_from_mean
        self.dynamic_weighting = dynamic_weighting
        self.focal_weighting = focal_weighting
        self.present = present

        if class_weights is not None:
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch.Tensor"
            assert class_weights.dim() == 1, "class_weights must be a 1D tensor"
            self.class_weights = class_weights
        else:
            self.class_weights = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.from_logits:
            if self.mode == 'multiclass':
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        if self.mode == 'multiclass' and self.thresh is not None:
            confs, _ = torch.max(y_pred, dim=1)
            confident_mask = confs >= self.thresh
            y_true = torch.where(confident_mask, y_true, torch.full_like(y_true, self.ignore_index))

        if self.mode == 'binary':
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        elif self.mode == 'multiclass':
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes).permute(0, 2, 1)

        elif self.mode == 'multilabel':
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        # Dynamic class weights from current batch
        weights = self.class_weights
        if self.dynamic_weighting:
            with torch.no_grad():
                pixel_per_class = y_true.sum(dim=(0, 2))
                total_pixels = pixel_per_class.sum()
                class_frequencies = pixel_per_class.float() / total_pixels.clamp(min=1)
                weights = 1.0 / (torch.log(1.02 + class_frequencies))
                weights = weights / weights.sum()

        # Compute Dice/Tversky score
        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        # Apply log loss
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Determine which classes to keep
        if self.classes is not None:
            classes = torch.tensor(self.classes, device=y_pred.device)
        else:
            classes = torch.arange(num_classes, device=y_pred.device)

        # Optionally filter only classes present in the batch
        if self.present:
            present_mask = (y_true.sum(dims) > 0)
            classes = classes[present_mask[classes]]

        # Filter loss and weights
        loss = loss[classes]
        if weights is not None:
            weights = weights[classes]

        # Optionally exclude background class
        if self.exclude_background_from_mean:
            keep = classes != 0
            loss = loss[keep]
            if weights is not None:
                weights = weights[keep]

        # Apply class weights
        if weights is not None:
            weights = weights.to(loss.device)
            loss = loss * weights

        # Add optional confidence penalty
        if self.lambda_conf > 0:
            conf_penalty = y_pred * (y_pred.clamp_min(self.eps).log())
            conf_penalty = -conf_penalty.sum(dim=1).mean()
            loss += self.lambda_conf * conf_penalty

        # Focal-style gamma scaling
        if self.focal_weighting:
            loss = loss ** self.gamma

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        if not self.focal_weighting:
            return loss.mean() ** self.gamma
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)


class IoULoss(nn.Module):
    def __init__(self, reduction='mean', weights=None, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps  # Small constant to avoid division by zero
        self.reduction = reduction
        self.weights= weights

    def forward(self, y_pred, y_true,):
        """
        Compute IoU loss: 1 - IoU per image, averaging only over the classes present in the image.
        Supports optional per-image weighting.

        Args:
            y_pred (Tensor): (B, C, H, W) Raw logits from the model.
            y_true (Tensor): (B, H, W) Ground truth mask with class indices.
            weight (Tensor, optional): (B,) Weights for each image loss.
            reduction (str, optional): 'mean' returns a single scalar loss,
                                       'none' returns loss per image. Default is 'mean'.

        Returns:
            Tensor: IoU loss. A scalar if reduction=='mean', or a (B,) tensor if reduction=='none'.
        """
        # Convert logits to probabilities
        y_pred_probs = F.softmax(y_pred, dim=1)  # (B, C, H, W)

        # Convert y_true to one-hot encoding and reshape to (B, C, H, W)
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1])  # (B, H, W, C)
        y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute intersection and union per image and class
        intersection = torch.sum(y_pred_probs * y_true_one_hot, dim=(2, 3))  # (B, C)
        union = (torch.sum(y_pred_probs, dim=(2, 3)) +
                 torch.sum(y_true_one_hot, dim=(2, 3)) - intersection)  # (B, C)

        # Compute IoU per image and per class
        iou = (intersection + self.eps) / (union + self.eps)  # (B, C)

        # Create a mask for classes present in the ground truth (i.e., at least one pixel)
        class_present = torch.sum(y_true_one_hot, dim=(2, 3)) > 0  # (B, C) boolean mask
        class_present_float = class_present.float()  # Convert mask to float for computations

        # Sum IoU for only the classes present and count them for each image
        iou_sum = (iou * class_present_float).sum(dim=1)  # (B,)
        num_classes_present = class_present_float.sum(dim=1)  # (B,)

        # Compute the mean IoU over the classes present (avoid division by zero)
        mean_iou = iou_sum / (num_classes_present + self.eps)  # (B,)

        # IoU loss per image
        loss = 1 - mean_iou  # (B,)

        # Reduction options: weighted mean or per-image loss
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            if weight is not None:
                weight = weight.to(loss.device)
                return (loss * weight).sum() / weight.sum()
            else:
                return loss.mean()
        else:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'none'.")
