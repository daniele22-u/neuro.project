import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
import itertools
import torch


class WeightedCrossEntropyLoss(torch.nn.Module):
    # works for any kind of images
    def __init__(self, target: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self._weight = None
        if target is not None:
            self.update_weight(target)
        self.label_smoothing = label_smoothing

    def update_weight(self, target: torch.Tensor) -> None:
        # expected target with clases expressed as integers
        class_counts = torch.bincount(target.flatten(), minlength=2)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.sum()  # Normalize
        self._weight = class_weights

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Cross Entropy Loss for multi-class segmentation.

        # Args
            - x: Tensor of predictions logits (type float) (batch_size, C, K1, K2 [, K3]).
            - target: Ground truth (type longint) (batch_size, 1, K1, K2 [, K3]).

        # Returns
            - Scalar (torch.Tensor).
        """
        target = target.squeeze(1) # torch.nn.functional.cross_entropy() does not want target with a channel dimension
        return torch.nn.functional.cross_entropy(
            x, target, 
            weight=self._weight.to(x.device) if self._weight is not None else None, 
            reduction='mean', 
            label_smoothing=self.label_smoothing
        )


class DiceLoss(torch.nn.Module):
    # (F1 / Sorensen-Dice)
    # from:
    # https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    def __init__(
            self, 
            smooth: int=1, 
            class_reduction: Literal['none', 'mean', 'sum', 'foreground mean'] = 'mean', 
            batch_reduction: Literal['none', 'mean', 'sum'] = 'mean', 
            which: Literal['score', 'loss'] = 'loss'
        ):
        super().__init__()
        self.smooth = smooth
        self.class_reduction = class_reduction # applies class-wise
        self.batch_reduction = batch_reduction # applies batch element-wise # todo: to implement
        self.which = which

        # Runtime
        self.worst_indices = None # list[int]

        
    
    def __call__(self, x: torch.Tensor, target: torch.Tensor, x_is_probabilities: bool = False, save_worst_n_batch_indices: int = 0) -> torch.Tensor:
        """
        Computes Dice Loss for multi-class segmentation.

        # Args
            - x: Tensor of predictions logits (type float) (batch_size, C, K1, K2 [, K3]).
            - target: Ground truth (type longint) (batch_size, 1, K1, K2 [, K3]).
            - x_is_probabilities: (bool) wether x is probability in [0;1] instead of logits.

        # Returns
            - Scalar or vector Dice Loss (torch.Tensor).
        """
        if not x_is_probabilities:
            x = torch.nn.functional.softmax(x, dim=1)  # Convert logits to probabilities
        batch_size = x.shape[0]   # Number of samples (B)
        num_classes = x.shape[1]  # Number of classes (C)

        # Compute per-class, per-batch dice scores (higher better)
        dice = torch.zeros((batch_size, num_classes), device=x.device)
        for b, c in itertools.product(range(batch_size), range(num_classes)):
            pred_bc = x[b, c]
            target_bc = (target[b, 0] == c).float()
            intersection = (pred_bc * target_bc).sum()
            union = pred_bc.sum() + target_bc.sum()
            dice[b, c] = (2. * intersection + self.smooth) / (union + self.smooth)

        # If requested, return the worst samples
        worst_indices = None
        if (0 > save_worst_n_batch_indices) or (save_worst_n_batch_indices > batch_size):
            raise ValueError(print("Argument 'return_worst_n_batch_indices' must be in (0, batch_size)", True))
        if save_worst_n_batch_indices > 0:
            dice_avg_over_classes = dice.mean(dim=1).flatten()
            worst_indices = dice_avg_over_classes.argsort()
            worst_indices = worst_indices[0:save_worst_n_batch_indices]
            worst_indices = worst_indices.detach().cpu().numpy().tolist()
        self.worst_indices = worst_indices

        # Apply class reduction
        match self.class_reduction:
            case 'none':
                pass                     # nothing, -> (batch_size, num_classes)
            case 'mean':
                dice = dice.mean(dim=1)  # mean over classes, -> (batch_size,)
            case 'sum':
                dice = dice.sum(dim=1)   # sum over classes, -> (batch_size,)
            case 'foreground mean':
                # at leas one background (index 0) 
                # and one foreground (index 1 onward) 
                # are needed
                dice = dice[:, 1:].mean(dim=1)  # mean over foreground classes (all but first class dim), -> (batch_size,)
            case _:
                pass

        # Apply batch reduction
        match self.batch_reduction:
            case 'none':
                pass
            case 'mean':
                dice = dice.mean(dim=0)  # mean over batch
            case 'sum':
                dice = dice.sum(dim=0)   # sum over batch
            case _:
                pass
        
        match self.which:
            case 'score':
                pass
            case 'loss':
                dice = 1.0 - dice
            case _:
                pass
        
        return dice


class IoULoss(torch.nn.Module):
    # Jaccard
    # from:
    #
    def __init__(
            self, 
            smooth: int=1, 
            class_reduction: Literal['none', 'mean', 'sum', 'foreground mean'] = 'mean', 
            batch_reduction: Literal['none', 'mean', 'sum'] = 'mean', 
            which: Literal['score', 'loss'] = 'loss'
        ):
        super().__init__()
        self.smooth = smooth
        self.class_reduction = class_reduction # applies class-wise
        self.batch_reduction = batch_reduction # applies batch element-wise # todo: to implement
        self.which = which

        # Runtime
        self.worst_indices = None # list[int]

        
    
    def __call__(self, x: torch.Tensor, target: torch.Tensor, x_is_probabilities: bool = False, save_worst_n_batch_indices: int = 0) -> torch.Tensor:
        """
        Computes IoU Loss for multi-class segmentation.

        # Args
            - x: Tensor of predictions logits (type float) (batch_size, C, K1, K2 [, K3]).
            - target: Ground truth (type longint) (batch_size, 1, K1, K2 [, K3]).
            - x_is_probabilities: (bool) wether x is probability in [0;1] instead of logits.

        # Returns
            - Scalar or vector IoU Loss (torch.Tensor).
        """
        if not x_is_probabilities:
            x = torch.nn.functional.softmax(x, dim=1)  # Convert logits to probabilities
        batch_size = x.shape[0]   # Number of samples (B)
        num_classes = x.shape[1]  # Number of classes (C)

        # Compute per-class, per-batch IoU scores (higher better)
        iou = torch.zeros((batch_size, num_classes), device=x.device)
        for b, c in itertools.product(range(batch_size), range(num_classes)):
            pred_bc = x[b, c]
            target_bc = (target[b, 0] == c).float()
            intersection = (pred_bc * target_bc).sum()
            union = pred_bc.sum() + target_bc.sum() - intersection
            iou[b, c] = (intersection + self.smooth) / (union + self.smooth)

        # If requested, return the worst samples
        worst_indices = None
        if (0 > save_worst_n_batch_indices) or (save_worst_n_batch_indices > batch_size):
            raise ValueError(print("Argument 'return_worst_n_batch_indices' must be in (0, batch_size)", True))
        if save_worst_n_batch_indices > 0:
            iou_avg_over_classes = iou.mean(dim=1).flatten()
            worst_indices = iou_avg_over_classes.argsort()
            worst_indices = worst_indices[0:save_worst_n_batch_indices]
            worst_indices = worst_indices.detach().cpu().numpy().tolist()
        self.worst_indices = worst_indices

        # Apply class reduction
        match self.class_reduction:
            case 'none':
                pass                     # nothing, -> (batch_size, num_classes)
            case 'mean':
                iou = iou.mean(dim=1)  # mean over classes, -> (batch_size,)
            case 'sum':
                iou = iou.sum(dim=1)   # sum over classes, -> (batch_size,)
            case 'foreground mean':
                # at leas one background (index 0) 
                # and one foreground (index 1 onward) 
                # are needed
                iou = iou[:, 1:].mean(dim=1)  # mean over foreground classes (all but first class dim), -> (batch_size,)
            case _:
                pass

        # Apply batch reduction
        match self.batch_reduction:
            case 'none':
                pass
            case 'mean':
                iou = iou.mean(dim=0)  # mean over batch
            case 'sum':
                iou = iou.sum(dim=0)   # sum over batch
            case _:
                pass
        
        match self.which:
            case 'score':
                pass
            case 'loss':
                iou = 1.0 - iou
            case _:
                pass
        
        return iou
