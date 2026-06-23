# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


KEY_OUTPUT = 'metric_depth'


def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


class EdgeAwareGradientLoss(nn.Module):
    """Depth gradient loss with extra weight on RGB and GT depth edges."""
    def __init__(self, image_edge_weight=2.0, depth_edge_weight=4.0, eps=1e-6):
        super(EdgeAwareGradientLoss, self).__init__()
        self.name = 'EdgeGrad'
        self.image_edge_weight = image_edge_weight
        self.depth_edge_weight = depth_edge_weight
        self.eps = eps

    def _norm(self, x, valid):
        denom = x[valid].mean().detach() if valid.any() else x.mean().detach()
        return x / (denom + self.eps)

    def forward(self, input, target, image, mask=None, interpolate=True):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-2:] != target.shape[-2:] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)

        if image.shape[-2:] != target.shape[-2:]:
            image = nn.functional.interpolate(
                image, target.shape[-2:], mode='bilinear', align_corners=True)

        if target.ndim == 3:
            target = target.unsqueeze(1)
        if mask is None:
            mask = target > 0
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(torch.bool)

        input_log = torch.log(input.clamp_min(self.eps))
        target_log = torch.log(target.clamp_min(self.eps))
        gray = image.mean(dim=1, keepdim=True)

        pred_dx = input_log[:, :, :, 1:] - input_log[:, :, :, :-1]
        gt_dx = target_log[:, :, :, 1:] - target_log[:, :, :, :-1]
        img_dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        valid_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]

        pred_dy = input_log[:, :, 1:, :] - input_log[:, :, :-1, :]
        gt_dy = target_log[:, :, 1:, :] - target_log[:, :, :-1, :]
        img_dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        valid_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        depth_edge_x = gt_dx.abs()
        depth_edge_y = gt_dy.abs()
        image_edge_x = img_dx.abs()
        image_edge_y = img_dy.abs()

        weight_x = 1.0 + self.image_edge_weight * self._norm(image_edge_x, valid_dx) \
            + self.depth_edge_weight * self._norm(depth_edge_x, valid_dx)
        weight_y = 1.0 + self.image_edge_weight * self._norm(image_edge_y, valid_dy) \
            + self.depth_edge_weight * self._norm(depth_edge_y, valid_dy)

        loss_x = (pred_dx - gt_dx).abs() * weight_x
        loss_y = (pred_dy - gt_dy).abs() * weight_y
        losses = []
        if valid_dx.any():
            losses.append(loss_x[valid_dx].mean())
        if valid_dy.any():
            losses.append(loss_y[valid_dy].mean())
        if not losses:
            return input.sum() * 0.0
        return sum(losses) / len(losses)


def _as_nchw(x):
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def depth_discontinuity_target(depth, mask=None, threshold=0.15, alpha=20.0, eps=1e-6):
    depth = _as_nchw(depth)
    if mask is None:
        mask = depth > 0
    mask = _as_nchw(mask).to(torch.bool)
    log_depth = torch.log(depth.clamp_min(eps))

    dx = (log_depth[:, :, :, 1:] - log_depth[:, :, :, :-1]).abs()
    dy = (log_depth[:, :, 1:, :] - log_depth[:, :, :-1, :]).abs()
    valid_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]
    valid_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

    bx = torch.sigmoid(alpha * (dx - threshold)) * valid_dx.float()
    by = torch.sigmoid(alpha * (dy - threshold)) * valid_dy.float()

    target = torch.zeros_like(depth)
    valid = torch.zeros_like(mask)
    target[:, :, :, 1:] = torch.maximum(target[:, :, :, 1:], bx)
    target[:, :, :, :-1] = torch.maximum(target[:, :, :, :-1], bx)
    target[:, :, 1:, :] = torch.maximum(target[:, :, 1:, :], by)
    target[:, :, :-1, :] = torch.maximum(target[:, :, :-1, :], by)
    valid[:, :, :, 1:] |= valid_dx
    valid[:, :, :, :-1] |= valid_dx
    valid[:, :, 1:, :] |= valid_dy
    valid[:, :, :-1, :] |= valid_dy
    return target.detach(), valid.detach()


class BoundaryClassificationLoss(nn.Module):
    """Supervise predicted depth discontinuities from GT log-depth jumps."""
    def __init__(self, threshold=0.15, alpha=20.0, pos_weight=8.0):
        super(BoundaryClassificationLoss, self).__init__()
        self.name = 'BoundaryCls'
        self.threshold = threshold
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits, target, mask=None, interpolate=True):
        target = _as_nchw(target)
        if logits.shape[-2:] != target.shape[-2:] and interpolate:
            logits = F.interpolate(logits, target.shape[-2:], mode='bilinear', align_corners=True)
        boundary, valid = depth_discontinuity_target(
            target, mask=mask, threshold=self.threshold, alpha=self.alpha)
        if not valid.any():
            return logits.sum() * 0.0
        loss = F.binary_cross_entropy_with_logits(logits, boundary, reduction='none')
        weight = 1.0 + (self.pos_weight - 1.0) * boundary
        return (loss[valid] * weight[valid]).mean()


class BoundaryContrastLoss(nn.Module):
    """Match foreground/background log-depth jumps across GT discontinuities."""
    def __init__(self, threshold=0.15, eps=1e-6):
        super(BoundaryContrastLoss, self).__init__()
        self.name = 'BoundaryContrast'
        self.threshold = threshold
        self.eps = eps

    def forward(self, input, target, mask=None, interpolate=True):
        input = extract_key(input, KEY_OUTPUT)
        target = _as_nchw(target)
        if input.shape[-2:] != target.shape[-2:] and interpolate:
            input = F.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        if mask is None:
            mask = target > 0
        mask = _as_nchw(mask).to(torch.bool)

        pred_log = torch.log(input.clamp_min(self.eps))
        target_log = torch.log(target.clamp_min(self.eps))
        pred_dx = pred_log[:, :, :, 1:] - pred_log[:, :, :, :-1]
        gt_dx = target_log[:, :, :, 1:] - target_log[:, :, :, :-1]
        pred_dy = pred_log[:, :, 1:, :] - pred_log[:, :, :-1, :]
        gt_dy = target_log[:, :, 1:, :] - target_log[:, :, :-1, :]
        valid_dx = (mask[:, :, :, 1:] & mask[:, :, :, :-1]) & (gt_dx.abs() > self.threshold)
        valid_dy = (mask[:, :, 1:, :] & mask[:, :, :-1, :]) & (gt_dy.abs() > self.threshold)

        losses = []
        if valid_dx.any():
            losses.append((pred_dx - gt_dx).abs()[valid_dx].mean())
        if valid_dy.any():
            losses.append((pred_dy - gt_dy).abs()[valid_dy].mean())
        if not losses:
            return input.sum() * 0.0
        return sum(losses) / len(losses)


class BoundaryBandLoss(nn.Module):
    """Supervise final depth inside a small band around GT discontinuities."""
    def __init__(self, threshold=0.15, alpha=20.0, radius=3, eps=1e-6):
        super(BoundaryBandLoss, self).__init__()
        self.name = 'BoundaryBand'
        self.threshold = threshold
        self.alpha = alpha
        self.radius = radius
        self.eps = eps

    def _band(self, target, mask):
        boundary, valid = depth_discontinuity_target(
            target, mask=mask, threshold=self.threshold, alpha=self.alpha)
        hard_boundary = (boundary > 0.5).float()
        radius = max(int(self.radius), 0)
        if radius > 0:
            hard_boundary = F.max_pool2d(
                hard_boundary, kernel_size=2 * radius + 1, stride=1, padding=radius)
        return (hard_boundary > 0) & valid

    def forward(self, input, target, mask=None, interpolate=True):
        input = extract_key(input, KEY_OUTPUT)
        target = _as_nchw(target)
        if input.shape[-2:] != target.shape[-2:] and interpolate:
            input = F.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        if mask is None:
            mask = target > 0
        mask = _as_nchw(mask).to(torch.bool)
        band = self._band(target, mask)
        if not band.any():
            return input.sum() * 0.0
        pred_log = torch.log(input.clamp_min(self.eps))
        target_log = torch.log(target.clamp_min(self.eps))
        return (pred_log - target_log).abs()[band].mean()


class NonBoundaryPreserveLoss(nn.Module):
    """Keep residual refinement local by preserving the base depth away from boundaries."""
    def __init__(self, threshold=0.15, alpha=20.0, radius=5, eps=1e-6):
        super(NonBoundaryPreserveLoss, self).__init__()
        self.name = 'NonBoundaryPreserve'
        self.threshold = threshold
        self.alpha = alpha
        self.radius = radius
        self.eps = eps

    def forward(self, refined, base, target, mask=None, interpolate=True):
        refined = extract_key(refined, KEY_OUTPUT)
        target = _as_nchw(target)
        if refined.shape[-2:] != target.shape[-2:] and interpolate:
            refined = F.interpolate(refined, target.shape[-2:], mode='bilinear', align_corners=True)
        if base.shape[-2:] != target.shape[-2:] and interpolate:
            base = F.interpolate(base, target.shape[-2:], mode='bilinear', align_corners=True)
        if mask is None:
            mask = target > 0
        mask = _as_nchw(mask).to(torch.bool)

        boundary, valid = depth_discontinuity_target(
            target, mask=mask, threshold=self.threshold, alpha=self.alpha)
        boundary = (boundary > 0.5).float()
        radius = max(int(self.radius), 0)
        if radius > 0:
            boundary = F.max_pool2d(boundary, kernel_size=2 * radius + 1, stride=1, padding=radius)
        nonboundary = (boundary <= 0) & valid & mask
        if not nonboundary.any():
            return refined.sum() * 0.0
        refined_log = torch.log(refined.clamp_min(self.eps))
        base_log = torch.log(base.detach().clamp_min(self.eps))
        return (refined_log - base_log).abs()[nonboundary].mean()


class BoundaryAlignmentLoss(nn.Module):
    """Align predicted depth discontinuities to GT boundary locations."""
    def __init__(self, threshold=0.15, alpha=20.0, max_distance=5, pred_weight=1.0,
                 coverage_weight=1.0, eps=1e-6):
        super(BoundaryAlignmentLoss, self).__init__()
        self.name = 'BoundaryAlign'
        self.threshold = threshold
        self.alpha = alpha
        self.max_distance = max_distance
        self.pred_weight = pred_weight
        self.coverage_weight = coverage_weight
        self.eps = eps

    def _soft_boundary(self, depth, mask):
        log_depth = torch.log(depth.clamp_min(self.eps))
        dx = (log_depth[:, :, :, 1:] - log_depth[:, :, :, :-1]).abs()
        dy = (log_depth[:, :, 1:, :] - log_depth[:, :, :-1, :]).abs()
        valid_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        valid_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        edge_x = torch.sigmoid(self.alpha * (dx - self.threshold)) * valid_dx.float()
        edge_y = torch.sigmoid(self.alpha * (dy - self.threshold)) * valid_dy.float()

        boundary = torch.maximum(F.pad(edge_x, (1, 0, 0, 0)), F.pad(edge_x, (0, 1, 0, 0)))
        boundary = torch.maximum(boundary, F.pad(edge_y, (0, 0, 1, 0)))
        boundary = torch.maximum(boundary, F.pad(edge_y, (0, 0, 0, 1)))

        valid = F.pad(valid_dx.float(), (1, 0, 0, 0)) + F.pad(valid_dx.float(), (0, 1, 0, 0))
        valid = valid + F.pad(valid_dy.float(), (0, 0, 1, 0)) + F.pad(valid_dy.float(), (0, 0, 0, 1))
        valid = valid > 0
        return boundary, valid

    def _normalized_distance(self, hard_boundary, valid):
        max_distance = max(int(self.max_distance), 1)
        boundary = hard_boundary.float()
        reached = boundary > 0
        distance = torch.full_like(boundary, float(max_distance))
        distance = torch.where(reached, torch.zeros_like(distance), distance)

        dilated = boundary
        for step in range(1, max_distance + 1):
            dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
            newly_reached = (dilated > 0) & (~reached)
            distance = torch.where(newly_reached, torch.full_like(distance, float(step)), distance)
            reached |= newly_reached

        return (distance / float(max_distance)).clamp(0, 1) * valid.float()

    def forward(self, input, target, mask=None, interpolate=True):
        input = extract_key(input, KEY_OUTPUT)
        target = _as_nchw(target)
        if input.shape[-2:] != target.shape[-2:] and interpolate:
            input = F.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        if mask is None:
            mask = target > 0
        mask = _as_nchw(mask).to(torch.bool)

        pred_edge, valid = self._soft_boundary(input, mask)
        target_edge, _ = depth_discontinuity_target(
            target, mask=mask, threshold=self.threshold, alpha=self.alpha)
        hard_target_edge = (target_edge > 0.5) & valid
        if not hard_target_edge.any():
            return input.sum() * 0.0

        target_dt = self._normalized_distance(hard_target_edge.detach(), valid).detach()
        pred_to_target = (pred_edge * target_dt)[valid].mean()

        radius = max(int(self.max_distance), 1)
        local_pred = F.max_pool2d(pred_edge, kernel_size=2 * radius + 1, stride=1, padding=radius)
        target_weight = target_edge.detach()
        coverage = (target_weight * (1.0 - local_pred).clamp_min(0))[valid].sum()
        coverage = coverage / target_weight[valid].sum().clamp_min(1.0)

        return self.pred_weight * pred_to_target + self.coverage_weight * coverage


class NonBoundarySmoothLoss(nn.Module):
    """Smooth prediction only where GT says depth should be continuous."""
    def __init__(self, threshold=0.15, eps=1e-6):
        super(NonBoundarySmoothLoss, self).__init__()
        self.name = 'NonBoundarySmooth'
        self.threshold = threshold
        self.eps = eps

    def forward(self, input, target, mask=None, interpolate=True):
        input = extract_key(input, KEY_OUTPUT)
        target = _as_nchw(target)
        if input.shape[-2:] != target.shape[-2:] and interpolate:
            input = F.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        if mask is None:
            mask = target > 0
        mask = _as_nchw(mask).to(torch.bool)

        pred_log = torch.log(input.clamp_min(self.eps))
        target_log = torch.log(target.clamp_min(self.eps))
        pred_dx = pred_log[:, :, :, 1:] - pred_log[:, :, :, :-1]
        gt_dx = target_log[:, :, :, 1:] - target_log[:, :, :, :-1]
        pred_dy = pred_log[:, :, 1:, :] - pred_log[:, :, :-1, :]
        gt_dy = target_log[:, :, 1:, :] - target_log[:, :, :-1, :]
        valid_dx = (mask[:, :, :, 1:] & mask[:, :, :, :-1]) & (gt_dx.abs() <= self.threshold)
        valid_dy = (mask[:, :, 1:, :] & mask[:, :, :-1, :]) & (gt_dy.abs() <= self.threshold)

        losses = []
        if valid_dx.any():
            losses.append(pred_dx.abs()[valid_dx].mean())
        if valid_dy.any():
            losses.append(pred_dy.abs()[valid_dy].mean())
        if not losses:
            return input.sum() * 0.0
        return sum(losses) / len(losses)


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N,one, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""
    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        # self._loss_func = nn.NLLLoss(ignore_index=self.ignore_index)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        # depth : N1HW
        # output : NCHW

        # Quantize depth log-uniformly on [1, self.beta] into self.depth_bins bins
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth) 
        depth = depth.long()
        return depth
        

    
    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        # Get the center of the bin




    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        # assert torch.all(input <= 0), "Input should be negative"

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # assert torch.all(input)<=1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Set the mask to ignore_index
            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        

        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input
    



def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction


        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss
        return loss, intr_input




if __name__ == '__main__':
    # Tests for DiscreteNLLLoss
    celoss = DiscreteNLLLoss()
    print(celoss(torch.rand(4, 64, 26, 32)*10, torch.rand(4, 1, 26, 32)*10, ))

    d = torch.Tensor([6.59, 3.8, 10.0])
    print(celoss.dequantize_depth(celoss.quantize_depth(d)))
