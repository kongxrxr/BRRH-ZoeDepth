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
import torch.cuda.amp as amp
import torch.nn as nn

from zoedepth.trainers.loss import (BoundaryAlignmentLoss, BoundaryBandLoss, BoundaryClassificationLoss,
                                    BoundaryContrastLoss, EdgeAwareGradientLoss, GradL1Loss,
                                    NonBoundaryPreserveLoss, NonBoundarySmoothLoss, SILogLoss)
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics
from zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer
from torchvision import transforms
from PIL import Image
import numpy as np

class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss()
        self.grad_loss = GradL1Loss()
        self.edge_grad_loss = EdgeAwareGradientLoss(
            image_edge_weight=self.config.get("edge_image_weight", 2.0),
            depth_edge_weight=self.config.get("edge_depth_weight", 4.0))
        boundary_threshold = self.config.get("boundary_log_grad_threshold", 0.15)
        self.boundary_cls_loss = BoundaryClassificationLoss(
            threshold=boundary_threshold,
            alpha=self.config.get("boundary_target_alpha", 20.0),
            pos_weight=self.config.get("boundary_pos_weight", 8.0))
        self.boundary_contrast_loss = BoundaryContrastLoss(threshold=boundary_threshold)
        self.boundary_band_loss = BoundaryBandLoss(
            threshold=boundary_threshold,
            alpha=self.config.get("boundary_target_alpha", 20.0),
            radius=self.config.get("boundary_band_radius", 3))
        self.nonboundary_preserve_loss = NonBoundaryPreserveLoss(
            threshold=boundary_threshold,
            alpha=self.config.get("boundary_target_alpha", 20.0),
            radius=self.config.get("nonboundary_preserve_radius", 5))
        self.boundary_align_loss = BoundaryAlignmentLoss(
            threshold=boundary_threshold,
            alpha=self.config.get("boundary_target_alpha", 20.0),
            max_distance=self.config.get("boundary_align_max_distance", 5),
            pred_weight=self.config.get("boundary_align_pred_weight", 1.0),
            coverage_weight=self.config.get("boundary_align_coverage_weight", 1.0))
        self.nonboundary_smooth_loss = NonBoundarySmoothLoss(threshold=boundary_threshold)
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.grad_accum_steps = max(int(self.config.get("grad_accum_steps", 1)), 1)
        self.memory_log_every = int(self.config.get("memory_log_every", 0) or 0)

    def _reset_memory_peak_if_needed(self):
        if self.memory_log_every > 0 and torch.cuda.is_available() and self.step % self.memory_log_every == 0:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _log_memory_if_needed(self, train_step, optimizer_stepped):
        if self.memory_log_every <= 0 or not torch.cuda.is_available():
            return
        completed_step = self.step + 1
        if completed_step % self.memory_log_every != 0:
            return
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        max_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
        print(
            f"[cuda-memory] step={completed_step} batch={train_step} "
            f"optimizer_stepped={int(optimizer_stepped)} "
            f"allocated={allocated:.3f}GiB reserved={reserved:.3f}GiB "
            f"max_allocated={max_allocated:.3f}GiB max_reserved={max_reserved:.3f}GiB",
            flush=True,
        )

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """

        images, depths_gt = batch['image'].to(
            self.device), batch['depth'].to(self.device)
        dataset = batch['dataset'][0]

        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)

        losses = {}
        self._reset_memory_peak_if_needed()

        with amp.autocast(enabled=self.config.use_amp):

            output = self.model(images)
            pred_depths = output['metric_depth']

            l_si, pred = self.silog_loss(
                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
            loss = self.config.w_si * l_si
            losses[self.silog_loss.name] = l_si

            if self.config.w_grad > 0:
                l_grad = self.grad_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_grad * l_grad
                losses[self.grad_loss.name] = l_grad

            if self.config.get("w_edge", 0) > 0:
                l_edge = self.edge_grad_loss(pred, depths_gt, images, mask=mask)
                loss = loss + self.config.w_edge * l_edge
                losses[self.edge_grad_loss.name] = l_edge

            if self.config.get("w_boundary_cls", 0) > 0 and "boundary_logits" in output:
                l_boundary = self.boundary_cls_loss(output["boundary_logits"], depths_gt, mask=mask)
                loss = loss + self.config.w_boundary_cls * l_boundary
                losses[self.boundary_cls_loss.name] = l_boundary

            if self.config.get("w_boundary_contrast", 0) > 0:
                l_contrast = self.boundary_contrast_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_boundary_contrast * l_contrast
                losses[self.boundary_contrast_loss.name] = l_contrast

            if self.config.get("w_boundary_band", 0) > 0:
                l_band = self.boundary_band_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_boundary_band * l_band
                losses[self.boundary_band_loss.name] = l_band

            if self.config.get("w_boundary_align", 0) > 0:
                l_align = self.boundary_align_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_boundary_align * l_align
                losses[self.boundary_align_loss.name] = l_align

            if self.config.get("w_nonboundary_smooth", 0) > 0:
                l_smooth = self.nonboundary_smooth_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_nonboundary_smooth * l_smooth
                losses[self.nonboundary_smooth_loss.name] = l_smooth

            if self.config.get("w_nonboundary_preserve", 0) > 0 and "metric_depth_base" in output:
                l_preserve = self.nonboundary_preserve_loss(
                    pred, output["metric_depth_base"], depths_gt, mask=mask)
                loss = loss + self.config.w_nonboundary_preserve * l_preserve
                losses[self.nonboundary_preserve_loss.name] = l_preserve

        if self.config.get("skip_nan_batches", False) and not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            losses["__skip_nan_batch__"] = torch.ones((), device=self.device)
            return losses

        scaled_loss = loss / self.grad_accum_steps
        self.scaler.scale(scaled_loss).backward()

        optimizer_stepped = ((self.step + 1) % self.grad_accum_steps == 0) or (
            train_step + 1 >= self.iters_per_epoch)

        if optimizer_stepped:
            if self.config.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad)

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_stepped = scale_after >= scale_before
            self.optimizer.zero_grad(set_to_none=True)

        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt[torch.logical_not(mask)] = -99

            self.log_images(rgb={"Input": images[0, ...].detach()}, depth={"GT": depths_gt[0].detach(), "PredictedMono": pred[0].detach()}, prefix="Train",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][0].detach()}, prefix="TrainRel")

        self._log_memory_if_needed(train_step, optimizer_stepped)
        losses = {name: value.detach() if torch.is_tensor(value) else value
                  for name, value in losses.items()}
        losses["__optimizer_stepped__"] = optimizer_stepped

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x):
        with amp.autocast(enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
            pred_depths = m(x)['metric_depth']
        return pred_depths

    @torch.no_grad()
    def crop_aware_infer(self, x):
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths



    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        mask = batch["mask"].to(self.device)
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
        if dataset == 'nyu':
            pred_depths = self.crop_aware_infer(images)
        else:
            pred_depths = self.eval_infer(images)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)

        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.silog_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)

        metrics = compute_metrics(depths_gt, pred_depths, **self.config)
        losses = {f"{self.silog_loss.name}": l_depth.item()}

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0].detach()}, depth={"GT": depths_gt[0].detach(), "PredictedMono": pred_depths[0].detach()}, prefix="Test",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        return metrics, losses
