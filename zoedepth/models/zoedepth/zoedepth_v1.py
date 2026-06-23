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

import itertools

import torch
import torch.nn as nn
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import (Projector, SeedBinRegressor,
                                            SeedBinRegressorUnnormed)
from zoedepth.models.model_io import load_state_from_resource


class FrozenDepthAnythingPrior(nn.Module):
    """Frozen Depth Anything feature/prior branch for lightweight feature fusion."""
    def __init__(self, model_name, feature_channels=16, input_size=384, eps=1e-6):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ImportError(
                "Frozen Depth Anything prior requires transformers. "
                "Install it or disable use_frozen_da_prior."
            ) from exc

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.input_size = input_size
        self.eps = eps
        self.adapter = nn.Sequential(
            nn.Conv2d(1, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        image_mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _preprocess(self, x):
        x = x.clamp(0, 1)
        if self.input_size and self.input_size > 0:
            x = nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return (x - self.image_mean.to(x.device, x.dtype)) / self.image_std.to(x.device, x.dtype)

    @torch.no_grad()
    def _predict_prior(self, x):
        pixel_values = self._preprocess(x)
        prediction = self.model(pixel_values=pixel_values).predicted_depth
        prior = prediction.unsqueeze(1)
        prior = torch.log(prior.clamp_min(self.eps))
        prior_mean = prior.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
        prior_std = prior.flatten(1).std(dim=1).view(-1, 1, 1, 1).clamp_min(self.eps)
        return (prior - prior_mean) / prior_std

    def forward(self, x, target_size):
        prior = self._predict_prior(x)
        prior = nn.functional.interpolate(
            prior, size=target_size, mode="bilinear", align_corners=True)
        return self.adapter(prior)


class ZoeDepth(DepthModel):
    def __init__(self, core,  n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3, max_depth=10,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', min_temp=5, max_temp=50, train_midas=True,
                 midas_lr_factor=10, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False, use_boundary_refine=False,
                 boundary_refine_channels=32, boundary_refine_scale=0.1,
                 boundary_refine_mode="scale", boundary_refine_use_da_prior=False,
                 use_discontinuity_branch=False,
                 discontinuity_channels=32, use_discontinuity_temperature=False,
                 discontinuity_temperature_scale=1.5, use_frozen_da_prior=False,
                 frozen_da_model="depth-anything/Depth-Anything-V2-Small-hf",
                 frozen_da_feature_channels=16, frozen_da_input_size=384,
                 frozen_da_fusion_scale=0.1, use_frozen_da_boundary_gate=False,
                 frozen_da_min_gate=0.05, **kwargs):
        """ZoeDepth model. This is the version of ZoeDepth that has a single metric head

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.
        """
        super().__init__()

        self.core = core
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.train_midas = train_midas
        self.inverse_midas = inverse_midas
        self.use_boundary_refine = use_boundary_refine
        self.boundary_refine_scale = boundary_refine_scale
        self.boundary_refine_mode = boundary_refine_mode
        self.boundary_refine_use_da_prior = boundary_refine_use_da_prior
        self.use_discontinuity_branch = use_discontinuity_branch
        self.use_discontinuity_temperature = use_discontinuity_temperature
        self.discontinuity_temperature_scale = discontinuity_temperature_scale
        self.use_frozen_da_prior = use_frozen_da_prior
        self.frozen_da_fusion_scale = frozen_da_fusion_scale
        self.use_frozen_da_boundary_gate = use_frozen_da_boundary_gate
        self.frozen_da_min_gate = frozen_da_min_gate

        if self.encoder_lr_factor <= 0:
            self.core.freeze_encoder(
                freeze_rel_pos=self.pos_enc_lr_factor <= 0)

        N_MIDAS_OUT = 32
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features,
                               kernel_size=1, stride=1, padding=0)  # btlnck conv

        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")

        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])
        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        if self.use_frozen_da_prior:
            self.frozen_da_prior = FrozenDepthAnythingPrior(
                frozen_da_model,
                feature_channels=frozen_da_feature_channels,
                input_size=frozen_da_input_size)
            self.frozen_da_fuser = nn.Sequential(
                nn.Conv2d(last_in + frozen_da_feature_channels, last_in, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(last_in, last_in, kernel_size=3, padding=1),
            )

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

        if self.use_boundary_refine:
            boundary_refine_in = 4
            if self.boundary_refine_mode == "log_residual":
                boundary_refine_in += 1
                if self.use_frozen_da_prior and self.boundary_refine_use_da_prior:
                    boundary_refine_in += frozen_da_feature_channels
            self.boundary_refiner = nn.Sequential(
                nn.Conv2d(boundary_refine_in, boundary_refine_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(boundary_refine_channels, boundary_refine_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(boundary_refine_channels, 1, kernel_size=3, padding=1),
            )
        if self.use_discontinuity_branch:
            self.discontinuity_head = nn.Sequential(
                nn.Conv2d(last_in + bin_embedding_dim + 3, discontinuity_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(discontinuity_channels, discontinuity_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(discontinuity_channels, 1, kernel_size=1),
            )

    def forward(self, x, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.
        
        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """
        b, c, h, w = x.shape
        x_input = x
        # print("input shape ", x.shape)
        self.orig_input_width = w
        self.orig_input_height = h
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)
        # print("output shapes", rel_depth.shape, out.shape)

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        if self.inverse_midas:
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-6)
            rel_depth = (rel_depth - rel_depth.min()) / \
                (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        boundary_logits = None
        boundary_prob = None
        if self.use_discontinuity_branch:
            image_context = nn.functional.interpolate(
                x_input, size=last.shape[-2:], mode='bilinear', align_corners=True)
            boundary_logits = self.discontinuity_head(
                torch.cat([last, b_embedding, image_context], dim=1))
            boundary_prob = torch.sigmoid(boundary_logits)

        da_prior = None
        da_gate = None
        if self.use_frozen_da_prior:
            da_prior = self.frozen_da_prior(x_input, target_size=last.shape[-2:])
            da_residual = self.frozen_da_fuser(torch.cat([last, da_prior], dim=1))
            if self.use_frozen_da_boundary_gate and boundary_prob is not None:
                da_gate = self.frozen_da_min_gate + (1.0 - self.frozen_da_min_gate) * boundary_prob
                da_residual = da_residual * da_gate
            last = last + self.frozen_da_fusion_scale * da_residual

        temperature_scale = None
        if boundary_prob is not None and self.use_discontinuity_temperature:
            temperature_scale = 1.0 / (1.0 + self.discontinuity_temperature_scale * boundary_prob)
        x = self.conditional_log_binomial(last, b_embedding, temperature_scale=temperature_scale)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)
        out_base = out
        boundary_refine_mask = None
        boundary_log_residual = None
        if self.use_boundary_refine:
            image_context = nn.functional.interpolate(
                x_input, size=out.shape[-2:], mode='bilinear', align_corners=True)
            depth_context = out / self.max_depth
            gate = 1.0
            if boundary_prob is not None:
                gate = nn.functional.interpolate(
                    boundary_prob, size=out.shape[-2:], mode='bilinear', align_corners=True)
            if self.boundary_refine_mode == "log_residual":
                boundary_refine_mask = gate if torch.is_tensor(gate) else torch.ones_like(out)
                refiner_inputs = [
                    image_context,
                    torch.log(out.clamp_min(self.min_depth)) / torch.log(
                        torch.as_tensor(self.max_depth, device=out.device, dtype=out.dtype)),
                    boundary_refine_mask,
                ]
                if da_prior is not None and self.boundary_refine_use_da_prior:
                    refiner_inputs.append(nn.functional.interpolate(
                        da_prior, size=out.shape[-2:], mode='bilinear', align_corners=True))
                residual = self.boundary_refiner(torch.cat(refiner_inputs, dim=1))
                boundary_log_residual = self.boundary_refine_scale * boundary_refine_mask * torch.tanh(residual)
                out = torch.exp(torch.log(out.clamp_min(self.min_depth)) + boundary_log_residual)
            else:
                residual = self.boundary_refiner(torch.cat([image_context, depth_context], dim=1))
                out = out * (1 + self.boundary_refine_scale * gate * torch.tanh(residual))
            out = out.clamp(min=self.min_depth, max=self.max_depth)

        # Structure output dict
        output = dict(metric_depth=out, metric_depth_base=out_base)
        if boundary_refine_mask is not None:
            output['boundary_refine_mask'] = boundary_refine_mask
        if boundary_log_residual is not None:
            output['boundary_log_residual'] = boundary_log_residual
        if da_prior is not None:
            output['frozen_da_prior'] = da_prior
        if da_gate is not None:
            output['frozen_da_gate'] = da_gate
        if boundary_logits is not None:
            output['boundary_logits'] = boundary_logits
            output['boundary_prob'] = boundary_prob
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x

        return output

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            if self.encoder_lr_factor > 0:
                param_conf.append({'params': self.core.get_enc_params_except_rel_pos(
                ), 'lr': lr / self.encoder_lr_factor})

            if self.pos_enc_lr_factor > 0:
                param_conf.append(
                    {'params': self.core.get_rel_pos_params(), 'lr': lr / self.pos_enc_lr_factor})

            midas_params = self.core.core.scratch.parameters()
            midas_lr_factor = self.midas_lr_factor
            param_conf.append(
                {'params': midas_params, 'lr': lr / midas_lr_factor})

        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules])

        param_conf.append({'params': remaining_params, 'lr': lr})

        return param_conf

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, **kwargs):
        core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                               train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)
        model = ZoeDepth(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepth.build(**config)
