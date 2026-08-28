import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RocketExtractor(nn.Module):
    """Deterministic, MiniRocket-inspired image feature extractor.

    Filters and colour projections are fixed. ``fit`` only estimates
    unlabelled response quantiles; no parameter here is gradient-trained.
    """

    def __init__(self, in_channels=1, input_shape=(128, 128), target_features=15000, fit_samples=512):
        super().__init__()
        self.in_channels = in_channels
        self.H, self.W = input_shape
        self.is_fitted = False
        self.num_kernels = 84
        self.num_biases = 3
        self.fit_samples = fit_samples

        # Three PPVs on 2x2 cells, signed magnitudes on 2x2 cells,
        # and a global absolute maximum for every dilation and filter.
        self.spatial_regions = self.num_biases * 4 + 4 + 4 + 1
        features_per_dilation = self.num_kernels * self.spatial_regions
        allowed_dilations = max(1, target_features // features_per_dilation)

        self.register_buffer("weight", self._generate_base_kernels())
        self.num_scattering_filters = 4
        self.register_buffer("scattering_weight", self._generate_scattering_kernels())
        self.num_cross_channels = 128
        self.register_buffer("cross_weight", self._generate_cross_kernels())
        self.dilations = self._calculate_dynamic_dilations(allowed_dilations)
        self.num_dilations = len(self.dilations)
        self.scattering_dilations = min(3, self.num_dilations)
        self.scattering_grid = 2
        self.scattering_features = (self.scattering_dilations * self.num_kernels
                                    * self.num_scattering_filters * self.scattering_grid**2)
        self.color_grid = 4
        self.cross_features = self.scattering_dilations * self.num_cross_channels * 16
        self.color_features = self.in_channels * self.color_grid**2
        self.total_features = (self.num_kernels * self.num_dilations * self.spatial_regions
                               + self.scattering_features + self.cross_features
                               + self.color_features)

        # Preserve more geometry: 24x24 responses on STL-10 instead of 16x16.
        self.stride = max(1, min(self.H, self.W) // 24)
        self.register_buffer("biases", torch.zeros(self.num_dilations, self.num_kernels, self.num_biases))

        print(f"[FilterBank] Dilations    : {self.dilations}")
        print(f"[FilterBank] Stride       : {self.stride}")
        print(f"[FilterBank] Total feats  : {self.total_features}")

    def _generate_base_kernels(self):
        """Generate 84 zero-sum 3x3 filters with fixed colour projections."""
        combinations = list(itertools.combinations(range(9), 3))
        spatial = torch.full((len(combinations), 9), -1.0)
        for index, positions in enumerate(combinations):
            spatial[index, list(positions)] = 2.0
        spatial = spatial.view(-1, 3, 3)

        if self.in_channels == 1:
            colour_vectors = torch.ones(1, 1)
        elif self.in_channels == 3:
            colour_vectors = torch.tensor([
                [0.299, 0.587, 0.114],  # luminance
                [1.0, -1.0, 0.0],      # red-green
                [0.5, 0.5, -1.0],      # yellow-blue
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ])
        else:
            colour_vectors = torch.eye(self.in_channels)

        weights = torch.empty(self.num_kernels, self.in_channels, 3, 3)
        for index in range(self.num_kernels):
            colour = colour_vectors[index % len(colour_vectors)]
            weight = colour[:, None, None] * spatial[index][None, :, :]
            weights[index] = weight / weight.square().sum().sqrt().clamp_min(1e-8)
        return weights

    def _generate_scattering_kernels(self):
        """Four fixed directional derivatives, applied depthwise."""
        kernels = torch.tensor([
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            [[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]],
            [[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]],
        ]).unsqueeze(1)
        kernels = kernels / kernels.square().sum((1, 2, 3), keepdim=True).sqrt()
        return kernels.repeat(self.num_kernels, 1, 1, 1)

    def _generate_cross_kernels(self):
        """Fixed normalized projections mixing all first-order response maps."""
        generator = torch.Generator().manual_seed(314159)
        weights = torch.randn(
            self.num_cross_channels, self.num_kernels, 1, 1, generator=generator
        )
        weights -= weights.mean(dim=1, keepdim=True)
        return weights / weights.square().sum(dim=1, keepdim=True).sqrt()

    def _calculate_dynamic_dilations(self, num_dilations):
        max_dilation = max(1, (min(self.H, self.W) - 1) // 2)
        exponent = np.linspace(0, np.log2(max_dilation), num_dilations)
        return np.unique(np.floor(2**exponent).astype(int)).tolist()

    def _convolve(self, X, dilation):
        X = F.pad(X, (dilation,) * 4, mode="reflect")
        return F.conv2d(X, self.weight, stride=self.stride, dilation=dilation)

    def fit(self, X):
        """Estimate three response quantiles without labels or gradients."""
        if self.is_fitted:
            return
        if X.ndim != 4 or X.size(1) != self.in_channels:
            raise ValueError(f"Expected input shaped (N, {self.in_channels}, H, W), got {tuple(X.shape)}")

        self.eval()
        # Evenly spaced samples bound memory and keep fitting deterministic.
        if X.size(0) > self.fit_samples:
            indices = torch.linspace(0, X.size(0) - 1, self.fit_samples).long()
            X = X.index_select(0, indices)

        quantiles = torch.tensor((0.25, 0.5, 0.75), device=X.device)
        with torch.no_grad():
            for d_idx, dilation in enumerate(self.dilations):
                responses = self._convolve(X, dilation)
                responses = responses.permute(1, 0, 2, 3).reshape(self.num_kernels, -1)
                fitted = torch.quantile(responses, quantiles, dim=1).transpose(0, 1)
                self.biases[d_idx].copy_(fitted)
        self.is_fitted = True

    def forward(self, X):
        if not self.is_fitted:
            raise RuntimeError("You must call .fit(X) before .forward(X)")

        batch = X.size(0)
        features = []
        for d_idx, dilation in enumerate(self.dilations):
            response = self._convolve(X, dilation)
            median = self.biases[d_idx, :, 1].view(1, -1, 1, 1)
            centered = response - median

            ppv = []
            for bias_idx in range(self.num_biases):
                bias = self.biases[d_idx, :, bias_idx].view(1, -1, 1, 1)
                positive = (response > bias).to(response.dtype)
                ppv.append(F.adaptive_avg_pool2d(positive, (2, 2)).flatten(1))

            positive_magnitude = F.adaptive_avg_pool2d(F.relu(centered), (2, 2)).flatten(1)
            negative_magnitude = F.adaptive_avg_pool2d(F.relu(-centered), (2, 2)).flatten(1)
            absolute_peak = F.adaptive_max_pool2d(centered.abs(), (1, 1)).flatten(1)
            features.append(torch.cat([*ppv, positive_magnitude, negative_magnitude, absolute_peak], dim=1))

            # Second-order scattering remains entirely fixed and deterministic.
            if d_idx < self.scattering_dilations:
                modulus = F.pad(centered.abs(), (1, 1, 1, 1), mode="reflect")
                second = F.conv2d(modulus, self.scattering_weight, groups=self.num_kernels)
                features.append(F.adaptive_avg_pool2d(
                    second.abs(), (self.scattering_grid, self.scattering_grid)
                ).flatten(1))
                cross = F.conv2d(centered.abs(), self.cross_weight)
                features.append(F.adaptive_avg_pool2d(
                    F.relu(cross), (4, 4)
                ).flatten(1))

        # Zero-sum filters cannot retain uniform colour, so add its coarse layout.
        features.append(F.adaptive_avg_pool2d(
            X, (self.color_grid, self.color_grid)
        ).reshape(batch, -1))
        return torch.cat(features, dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D



class model(nn.Module):
    """Fixed MiniRocket + wavelet-scattering image representation.

    ``fit`` estimates only unsupervised response quantiles in the Rocket branch.
    Kymatio's wavelets are analytic and fixed.  Consequently the complete
    extractor contains no gradient-trainable parameters.
    """

    def __init__(self, in_channels=3, input_shape=(160, 160), target_features=15000,
                 fit_samples=512, scattering_grid=4):
        super().__init__()
        if in_channels != 3:
            raise ValueError("ExtractorV3 currently expects RGB input")
        self.rocket = RocketExtractor(
            in_channels=in_channels,
            input_shape=input_shape,
            target_features=target_features,
            fit_samples=fit_samples,
        )
        self.scattering = Scattering2D(J=3, shape=input_shape, L=8, max_order=2)
        self.scattering_grid = scattering_grid
        self.register_buffer("colour", torch.tensor([
            [0.299, 0.587, 0.114],
            [1.0, -1.0, 0.0],
            [0.5, 0.5, -1.0],
        ]))
        with torch.no_grad():
            probe = torch.zeros(1, 3, *input_shape)
            scattering_features = self._scatter(probe).shape[1]
        self.total_features = self.rocket.total_features + scattering_features
        self.is_fitted = False
        print(f"[Scattering] Fixed feats : {scattering_features}")
        print(f"[Hybrid] Total feats     : {self.total_features}")

    def fit(self, x):
        self.rocket.fit(x)
        self.is_fitted = True

    def _scatter(self, x):
        # Fixed colour projection: luminance plus two opponent channels.
        projected = torch.einsum("oc,bchw->bohw", self.colour.to(x.dtype), x)
        coefficients = self.scattering(projected.contiguous())
        # Kymatio returns B x colour x paths x H x W.
        if coefficients.ndim == 4:
            coefficients = coefficients.unsqueeze(1)
        batch, colours, paths, height, width = coefficients.shape
        coefficients = coefficients.reshape(batch, colours * paths, height, width)
        # Signed log compression makes large low-frequency coefficients less
        # dominant while preserving opponent-channel sign.
        coefficients = torch.sign(coefficients) * torch.log1p(coefficients.abs())
        return F.adaptive_avg_pool2d(
            coefficients, (self.scattering_grid, self.scattering_grid)
        ).flatten(1)

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("You must call .fit(X) before .forward(X)")
        return torch.cat((self.rocket(x), self._scatter(x)), dim=1)
