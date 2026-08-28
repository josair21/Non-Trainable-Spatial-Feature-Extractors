"""Improved deterministic visual-primitives extractor (VisionRocket v5)."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, in_channels=3, input_shape=(96, 96), target_features=24000,
                 fit_samples=512):
        super().__init__()
        self.in_channels, (self.H, self.W) = in_channels, input_shape
        self.fit_samples, self.is_fitted, self.num_biases = fit_samples, False, 3
        self.register_buffer("weight", self._kernels())
        self.num_kernels = self.weight.shape[0]
        self.features_per_response = 37
        count = max(1, target_features // (self.num_kernels * 37))
        maximum = max(1, (min(input_shape) - 1) // 2)
        self.dilations = np.unique(np.floor(2 ** np.linspace(
            0, np.log2(maximum), count)).astype(int)).tolist()
        self.stride = max(1, min(input_shape) // 32)
        self.register_buffer("biases", torch.zeros(
            len(self.dilations), self.num_kernels, self.num_biases))
        self.total_features = (self.num_kernels * len(self.dilations) * 37
                               + 160 + 64 + in_channels * 32)
        print(f"[VisualPrimitivesV5] kernels={self.num_kernels} "
              f"dilations={self.dilations} stride={self.stride} "
              f"features={self.total_features}")

    def _kernels(self):
        spatial = torch.tensor([
            [[-1,0,1],[-2,0,2],[-1,0,1]], [[-1,-2,-1],[0,0,0],[1,2,1]],
            [[-2,-1,0],[-1,0,1],[0,1,2]], [[0,1,2],[-1,0,1],[-2,-1,0]],
            [[-1,2,-1],[-1,2,-1],[-1,2,-1]], [[-1,-1,-1],[2,2,2],[-1,-1,-1]],
            [[-1,-1,2],[-1,2,-1],[2,-1,-1]], [[2,-1,-1],[-1,2,-1],[-1,-1,2]],
            [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], [[0,-1,0],[-1,4,-1],[0,-1,0]],
            [[1,-2,1],[-2,4,-2],[1,-2,1]], [[2,2,0],[2,0,-1],[0,-1,-2]],
            [[0,2,2],[-1,0,2],[-2,-1,0]], [[0,-1,-2],[2,0,-1],[2,2,0]],
            [[-2,-1,0],[-1,0,2],[0,2,2]], [[0,0,0],[0,1,0],[0,0,-1]],
            [[0,0,0],[0,1,0],[-1,0,0]], [[-1,0,0],[0,1,0],[0,0,0]],
            [[0,0,-1],[0,1,0],[0,0,0]], [[1,1,1],[1,-2,1],[-1,-2,-1]],
            [[1,1,-1],[1,-2,-2],[1,1,-1]],
        ], dtype=torch.float32)
        spatial = torch.cat((spatial, -spatial))
        colours = (torch.ones(1, 1) if self.in_channels == 1 else torch.tensor([
            [0.299,0.587,0.114], [1.,-1.,0.], [.5,.5,-1.]]))
        weights = torch.cat([colour[:,None,None] * spatial[:,None]
                             for colour in colours])
        return weights / weights.square().sum((1,2,3), keepdim=True).sqrt().clamp_min(1e-8)

    def _conv(self, x, dilation):
        return F.conv2d(F.pad(x, (dilation,) * 4, mode="reflect"), self.weight,
                        stride=self.stride, dilation=dilation)

    def fit(self, x):
        if self.is_fitted:
            return
        if x.size(0) > self.fit_samples:
            ids = torch.linspace(0, x.size(0)-1, self.fit_samples,
                                 device=x.device).long()
            x = x.index_select(0, ids)
        qs = torch.tensor((.25,.5,.75), device=x.device)
        with torch.no_grad():
            for i, dilation in enumerate(self.dilations):
                response = self._conv(x, dilation).permute(1,0,2,3).reshape(self.num_kernels,-1)
                self.biases[i].copy_(torch.quantile(response, qs, dim=1).T)
        self.is_fitted = True

    def _hog(self, x):
        lum = (x[:,:1]*.299+x[:,1:2]*.587+x[:,2:3]*.114
               if x.size(1) == 3 else x.mean(1,keepdim=True))
        kernels = torch.tensor([[[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]],
                                [[[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]]],
                               device=x.device,dtype=x.dtype)
        gradient = F.conv2d(F.pad(lum,(1,1,1,1),mode="reflect"),kernels)
        gx,gy=gradient[:,:1],gradient[:,1:]
        oriented=torch.cat([F.relu(gx*math.cos(i*math.pi/4)+gy*math.sin(i*math.pi/4))
                            for i in range(8)],1)
        return torch.cat((F.adaptive_avg_pool2d(oriented,(4,4)).flatten(1),
                          F.adaptive_max_pool2d(oriented,(2,2)).flatten(1)),1)

    def _lbp(self,x):
        lum=x.mean(1,keepdim=True); pad=F.pad(lum,(1,1,1,1),mode="reflect")
        h,w=lum.shape[-2:]; code=0
        for i,(dy,dx) in enumerate(((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))):
            code=code+(pad[:,:,1+dy:1+dy+h,1+dx:1+dx+w]>lum).to(x.dtype)*(2**i)
        hist=torch.cat([(torch.remainder(code,16)==i).to(x.dtype) for i in range(16)],1)
        return F.adaptive_avg_pool2d(hist,(2,2)).flatten(1)

    def forward(self,x):
        if not self.is_fitted:
            raise RuntimeError("Call fit before forward")
        features=[]
        for i,dilation in enumerate(self.dilations):
            response=self._conv(x,dilation); median=self.biases[i,:,1].view(1,-1,1,1)
            centred=response-median
            for q in range(3):
                features.append(F.adaptive_avg_pool2d(
                    (response>self.biases[i,:,q].view(1,-1,1,1)).to(x.dtype),(2,2)).flatten(1))
            features += [F.adaptive_avg_pool2d(F.relu(centred),(2,2)).flatten(1),
                         F.adaptive_avg_pool2d(F.relu(-centred),(2,2)).flatten(1),
                         F.adaptive_avg_pool2d(centred.abs(),(4,4)).flatten(1),
                         F.adaptive_max_pool2d(centred.abs(),(1,1)).flatten(1)]
        avg=F.adaptive_avg_pool2d(x,(4,4)); second=F.adaptive_avg_pool2d(x.square(),(4,4))
        features += [avg.flatten(1),(second-avg.square()).clamp_min(0).sqrt().flatten(1),
                     self._hog(x),self._lbp(x)]
        return torch.cat(features,1)
