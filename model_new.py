import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.ops import SqueezeExcitation

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

# Enhanced Laplacian Module for Localization Branch
class EnhancedLaplacian(nn.Module):
    """Learnable multi-scale feature extractor with SE blocks"""
    def __init__(self, scales=[1,2,4]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=2**i, dilation=2**i),
                nn.ReLU(),
                SqueezeExcitation(32, squeeze_channels=4),
                nn.Conv2d(32, 16, 1)
            ) for i in range(len(scales))
        ])
        
    def forward(self, x):
        features = []
        for conv in self.convs:
            feat = conv(x)
            features.append(F.adaptive_avg_pool2d(feat, x.shape[-2:]))
        return torch.cat(features, dim=1)

# Spatial Attention Gate for Localization Branch
class SpatialAttentionGate(nn.Module):
    """Cross-scale attention mechanism"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.query(x).view(B, -1, H*W)
        K = self.key(x).view(B, -1, H*W)
        energy = torch.bmm(Q.permute(0,2,1), K)
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(x.view(B,C,H*W), attention).view(B,C,H,W)
        return self.gamma*out + x

# Transformer Block for Classification Branch
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.pool = nn.AdaptiveAvgPool2d((14, 14))
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_small = self.pool(x)
        x_flat = x_small.flatten(2).permute(2, 0, 1)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm(x_flat))
        _, _, small_h, small_w = x_small.shape
        x_small = x_flat.permute(1, 2, 0).view(b, c, small_h, small_w)
        x = F.interpolate(x_small, size=(h, w), mode='bilinear', align_corners=False)
        return x

# Dynamic Convolution Module
class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel3 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.kernel5 = nn.Conv2d(in_channels, out_channels, 5, stride=stride, padding=2)
        self.attention = nn.Parameter(torch.ones(2) * 0.5)
        
    def forward(self, x):
        w = F.softmax(self.attention, 0)
        return w[0]*self.kernel3(x) + w[1]*self.kernel5(x)

# Frequency Analysis Branch (FAHA) for Classification
class FrequencyAnalysisBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Convolution for frequency domain features (real + imaginary parts)
        self.conv_freq = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1)
        self.bn_freq = nn.BatchNorm2d(in_channels * 2)
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        # Transform to frequency domain using 2D FFT
        x_fft = torch.fft.fft2(x)
        x_real = x_fft.real
        x_imag = x_fft.imag
        x_freq = torch.cat([x_real, x_imag], dim=1)  # Concatenate real and imaginary parts

        # Process frequency features
        x_freq = F.relu(self.bn_freq(self.conv_freq(x_freq)))

        # Split back into real and imaginary parts
        x_real = x_freq[:, :x_freq.size(1)//2, :, :]
        x_imag = x_freq[:, x_freq.size(1)//2:, :, :]

        # Combine into complex tensor and transform back to spatial domain
        x_complex = torch.complex(x_real, x_imag)
        x_spatial = torch.fft.ifft2(x_complex).real

        # Apply channel and spatial attention
        x_spatial = x_spatial * self.ca(x_spatial)
        x_spatial = x_spatial * self.sa(x_spatial)

        return x_spatial

# Multi-scale Gradient Consistency Module (MGCM) for Classification
class MultiScaleGradientConsistency(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Use grouped convolution to apply Sobel filter to each channel
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        
        # Initialize Sobel filters
        sobel_x_kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_y_kernel = torch.tensor([[[-1,-2,-1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.sobel_x.weight.data = sobel_x_kernel.repeat(in_channels, 1, 1, 1)
        self.sobel_y.weight.data = sobel_y_kernel.repeat(in_channels, 1, 1, 1)
        
        # Freeze Sobel filters
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

        self.conv_grad = DynamicConv(in_channels * 3, in_channels)
        self.bn_grad = nn.BatchNorm2d(in_channels)
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def compute_gradients(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad_mag

    def forward(self, x):
        # Original scale gradients
        grad_original = self.compute_gradients(x)

        # 0.5x scale gradients
        x_down_05 = F.avg_pool2d(x, kernel_size=2, stride=2)
        grad_05 = self.compute_gradients(x_down_05)
        grad_05 = F.interpolate(grad_05, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 0.25x scale gradients
        x_down_025 = F.avg_pool2d(x_down_05, kernel_size=2, stride=2)
        grad_025 = self.compute_gradients(x_down_025)
        grad_025 = F.interpolate(grad_025, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate gradients from all scales
        grad_combined = torch.cat([grad_original, grad_05, grad_025], dim=1)

        # Process combined gradients
        grad_features = F.relu(self.bn_grad(self.conv_grad(grad_combined)))
        grad_features = grad_features * self.ca(grad_features)
        grad_features = grad_features * self.sa(grad_features)

        return grad_features

# Combined Model for Classification and Localization
class EfficientLaFNetDual(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Common Backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Feature channels from different blocks
        self.skip_connections = [112, 160, 272, 448]
        
        # Classification Branch Components
        # FAHA module
        self.freq_branch = FrequencyAnalysisBranch(448)
        
        # MGCM module
        self.grad_branch = MultiScaleGradientConsistency(448)
        
        # Feature Combination
        self.channel_att = ChannelAttention(448*3)
        self.spatial_att = SpatialAttention()
        
        # Dual Attention Fusion
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(448*3, num_heads=4),
            TransformerBlock(448*3, num_heads=4)
        )
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classification_head = nn.Linear(448*3, num_classes)
        
        # Localization Branch Components
        # Channel reduction for Laplacian input
        self.channel_reduction = nn.Conv2d(448, 3, 1)
        
        # Enhanced Laplacian module
        self.laplacian = EnhancedLaplacian()
        
        # Fusion module
        self.loc_fusion = nn.Sequential(
            nn.Conv2d(448+48, 256, 3, padding=1),
            SpatialAttentionGate(256),
            nn.ReLU()
        )
        
        # Localization decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2),  # 64x64 -> 128x128
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            SpatialAttentionGate(64),
            nn.Upsample(scale_factor=2),  # 128x128 -> 256x256
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Upsample(scale_factor=2),  # 256x256 -> 512x512
            nn.Conv2d(32, 1, 1)
        )

    def extract_backbone_features(self, x):
        # Extract features using EfficientNet backbone
        endpoints = []
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        for idx, block in enumerate(self.backbone._blocks):
            x = block(x)
            if idx in [10, 18, 27, 38]:
                endpoints.append(x)
        return x, endpoints
    
    def forward_classification(self, features):
        # Process features through FAHA branch
        freq_features = self.freq_branch(features)
        
        # Process features through MGCM branch
        grad_features = self.grad_branch(features)
        
        # Combine features from backbone, FAHA, and MGCM
        combined = torch.cat([features, freq_features, grad_features], dim=1)
        
        # Apply dual attention
        channel_attention = self.channel_att(combined)
        combined = combined * channel_attention
        spatial_attention = self.spatial_att(combined)
        fused = combined * spatial_attention
        
        # Apply transformer blocks
        transformed = self.transformer_blocks(fused)
        
        # Classification
        pooled = self.avgpool(transformed)
        flattened = pooled.view(pooled.size(0), -1)
        output = self.classification_head(flattened)
        
        return output
    
    def forward_localization(self, features):
        # Reduce channels for Laplacian input
        x_for_lap = self.channel_reduction(features)
        
        # Process through Laplacian module
        lap_feat = self.laplacian(x_for_lap)
        
        # Fusion
        fused = self.loc_fusion(torch.cat([features, lap_feat], dim=1))
        
        # Decode to produce segmentation mask
        mask = self.decoder(fused)
        
        return mask
    
    def forward(self, x, mode='both'):
        # Extract backbone features
        features, _ = self.extract_backbone_features(x)
        
        if mode == 'classification':
            return self.forward_classification(features)
        elif mode == 'localization':
            return self.forward_localization(features)
        else:  # 'both'
            class_output = self.forward_classification(features)
            loc_output = self.forward_localization(features)
            return class_output, loc_output 