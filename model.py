import torch
import torch.nn as nn
import torch.nn.functional as F
from sttn import InpaintGenerator

class PReNet_STTN(nn.Module):
    def __init__(self, config):
        super(PReNet_STTN, self).__init__()
        self.config = config
        
        # Input: [B, 6, H, W] = rainy + previous output
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=1, padding=2),
            nn.ReLU()
        )
        
        # Residual blocks with dilated convolution
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=1, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                nn.ReLU()
            ) for _ in range(5)
        ])
        
        # Spatial-Temporal Transformer Network
        self.sttn = InpaintGenerator(init_weights=True)
        
        # Feature adapters
        self.sttn_adapter = nn.Sequential(
            nn.Conv2d(32, 256, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.sttn_output_adapter = nn.Sequential(
            nn.Conv2d(256, 32, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Output layer
        self.conv = nn.Conv2d(32, 3, 5, stride=1, padding=2)

    def forward(self, x, mask, prev_outputs=None):
        B, C, H, W = x.shape

        if prev_outputs and len(prev_outputs) > 0:
            prev = prev_outputs[-1]
        else:
            prev = torch.zeros_like(x)

        x_in = torch.cat([x, prev], dim=1)  # [B, 6, H, W]

        # Feature extraction
        features = self.conv0(x_in)
        for block in self.res_blocks:
            features = features + block(features)

        # STTN processing - handle single frame
        sttn_feat = self.sttn_adapter(features)              # [B, 256, H, W]
        
        # Prepare mask for STTN
        if mask.dim() == 3:  # [B, H, W]
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        
        # Use STTN infer method for single frame processing
        sttn_output = self.sttn.infer(sttn_feat, mask)       # [B, 256, H, W]
        sttn_out_feat = self.sttn_output_adapter(sttn_output)  # [B, 32, H, W]

        # Combine and decode
        features = features + sttn_out_feat
        output = torch.tanh(self.conv(features))  # Output in [-1, 1]

        return output

    def forward_sequence(self, rainy_seq, masks_seq):
        B, T, C, H, W = rainy_seq.shape
        outputs = []
        prev_outputs = []

        for t in range(T):
            x = rainy_seq[:, t]
            m = masks_seq[:, t]

            # Maintain limited temporal history
            if len(prev_outputs) >= self.config.num_frames:
                prev_outputs.pop(0)

            out = self.forward(x, m, prev_outputs)
            outputs.append(out.unsqueeze(1))
            prev_outputs.append(out)

        return torch.cat(outputs, dim=1)  # [B, T, 3, H, W]