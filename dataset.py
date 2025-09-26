import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import re


class FlatVideoDerainDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.seq_length = config.seq_length
        self.img_size = config.img_size
        self.crop_size = config.crop_size

        # Define paths
        base_dir = os.path.join(config.root_dir, 'Aldrin_Backside')
        self.rainy_dir = os.path.join(base_dir, 'Rainy')
        self.clean_img_path = os.path.join(base_dir, 'Ground_Truth', 'BACKSIDE_.jpg')

        # Get frame paths and sort them numerically
        frame_pattern = os.path.join(self.rainy_dir, f'BACKSIDE_*.{config.img_format}')
        frame_files = glob.glob(frame_pattern)
        
        # Sort numerically by extracting frame numbers
        def extract_frame_number(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'BACKSIDE_(\d+)\.', filename)
            return int(match.group(1)) if match else 0
        
        self.frames = sorted(frame_files, key=extract_frame_number)
        
        print(f"Found {len(self.frames)} frames in {self.rainy_dir}")
        if len(self.frames) == 0:
            raise ValueError(f"No frames found matching pattern: {frame_pattern}")

        if len(self.frames) < self.seq_length:
            raise ValueError(f"Not enough frames ({len(self.frames)}) for sequence length {self.seq_length}.")

        # Transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        # Load ground truth image once
        if not os.path.exists(self.clean_img_path):
            raise FileNotFoundError(f"Ground truth image not found: {self.clean_img_path}")

        self.clean_img_full = Image.open(self.clean_img_path).convert('RGB')
        print(f"Loaded ground truth image: {self.clean_img_path}")
        print(f"Ground truth size: {self.clean_img_full.size}")

    def __len__(self):
        return max(1, len(self.frames) - self.seq_length + 1)  # Sliding window over frames

    def __getitem__(self, idx):
        rainy_frames = []
        clean_frames = []

        # Ensure we don't go out of bounds
        start_idx = min(idx, len(self.frames) - self.seq_length)
        
        for i in range(start_idx, start_idx + self.seq_length):
            rainy_path = self.frames[i]
            rainy_img = Image.open(rainy_path).convert('RGB')

            if self.mode == 'train':
                w, h = rainy_img.size
                if w < self.crop_size[0] or h < self.crop_size[1]:
                    # Resize if image is smaller than crop size
                    rainy_img = rainy_img.resize((max(w, self.crop_size[0]), max(h, self.crop_size[1])))
                    w, h = rainy_img.size
                
                x = random.randint(0, max(0, w - self.crop_size[0]))
                y = random.randint(0, max(0, h - self.crop_size[1]))
                rainy_img = rainy_img.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))
                
                # Apply same crop to clean image
                clean_img_resized = self.clean_img_full.resize((w, h))
                clean_img = clean_img_resized.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))
            else:
                # For validation, resize to crop size
                rainy_img = rainy_img.resize(self.crop_size)
                clean_img = self.clean_img_full.resize(self.crop_size)

            rainy_frames.append(self.transform(rainy_img))
            clean_frames.append(self.transform(clean_img))

        rainy_seq = torch.stack(rainy_frames)
        clean_seq = torch.stack(clean_frames)

        # Create binary rain mask based on difference
        with torch.no_grad():
            diff = torch.abs(rainy_seq - clean_seq)
            masks = (diff.mean(1, keepdim=True) > 0.1).float()

        return {
            'rainy': rainy_seq,     # [T, C, H, W]
            'clean': clean_seq,     # [T, C, H, W]
            'mask': masks           # [T, 1, H, W]
        }


def get_dataloaders(config):
    train_set = FlatVideoDerainDataset(config, mode='train')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False)

    # No validation set for now
    val_loader = None
    print("Validation set skipped - no separate validation data")

    return train_loader, val_loader