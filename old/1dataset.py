# === dataset.py ===

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob

class VideoDerainDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.seq_length = config.seq_length
        self.img_size = config.img_size

        # Adjusted paths based on your directory structure
        if mode == 'train' or mode == 'val':
            print("ye dekh bhai : ",config.root_dir)
            rainy_dir = os.path.join(config.root_dir, 'Rainy')
            clean_dir = os.path.join(config.root_dir, 'Ground_Truth')
        else:
            raise ValueError("Unsupported mode for dataset.")

        self.rainy_frames = sorted(glob.glob(os.path.join(rainy_dir, '*.jpg')))
        self.clean_frame = os.path.join(clean_dir, 'BACKSIDE_.jpg')

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    #def __len__(self):
     #   return len(self.rainy_frames) - self.seq_length + 1

    def __len__(self):
        if len(self.rainy_frames) < self.seq_length:
            print(f"[WARNING] Not enough rainy frames: found {len(self.rainy_frames)}, required: {self.seq_length}")

        return max(len(self.rainy_frames) - self.seq_length + 1, 0)


    def __getitem__(self, idx):
        rainy_seq = []
        clean_seq = []

        for i in range(idx, idx + self.seq_length):
            rainy_img = Image.open(self.rainy_frames[i]).convert('RGB')
            rainy_seq.append(self.transform(rainy_img))

            clean_img = Image.open(self.clean_frame).convert('RGB')
            clean_seq.append(self.transform(clean_img))

        rainy_seq = torch.stack(rainy_seq)  # [T, C, H, W]
        clean_seq = torch.stack(clean_seq)

        with torch.no_grad():
            diff = torch.abs(rainy_seq - clean_seq)
            masks = (diff.mean(1, keepdim=True) > 0.1).float()

        return {
            'rainy': rainy_seq,
            'clean': clean_seq,
            'mask': masks
        }

def get_dataloaders(config):
    train_set = VideoDerainDataset(config, 'train')
    val_set = VideoDerainDataset(config, 'val')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
# import os
# import random
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import glob

# class VideoDerainDataset(Dataset):
#     def __init__(self, config, mode='train'):
#         self.config = config
#         self.mode = mode
#         self.seq_length = config.seq_length
#         self.img_size = config.img_size
        
#         # Get video pairs (rainy/clean)
#         self.rainy_videos = sorted(glob.glob(f'{config.root_dir}/{mode}/rainy/*'))
#         self.clean_videos = sorted(glob.glob(f'{config.root_dir}/{mode}/clean/*'))
        
#         # Precompute frame counts for each video
#         self.video_frame_counts = []
#         for v in self.rainy_videos:
#             frames = sorted(glob.glob(f'{v}/*.png'))
#             self.video_frame_counts.append(len(frames))
        
#         # Transformations
#         self.transform = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
    
#     def __len__(self):
#         return len(self.rainy_videos)
    
#     def __getitem__(self, idx):
#         # Get random starting frame
#         max_start = self.video_frame_counts[idx] - self.seq_length
#         start_frame = random.randint(0, max(max_start, 0))
        
#         # Load sequence of frames
#         rainy_frames = []
#         clean_frames = []
        
#         for i in range(start_frame, start_frame + self.seq_length):
#             # Load rainy frame
#             rainy_path = f'{self.rainy_videos[idx]}/frame_{i:04d}.png'
#             rainy_img = Image.open(rainy_path).convert('RGB')
#             rainy_frames.append(self.transform(rainy_img))
            
#             # Load clean frame
#             clean_path = f'{self.clean_videos[idx]}/frame_{i:04d}.png'
#             clean_img = Image.open(clean_path).convert('RGB')
#             clean_frames.append(self.transform(clean_img))
        
#         # Stack frames into tensors [T, C, H, W]
#         rainy_seq = torch.stack(rainy_frames)
#         clean_seq = torch.stack(clean_frames)
        
#         # Create binary mask (1 for rainy pixels, 0 for non-rainy)
#         # Simple thresholding for demo - in practice use more sophisticated method
#         with torch.no_grad():
#             diff = torch.abs(rainy_seq - clean_seq)
#             masks = (diff.mean(1, keepdim=True) > 0.1).float()
        
#         return {
#             'rainy': rainy_seq,
#             'clean': clean_seq,
#             'mask': masks
#         }

# def get_dataloaders(config):
#     train_set = VideoDerainDataset(config, 'train')
#     val_set = VideoDerainDataset(config, 'val')
    
#     train_loader = torch.utils.data.DataLoader(
#         train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
#     val_loader = torch.utils.data.DataLoader(
#         val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
#     return train_loader, val_loader