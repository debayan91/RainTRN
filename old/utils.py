import os
import torch
import numpy as np
from PIL import Image

def save_checkpoint(model, optimizer, epoch, save_dir):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, f'{save_dir}/epoch_{epoch}.pth')
    torch.save(state, f'{save_dir}/latest.pth')

def load_checkpoint(model, optimizer, save_dir):
    checkpoint = torch.load(f'{save_dir}/latest.pth')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def save_images(outputs, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for out, name in zip(outputs, filenames):
        out = out.squeeze().cpu().numpy()
        out = (out * 255).astype(np.uint8)
        Image.fromarray(out).save(f'{output_dir}/{name}.png')