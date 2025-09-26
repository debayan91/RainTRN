import torch
import os

def save_checkpoint(model, optimizer, epoch, save_dir):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, f'{save_dir}/latest.pth')

def load_checkpoint(model, optimizer, save_dir):
    checkpoint = torch.load(f'{save_dir}/latest.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch