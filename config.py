import torch

class Config:
    # Data
    root_dir = '/media/neurocomputinglab/DATA/vitdilli/RainTransformer-20250604T115928Z-1-001/RainTransformer/data'
    seq_length = 10         # Number of frames to process together
    img_size = (1744, 981)  # Original resolution
    crop_size = (512, 512)  # Reduced crop size for better memory usage
    img_format = 'jpg'
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recurrent_iter = 6
    num_frames = 3       # Reduced temporal window size for memory
    patchsize = [(64, 64), (32, 32), (16, 16), (8, 8)]  # Adjusted for crop size
    
    # Training
    batch_size = 1       # Reduced to 1 due to high resolution and memory constraints
    lr = 1e-4
    epochs = 100
    save_dir = './checkpoints'
    save_every = 5
    
    # Loss weights
    l1_weight = 1.0
    perceptual_weight = 0.1
    temporal_weight = 0.5
    
config = Config()