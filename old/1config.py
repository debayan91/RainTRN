import torch

class Config:
    # Data
    root_dir = './Datas/Aldrin_Backside' # Directory with rainy/clean video pairs
    seq_length = 10      # Number of frames to process together
    img_size = (256, 256)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recurrent_iter = 6
    num_frames = 5       # Temporal window size
    
    # Training
    batch_size = 4
    lr = 1e-4
    epochs = 100
    save_dir = './checkpoints'
    save_every = 5
    
    # Loss weights
    l1_weight = 1.0
    perceptual_weight = 0.1
    temporal_weight = 0.5
    
config = Config()