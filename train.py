
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloaders
from model import PReNet_STTN
from config import config
from utils import save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_msssim import SSIM  # pip install pytorch-msssim

class DerainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        # SSIM with proper data range for [-1,1] normalized images
        self.ssim = SSIM(data_range=2.0, channel=3, size_average=True)
        self.alpha = 0.84
        
    def forward(self, pred, clean, mask):
        # Handle 5D tensors (B, T, C, H, W)
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred_flat = pred.view(B*T, C, H, W)
            clean_flat = clean.view(B*T, C, H, W)
        else:
            pred_flat = pred
            clean_flat = clean
        
        # Calculate SSIM (returns value between 0 and 1, where 1 is perfect)
        ssim_val = self.ssim(pred_flat, clean_flat)
        ssim_loss = 1.0 - ssim_val  # convert to loss where lower is better
        
        # Calculate MSE
        mse_loss = self.mse_loss(pred_flat, clean_flat)
        
        # Weighted combination (SSIM is already in [0,1] range)
        combined_loss = self.alpha * ssim_loss + (1 - self.alpha) * mse_loss
        
        # Temporal consistency loss (using MSE for stability)
        temp_loss = 0
        if pred.dim() == 5 and pred.shape[1] > 1:
            for t in range(1, pred.shape[1]):
                temp_loss += self.mse_loss(pred[:, t] - pred[:, t-1], 
                                        clean[:, t] - clean[:, t-1])
            temp_loss /= (pred.shape[1] - 1)
        
        total_loss = combined_loss + config.temporal_weight * temp_loss
        
        return total_loss, {
            'ssim': ssim_loss,
            'mse': mse_loss,
            'temp': temp_loss,
            'ssim_value': ssim_val  # Track actual SSIM value for monitoring
        }

def train():
    os.makedirs(config.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config.save_dir, 'logs'))

    model = PReNet_STTN(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = DerainLoss()

    start_epoch = 0
    if os.path.exists(os.path.join(config.save_dir, 'latest.pth')):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, config.save_dir)
    
    train_loader, val_loader = get_dataloaders(config)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = epoch_ssim = epoch_mse = epoch_temp = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            rainy = batch['rainy'].to(config.device)
            clean = batch['clean'].to(config.device)
            masks = batch['mask'].to(config.device)

            try:
                outputs = model.forward_sequence(rainy, masks)
                loss, loss_dict = criterion(outputs, clean, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                epoch_ssim += loss_dict['ssim'].item()
                epoch_mse += loss_dict['mse'].item()
                epoch_temp += loss_dict['temp'].item() if isinstance(loss_dict['temp'], torch.Tensor) else loss_dict['temp']

                if batch_idx % 5 == 0:
                    print(f'Epoch {epoch:3d} Batch {batch_idx:3d} | '
                          f'Loss: {loss.item():.4f} | '
                          f'SSIM: {1 - loss_dict["ssim"].item():.4f} | '  # Show actual SSIM
                          f'MSE: {loss_dict["mse"].item():.4f} | '
                          f'Temp: {loss_dict["temp"]:.4f}')
                    
                    writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('train/ssim', 1 - loss_dict['ssim'].item(), epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('train/mse', loss_dict['mse'].item(), epoch * len(train_loader) + batch_idx)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise e

        # Validation and logging
        val_loss = validate(model, val_loader, criterion, epoch, writer) if val_loader else 0
        scheduler.step(val_loss)

        # Save checkpoint
        if epoch % config.save_every == 0 or epoch == config.epochs - 1:
            save_checkpoint(model, optimizer, epoch, config.save_dir)

        # Log epoch metrics
        avg_ssim = 1 - (epoch_ssim / len(train_loader))  # Convert back to SSIM value
        writer.add_scalar('metrics/train_ssim', avg_ssim, epoch)
        writer.add_scalar('metrics/val_loss', val_loss, epoch)
        writer.add_scalar('params/lr', optimizer.param_groups[0]['lr'], epoch)

    writer.close()
    print("Training completed!")

# ... (keep the validate() function from previous version)

# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from dataset import get_dataloaders
# from model import PReNet_STTN
# from config import config
# from utils import save_checkpoint, load_checkpoint
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from pytorch_msssim import SSIM  # You'll need to install this package

# class DerainLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse_loss = nn.MSELoss()
#         self.ssim_loss = SSIM(data_range=2.0, channel=3)  # Data range is 2 for [-1,1] to [1,1]
#         self.alpha = 0.84
        
#     def forward(self, pred, clean, mask):
#         # Handle 5D tensors (B, T, C, H, W)
#         if pred.dim() == 5:
#             B, T, C, H, W = pred.shape
#             pred_flat = pred.view(B*T, C, H, W)
#             clean_flat = clean.view(B*T, C, H, W)
#             mask_flat = mask.view(B*T, 1, H, W) if mask.dim() == 5 else mask.view(B*T, mask.shape[-3], H, W)
#         else:
#             pred_flat = pred
#             clean_flat = clean
#             mask_flat = mask
        
#         # Calculate SSIM (higher is better, so we use 1-SSIM)
#         ssim_val = self.ssim_loss(pred_flat, clean_flat)
#         ssim_loss = 1 - ssim_val
        
#         # Calculate MSE
#         mse_loss = self.mse_loss(pred_flat, clean_flat)
        
#         # Weighted combination
#         combined_loss = self.alpha * ssim_loss + (1 - self.alpha) * mse_loss
        
#         # Temporal consistency loss (only for sequences)
#         temp_loss = 0
#         if pred.dim() == 5 and pred.shape[1] > 1:
#             for t in range(1, pred.shape[1]):
#                 temp_loss += self.mse_loss(pred[:, t] - pred[:, t-1], 
#                                          clean[:, t] - clean[:, t-1])
#             temp_loss /= (pred.shape[1] - 1)
        
#         total_loss = combined_loss + config.temporal_weight * temp_loss
        
#         return total_loss, {'ssim': ssim_loss, 'mse': mse_loss, 'temp': temp_loss}

# def train():
#     os.makedirs(config.save_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=os.path.join(config.save_dir, 'logs'))

#     model = PReNet_STTN(config).to(config.device)
#     optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#     criterion = DerainLoss()

#     start_epoch = 0
#     checkpoint_path = os.path.join(config.save_dir, 'latest.pth')
#     if os.path.exists(checkpoint_path):
#         print(f"Loading checkpoint from {checkpoint_path}")
#         model, optimizer, start_epoch = load_checkpoint(model, optimizer, config.save_dir)
#         print(f"Resumed from epoch {start_epoch}")
    
#     train_loader, val_loader = get_dataloaders(config)
#     print(f"Training samples: {len(train_loader)}")

#     for epoch in range(start_epoch, config.epochs):
#         model.train()
#         epoch_loss = 0
#         epoch_ssim = 0
#         epoch_mse = 0
#         epoch_temp = 0
#         start_time = time.time()
        
#         for batch_idx, batch in enumerate(train_loader):
#             rainy = batch['rainy'].to(config.device)  # [B, T, C, H, W]
#             clean = batch['clean'].to(config.device)  # [B, T, C, H, W]
#             masks = batch['mask'].to(config.device)   # [B, T, 1, H, W]

#             optimizer.zero_grad()
            
#             try:
#                 # Forward pass through sequence
#                 outputs = model.forward_sequence(rainy, masks)  # [B, T, C, H, W]

#                 loss, loss_dict = criterion(outputs, clean, masks)
#                 loss.backward()
                
#                 # Gradient clipping to prevent exploding gradients
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
#                 optimizer.step()

#                 epoch_loss += loss.item()
#                 epoch_ssim += loss_dict['ssim'].item()
#                 epoch_mse += loss_dict['mse'].item()
#                 epoch_temp += loss_dict['temp'].item() if isinstance(loss_dict['temp'], torch.Tensor) else loss_dict['temp']

#                 if batch_idx % 5 == 0:  # Print every 5 batches
#                     print(f'Epoch {epoch:3d} Batch {batch_idx:3d} | '
#                           f'Loss: {loss.item():.4f} | '
#                           f'SSIM: {loss_dict["ssim"].item():.4f} | '
#                           f'MSE: {loss_dict["mse"].item():.4f} | '
#                           f'Temp: {loss_dict["temp"].item() if isinstance(loss_dict["temp"], torch.Tensor) else loss_dict["temp"]:.4f}')
                    
#                     # Log to tensorboard
#                     global_step = epoch * len(train_loader) + batch_idx
#                     writer.add_scalar('train/loss', loss.item(), global_step)
#                     writer.add_scalar('train/ssim_loss', loss_dict['ssim'].item(), global_step)
#                     writer.add_scalar('train/mse_loss', loss_dict['mse'].item(), global_step)
#                     if isinstance(loss_dict['temp'], torch.Tensor):
#                         writer.add_scalar('train/temporal_loss', loss_dict['temp'].item(), global_step)
                    
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print(f"WARNING: Out of memory at batch {batch_idx}. Skipping batch.")
#                     if hasattr(torch.cuda, 'empty_cache'):
#                         torch.cuda.empty_cache()
#                     continue
#                 else:
#                     raise e

#         # Validation
#         val_loss = 0
#         if val_loader:
#             val_loss = validate(model, val_loader, criterion, epoch, writer)
#             scheduler.step(val_loss)  # Update learning rate based on validation loss

#         # Save checkpoint
#         if epoch % config.save_every == 0 or epoch == config.epochs - 1:
#             save_checkpoint(model, optimizer, epoch, config.save_dir)
#             print(f"Checkpoint saved at epoch {epoch}")

#         # Calculate average losses
#         avg_loss = epoch_loss / len(train_loader)
#         avg_ssim = epoch_ssim / len(train_loader)
#         avg_mse = epoch_mse / len(train_loader)
#         avg_temp = epoch_temp / len(train_loader)
        
#         epoch_time = time.time() - start_time
        
#         print(f'\n=== Epoch {epoch:3d} Summary ===')
#         print(f'Time: {epoch_time:.2f}s')
#         print(f'Train Loss: {avg_loss:.4f} (SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.4f}, Temp: {avg_temp:.4f})')
#         if val_loader:
#             print(f'Val Loss: {val_loss:.4f}')
#         print('=' * 30)

#         # Log epoch averages
#         writer.add_scalar('epoch/train_loss', avg_loss, epoch)
#         writer.add_scalar('epoch/ssim_loss', avg_ssim, epoch)
#         writer.add_scalar('epoch/mse_loss', avg_mse, epoch)
#         writer.add_scalar('epoch/temporal_loss', avg_temp, epoch)
#         writer.add_scalar('epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)
#         if val_loader:
#             writer.add_scalar('epoch/val_loss', val_loss, epoch)

#     writer.close()
#     print("Training completed!")

def validate(model, val_loader, criterion, epoch, writer):
    if not val_loader:
        return 0

    model.eval()
    val_loss = 0
    val_ssim = 0
    val_mse = 0
    val_temp = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            rainy = batch['rainy'].to(config.device)
            clean = batch['clean'].to(config.device)
            masks = batch['mask'].to(config.device)

            try:
                outputs = model.forward_sequence(rainy, masks)
                loss, loss_dict = criterion(outputs, clean, masks)
                
                val_loss += loss.item()
                val_ssim += loss_dict['ssim'].item()
                val_mse += loss_dict['mse'].item()
                val_temp += loss_dict['temp'].item() if isinstance(loss_dict['temp'], torch.Tensor) else loss_dict['temp']

                # Log sample images every 10 epochs
                if batch_idx == 0 and epoch % 10 == 0:
                    # Take the first frame from the sequence for visualization
                    sample_rainy = rainy[0, 0]  # [C, H, W]
                    sample_clean = clean[0, 0]  # [C, H, W]
                    sample_output = outputs[0, 0]  # [C, H, W]
                    
                    # Concatenate horizontally: rainy | clean | output
                    comparison = torch.cat([
                        sample_rainy.unsqueeze(0),
                        sample_clean.unsqueeze(0),
                        sample_output.unsqueeze(0)
                    ], dim=0)  # [3, C, H, W]
                    
                    # Denormalize for visualization (from [-1,1] to [0,1])
                    comparison = (comparison + 1) / 2
                    comparison = torch.clamp(comparison, 0, 1)
                    
                    writer.add_images('val/comparison', comparison, epoch)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: Out of memory during validation at batch {batch_idx}. Skipping.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    avg_val_loss = val_loss / len(val_loader)
    avg_val_ssim = val_ssim / len(val_loader)
    avg_val_mse = val_mse / len(val_loader)
    avg_val_temp = val_temp / len(val_loader)
    
    # Log validation metrics
    writer.add_scalar('val/loss', avg_val_loss, epoch)
    writer.add_scalar('val/ssim', 1 - avg_val_ssim, epoch)
    writer.add_scalar('val/mse_loss', avg_val_mse, epoch)
    writer.add_scalar('val/temporal_loss', avg_val_temp, epoch)
    
    return avg_val_loss

if __name__ == '__main__':
    print("Starting training...")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.seq_length}")
    print(f"Crop size: {config.crop_size}")
    print(f"Save directory: {config.save_dir}")
    
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
























# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from dataset import get_dataloaders
# from model import PReNet_STTN
# from config import config
# from utils import save_checkpoint, load_checkpoint

# class DerainLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1_loss = nn.L1Loss()
        
#     def forward(self, pred, clean, mask):
#         # Handle 5D tensors (B, T, C, H, W)
#         if pred.dim() == 5:
#             B, T, C, H, W = pred.shape
#             pred_flat = pred.view(B*T, C, H, W)
#             clean_flat = clean.view(B*T, C, H, W)
#             mask_flat = mask.view(B*T, 1, H, W) if mask.dim() == 5 else mask.view(B*T, mask.shape[-3], H, W)
#         else:
#             pred_flat = pred
#             clean_flat = clean
#             mask_flat = mask
        
#         # L1 loss on masked regions (rain areas)
#         l1 = self.l1_loss(pred_flat * mask_flat, clean_flat * mask_flat)
        
#         # Global L1 loss
#         global_l1 = self.l1_loss(pred_flat, clean_flat)
        
#         # Temporal consistency loss (only for sequences)
#         temp_loss = 0
#         if pred.dim() == 5 and pred.shape[1] > 1:
#             for t in range(1, pred.shape[1]):
#                 temp_loss += self.l1_loss(pred[:, t] - pred[:, t-1], 
#                                           clean[:, t] - clean[:, t-1])
#             temp_loss /= (pred.shape[1] - 1)
        
#         total_loss = (config.l1_weight * l1 + 
#                       0.5 * global_l1 + 
#                       config.temporal_weight * temp_loss)
        
#         return total_loss, {'l1': l1, 'global': global_l1, 'temp': temp_loss}

# def train():
#     os.makedirs(config.save_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=os.path.join(config.save_dir, 'logs'))

#     model = PReNet_STTN(config).to(config.device)
#     optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
#     criterion = DerainLoss()

#     start_epoch = 0
#     checkpoint_path = os.path.join(config.save_dir, 'latest.pth')
#     if os.path.exists(checkpoint_path):
#         print(f"Loading checkpoint from {checkpoint_path}")
#         model, optimizer, start_epoch = load_checkpoint(model, optimizer, config.save_dir)
#         print(f"Resumed from epoch {start_epoch}")
    
#     train_loader, val_loader = get_dataloaders(config)
#     print(f"Training samples: {len(train_loader)}")

#     for epoch in range(start_epoch, config.epochs):
#         model.train()
#         epoch_loss = 0
#         epoch_l1 = 0
#         epoch_global = 0
#         epoch_temp = 0
#         start_time = time.time()
        
#         for batch_idx, batch in enumerate(train_loader):
#             rainy = batch['rainy'].to(config.device)  # [B, T, C, H, W]
#             clean = batch['clean'].to(config.device)  # [B, T, C, H, W]
#             masks = batch['mask'].to(config.device)   # [B, T, 1, H, W]

#             optimizer.zero_grad()
            
#             try:
#                 # Forward pass through sequence
#                 outputs = model.forward_sequence(rainy, masks)  # [B, T, C, H, W]

#                 loss, loss_dict = criterion(outputs, clean, masks)
#                 loss.backward()
                
#                 # Gradient clipping to prevent exploding gradients
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
#                 optimizer.step()

#                 epoch_loss += loss.item()
#                 epoch_l1 += loss_dict['l1'].item()
#                 epoch_global += loss_dict['global'].item()
#                 epoch_temp += loss_dict['temp'].item() if isinstance(loss_dict['temp'], torch.Tensor) else loss_dict['temp']

#                 if batch_idx % 5 == 0:  # Print every 5 batches
#                     print(f'Epoch {epoch:3d} Batch {batch_idx:3d} | '
#                           f'Loss: {loss.item():.4f} | '
#                           f'L1: {loss_dict["l1"].item():.4f} | '
#                           f'Global: {loss_dict["global"].item():.4f} | '
#                           f'Temp: {loss_dict["temp"].item() if isinstance(loss_dict["temp"], torch.Tensor) else loss_dict["temp"]:.4f}')
                    
#                     # Log to tensorboard
#                     global_step = epoch * len(train_loader) + batch_idx
#                     writer.add_scalar('train/loss', loss.item(), global_step)
#                     writer.add_scalar('train/l1_loss', loss_dict['l1'].item(), global_step)
#                     writer.add_scalar('train/global_loss', loss_dict['global'].item(), global_step)
#                     if isinstance(loss_dict['temp'], torch.Tensor):
#                         writer.add_scalar('train/temporal_loss', loss_dict['temp'].item(), global_step)
                    
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print(f"WARNING: Out of memory at batch {batch_idx}. Skipping batch.")
#                     if hasattr(torch.cuda, 'empty_cache'):
#                         torch.cuda.empty_cache()
#                     continue
#                 else:
#                     raise e

#         # Validation
#         val_loss = 0
#         if val_loader:
#             val_loss = validate(model, val_loader, criterion, epoch, writer)

#         # Save checkpoint
#         if epoch % config.save_every == 0 or epoch == config.epochs - 1:
#             save_checkpoint(model, optimizer, epoch, config.save_dir)
#             print(f"Checkpoint saved at epoch {epoch}")

#         # Calculate average losses
#         avg_loss = epoch_loss / len(train_loader)
#         avg_l1 = epoch_l1 / len(train_loader)
#         avg_global = epoch_global / len(train_loader)
#         avg_temp = epoch_temp / len(train_loader)
        
#         epoch_time = time.time() - start_time
        
#         print(f'\n=== Epoch {epoch:3d} Summary ===')
#         print(f'Time: {epoch_time:.2f}s')
#         print(f'Train Loss: {avg_loss:.4f} (L1: {avg_l1:.4f}, Global: {avg_global:.4f}, Temp: {avg_temp:.4f})')
#         if val_loader:
#             print(f'Val Loss: {val_loss:.4f}')
#         print('=' * 30)

#         # Log epoch averages
#         writer.add_scalar('epoch/train_loss', avg_loss, epoch)
#         writer.add_scalar('epoch/l1_loss', avg_l1, epoch)
#         writer.add_scalar('epoch/global_loss', avg_global, epoch)
#         writer.add_scalar('epoch/temporal_loss', avg_temp, epoch)
#         writer.add_scalar('epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)
#         if val_loader:
#             writer.add_scalar('epoch/val_loss', val_loss, epoch)

#     writer.close()
#     print("Training completed!")

# def validate(model, val_loader, criterion, epoch, writer):
#     if not val_loader:
#         return 0

#     model.eval()
#     val_loss = 0
#     val_l1 = 0
#     val_global = 0
#     val_temp = 0

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(val_loader):
#             rainy = batch['rainy'].to(config.device)
#             clean = batch['clean'].to(config.device)
#             masks = batch['mask'].to(config.device)

#             try:
#                 outputs = model.forward_sequence(rainy, masks)
#                 loss, loss_dict = criterion(outputs, clean, masks)
                
#                 val_loss += loss.item()
#                 val_l1 += loss_dict['l1'].item()
#                 val_global += loss_dict['global'].item()
#                 val_temp += loss_dict['temp'].item() if isinstance(loss_dict['temp'], torch.Tensor) else loss_dict['temp']

#                 # Log sample images every 10 epochs
#                 if batch_idx == 0 and epoch % 10 == 0:
#                     # Take the first frame from the sequence for visualization
#                     sample_rainy = rainy[0, 0]  # [C, H, W]
#                     sample_clean = clean[0, 0]  # [C, H, W]
#                     sample_output = outputs[0, 0]  # [C, H, W]
                    
#                     # Concatenate horizontally: rainy | clean | output
#                     comparison = torch.cat([
#                         sample_rainy.unsqueeze(0),
#                         sample_clean.unsqueeze(0),
#                         sample_output.unsqueeze(0)
#                     ], dim=0)  # [3, C, H, W]
                    
#                     # Denormalize for visualization (from [-1,1] to [0,1])
#                     comparison = (comparison + 1) / 2
#                     comparison = torch.clamp(comparison, 0, 1)
                    
#                     writer.add_images('val/comparison', comparison, epoch)
                    
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print(f"WARNING: Out of memory during validation at batch {batch_idx}. Skipping.")
#                     if hasattr(torch.cuda, 'empty_cache'):
#                         torch.cuda.empty_cache()
#                     continue
#                 else:
#                     raise e

#     avg_val_loss = val_loss / len(val_loader)
#     avg_val_l1 = val_l1 / len(val_loader)
#     avg_val_global = val_global / len(val_loader)
#     avg_val_temp = val_temp / len(val_loader)
    
#     # Log validation metrics
#     writer.add_scalar('val/loss', avg_val_loss, epoch)
#     writer.add_scalar('val/l1_loss', avg_val_l1, epoch)
#     writer.add_scalar('val/global_loss', avg_val_global, epoch)
#     writer.add_scalar('val/temporal_loss', avg_val_temp, epoch)
    
#     return avg_val_loss

# if __name__ == '__main__':
#     print("Starting training...")
#     print(f"Device: {config.device}")
#     print(f"Batch size: {config.batch_size}")
#     print(f"Sequence length: {config.seq_length}")
#     print(f"Crop size: {config.crop_size}")
#     print(f"Save directory: {config.save_dir}")
    
#     try:
#         train()
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user")
#     except Exception as e:
#         print(f"\nTraining failed with error: {e}")
#         raise