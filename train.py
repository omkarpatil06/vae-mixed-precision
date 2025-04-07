import time
import torch
import configuration
import numpy as np
import torch.nn.functional as F

from torch.amp import autocast, GradScaler 
from contextlib import nullcontext

from tqdm import tqdm
from utilities import get_memory_usage

def train_epoch(model, train_loader, optimizer, epoch, loss_function, use_mixed_precision=False):
    model.train()

    train_loss = 0
    train_recon_loss = 0
    train_reg_loss = 0  # KL divergence for VAE, VQ loss for VQ-VAE
    
    # for mixed precision training
    scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    
    # for timing
    batch_times = []
    memory_usage = []
    
    # wrap with tqdm for progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (data, _) in enumerate(pbar):
        batch_start = time.time()
        
        data = data.to(configuration.config['device'])
        optimizer.zero_grad()
        
        # use autocast for mixed precision only if CUDA is available
        with autocast(device_type='cuda', enabled=False) if use_mixed_precision and torch.cuda.is_available() else nullcontext():

            # forward pass based on model type
            if configuration.config['model_type'].lower() == 'vae':
                recon_batch, mu, logvar = model(data)
                loss, recon_loss, kld = loss_function(recon_batch, data, mu, logvar, configuration.config['beta'])
                reg_loss = kld

            else:  # VQ-VAE
                recon_batch, vq_loss, _ = model(data)
                loss, recon_loss, _ = loss_function(recon_batch, data, vq_loss)
                reg_loss = vq_loss
        
        # scale gradients if using mixed precision and CUDA is available
        if use_mixed_precision and torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        else:
            loss.backward()
            optimizer.step()
        
        # update metrics
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_reg_loss += reg_loss.item()
        
        # calculate batch processing time
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        # track memory usage
        if batch_idx % 10 == 0:  # only track every 10 batches to reduce overhead
            memory_usage.append({**get_memory_usage(), 'batch_idx': batch_idx})
        
        # update progress bar
        avg_loss = loss.item() / len(data)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'time/batch': f'{batch_time:.4f}s'})
    
    # calculate average metrics
    num_samples = len(train_loader.dataset)
    avg_loss = train_loss / num_samples
    avg_recon_loss = train_recon_loss / num_samples
    avg_reg_loss = train_reg_loss / num_samples
    
    # calculate average batch time
    avg_batch_time = np.mean(batch_times)
    time_per_image = avg_batch_time / configuration.config['batch_size']
    
    return {'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'reg_loss': avg_reg_loss,
            'time_per_image': time_per_image,
            'batch_times': batch_times,
            'memory_usage': memory_usage}

def evaluate(model, test_loader, loss_function, use_mixed_precision=False):
    model.eval()

    test_loss = 0
    test_recon_loss = 0
    test_reg_loss = 0  # KL divergence for VAE, VQ loss for VQ-VAE
    
    # for calculating MSE and PSNR
    mse_values = []
    psnr_values = []
    
    # for computing SSIM
    ssim_values = []
    
    # for timing inference
    inference_times = []
    
    # for perplexity calculation in VQ-VAE
    if configuration.config['model_type'].lower() == 'vqvae':
        perplexity_values = []
    
    # wrap with tqdm for progress bar
    pbar = tqdm(test_loader, desc='Evaluation')
    
    with torch.no_grad():
        for data, _ in pbar:
            data = data.to(configuration.config['device'])
            
            # time the inference
            start_time = time.time()
            
            # use autocast for mixed precision only if CUDA is available
            with autocast(device_type='cuda', enabled=False) if use_mixed_precision and torch.cuda.is_available() else nullcontext():

                # forward pass based on model type
                if configuration.config['model_type'].lower() == 'vae':
                    recon_batch, mu, logvar = model(data)
                    loss, recon_loss, kld = loss_function(recon_batch, data, mu, logvar, configuration.config['beta'])
                    reg_loss = kld

                else:  # VQ-VAE
                    recon_batch, vq_loss, indices = model(data)
                    loss, recon_loss, _ = loss_function(recon_batch, data, vq_loss)
                    reg_loss = vq_loss
                    
                # calculate perplexity for VQ-VAE (codebook usage metric)
                if configuration.config['model_type'].lower() == 'vqvae':
                    encoding_onehot = F.one_hot(indices.flatten(), num_classes=configuration.config['num_embeddings']).float()
                    avg_probs = torch.mean(encoding_onehot, dim=0)
                    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                    perplexity_values.append(perplexity.item())
                
                # calculate MSE (uniform across model types)
                mse = F.mse_loss(recon_batch, data).item()
                mse_values.append(mse)
                
                # calculate PSNR
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
                psnr_values.append(psnr)
                
                # calculate SSIM
                ssim = compute_ssim(data, recon_batch)
                ssim_values.append(ssim)
            
            # measure inference time
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_reg_loss += reg_loss.item()
            
            # update progress bar
            pbar.set_postfix({'loss': f'{loss.item() / len(data):.4f}', 'time': f'{inference_time:.4f}s'})
    
    # calculate average metrics
    num_samples = len(test_loader.dataset)
    avg_loss = test_loss / num_samples
    avg_recon_loss = test_recon_loss / num_samples
    avg_reg_loss = test_reg_loss / num_samples
    avg_mse = np.mean(mse_values)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    # calculate average inference time
    avg_inference_time = np.mean(inference_times)
    inference_time_per_image = avg_inference_time / configuration.config['batch_size']
    
    # create result dictionary
    result = {'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'reg_loss': avg_reg_loss,
            'mse': avg_mse,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'inference_time_per_image': inference_time_per_image,
            'inference_times': inference_times}
    
    return result

def compute_ssim(x1, x2):

    # move to CPU for computation
    x1 = x1.cpu()
    x2 = x2.cpu()
    
    # compute means
    mu1 = x1.mean(dim=(-1, -2), keepdim=True)
    mu2 = x2.mean(dim=(-1, -2), keepdim=True)
    
    # compute variances and covariance
    sigma1_sq = ((x1 - mu1) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma2_sq = ((x2 - mu2) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma12 = ((x1 - mu1) * (x2 - mu2)).mean(dim=(-1, -2), keepdim=True)
    
    # constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # compute SSIM
    ssim = (((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / 
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)))
    
    # average over batch and channels
    return ssim.mean().item()