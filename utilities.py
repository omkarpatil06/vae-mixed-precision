import os
import torch
import psutil
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F

import configuration
import matplotlib.pyplot as plt
from torch.amp import autocast
from contextlib import nullcontext

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_data():

    # --- load training and test data --- #
    if configuration.config['data'] == 'mnist':
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

    elif configuration.config['data'] == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        test_dataset = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
    
    elif configuration.config['data'] == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=configuration.config['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=configuration.config['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, test_loader

# --- get a sample batch from the test loader --- #
def get_sample_batch(test_loader):
    for data, labels in test_loader:
        return data, labels

# --- set the seed for reproducibility --- #
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- get GPU memory usage --- #
def get_memory_usage():

    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        return {'gpu_allocated_mb': gpu_memory_allocated,
                'gpu_reserved_mb': gpu_memory_reserved, }
    
    else:
        return {'gpu_allocated_mb': 0,
                'gpu_reserved_mb': 0, }

# --- get CPU memory usage --- #
def get_cpu_memory():
    return {'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_mb': psutil.virtual_memory().used / (1024 * 1024)}

# --- ensure a directory exists --- #
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- visualize reconstructions from sample batch --- #
def visualize_reconstructions(model, test_loader, epoch, precision_mode, use_mixed_precision=False):
    model.eval()

    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(configuration.config['device'])
        
        # use autocast for inference if mixed precision and CUDA is available
        with autocast(device_type='cuda', enabled=False) if use_mixed_precision and torch.cuda.is_available() else nullcontext():

            if configuration.config['model_type'] == 'vae':
                x_recon_batch, _, _ = model(data)

            else:  # VQ-VAE
                x_recon_batch, _, _ = model(data)
        
        # - move to CPU and convert to numpy - #
        data_np = data.cpu().numpy()
        recon_np = x_recon_batch.cpu().numpy()
        
        # compute per-image MSE, PSNR, and SSIM
        metrics = []
        for i in range(len(data)):
            if configuration.config['data'].lower() == 'cifar10':
                #  find mse
                mse = np.mean((data_np[i] - recon_np[i]) ** 2)
                
                #  find psnr
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
                
                # compute simple ssim
                x1 = data_np[i]
                x2 = recon_np[i]
                
                # for rgb images, compute ssim per channel and average
                ssim_vals = []
                for c in range(data_np.shape[1]):  # for each channel
                    x1_c = x1[c][np.newaxis, :, :]  # add channel dimension back using numpy
                    x2_c = x2[c][np.newaxis, :, :]
                    
                    # means
                    mu1 = x1_c.mean()
                    mu2 = x2_c.mean()
                    
                    # variances and covariance
                    sigma1_sq = ((x1_c - mu1) ** 2).mean()
                    sigma2_sq = ((x2_c - mu2) ** 2).mean()
                    sigma12 = ((x1_c - mu1) * (x2_c - mu2)).mean()
                    
                    # constants
                    C1 = 0.01 ** 2
                    C2 = 0.03 ** 2
                    
                    # ssim for this channel
                    ssim_c = (((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / 
                             ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))).item()
                    ssim_vals.append(ssim_c)
                
                # average ssim across channels
                ssim = np.mean(ssim_vals)
                
                metrics.append({'mse': mse,
                              'psnr': psnr,
                              'ssim': ssim})
            else:
                # find mse
                mse = np.mean((data_np[i][0] - recon_np[i][0]) ** 2)
                
                # find psnr
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
                
                # compute simple ssim
                x1 = data_np[i][0]
                x2 = recon_np[i][0]
                
                # find means
                mu1, mu2 = x1.mean(), x2.mean()
                
                # find variances and covariance
                sigma1_sq = ((x1 - mu1) ** 2).mean()
                sigma2_sq = ((x2 - mu2) ** 2).mean()
                sigma12 = ((x1 - mu1) * (x2 - mu2)).mean()
                
                # constants
                C1 = 0.01 ** 2
                C2 = 0.03 ** 2
                
                # find ssim
                ssim = (((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / 
                    ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)))
                
                metrics.append({'mse': mse,
                                'psnr': psnr,
                                'ssim': ssim})
        
        # plot original and reconstruction with metrics
        num_samples = min(10, len(data))
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
        
        for i in range(num_samples):
            # original images
            if configuration.config['data'].lower() == 'cifar10':
                # Normalize to [0,1] range before transposing
                # Note: The model's output might be outside [0,1] range, so we clip it
                data_normalized = np.clip(data_np[i], 0, 1)
                recon_normalized = np.clip(recon_np[i], 0, 1)
                # Transpose after normalization to maintain proper RGB order
                axes[0, i].imshow(np.transpose(data_normalized, (1, 2, 0)), vmin=0, vmax=1)
            else:
                axes[0, i].imshow(data_np[i][0], cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title('Original')
            
            # reconstructed images
            if configuration.config['data'].lower() == 'cifar10':
                axes[1, i].imshow(np.transpose(recon_normalized, (1, 2, 0)), vmin=0, vmax=1)
            else:
                axes[1, i].imshow(recon_np[i][0], cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'MSE: {metrics[i]["mse"]:.2f}\nPSNR: {metrics[i]["psnr"]:.2f}\nSSIM: {metrics[i]["ssim"]:.2f}')
        
        plt.tight_layout()
        ensure_dir(configuration.config['output_dir'])
        plt.savefig(f"{configuration.config['output_dir']}/{configuration.config['model_type']}_{precision_mode}_recon_epoch_{epoch}.png")
        plt.close()
        
        return metrics

# --- generate samples --- #
def generate_samples(model, num_samples, latent_dim, use_mixed_precision=False):
    model.eval()
    with torch.no_grad():
        
        if configuration.config['model_type'].lower() == 'vae':
            # for VAE: sample from latent space 
            z = torch.randn(num_samples, latent_dim).to(configuration.config['device'])
            
            # use autocast for inference if mixed precision and CUDA is available
            with autocast(device_type='cuda', enabled=False) if use_mixed_precision and torch.cuda.is_available() else nullcontext():
                samples = model.decode(z)
        
        else:  
            # for VQ-VAE: sample random indices from codebook
            if configuration.config['data'].lower() == 'cifar10':
                h, w = 8, 8
            else:
                h, w = 7, 7 
            
            # sample random indices for each position
            random_indices = torch.randint(0, configuration.config['num_embeddings'], (num_samples, h, w)).to(configuration.config['device'])
            
            # use autocast for inference if mixed precision and CUDA is available
            with autocast(device_type='cuda', enabled=False) if use_mixed_precision and torch.cuda.is_available() else nullcontext():
                
                # convert indices to one-hot
                random_indices_flat = random_indices.view(-1)
                random_one_hot = F.one_hot(random_indices_flat, configuration.config['num_embeddings']).float()
                
                # get the corresponding embeddings
                quantized_flat = torch.matmul(random_one_hot, model.vq.embeddings.weight)
                quantized = quantized_flat.view(num_samples, h, w, configuration.config['embedding_dim'])
                quantized = quantized.permute(0, 3, 1, 2).contiguous()
                
                # decode the random embeddings
                samples = model.decode(quantized)
        
        # move to CPU and convert to numpy
        samples_np = samples.cpu().numpy()
        
        # plot samples
        plt.figure(figsize=(20, 2))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            if configuration.config['data'].lower() == 'cifar10':
                plt.imshow(np.transpose(samples_np[i], (1, 2, 0)))
            else:
                plt.imshow(samples_np[i][0], cmap='gray')
            plt.axis('off')
            plt.title(f'Sample {i+1}')
        
        plt.tight_layout()
        ensure_dir(configuration.config['output_dir'])
        plt.savefig(f"{configuration.config['output_dir']}/{configuration.config['model_type']}_samples.png")
        plt.close()

# --- visualize latent space --- #
def visualize_latent_space(model, test_loader, precision_mode, use_mixed_precision=False):
    model.eval()
    max_samples = 2000 if configuration.config['data'].lower() == 'cifar10' else 10000

    # get latent representations and labels for test data
    sample_count = 0
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            if sample_count >= max_samples:
                break

            data = data.to(configuration.config['device'])
            sample_count += len(data)
            
            # use autocast for inference if mixed precision and CUDA is available
            with autocast(device_type='cuda', enabled=False) if use_mixed_precision and torch.cuda.is_available() else nullcontext():

                if configuration.config['model_type'].lower() == 'vae':
                    mu, _ = model.encode(data)
                    latent_vectors.append(mu.cpu().numpy())

                else:  # VQ-VAE
                    z = model.encode(data)
                    _, _, indices = model.vq(z)
                    
                    # convert indices to a bag-of-words representation
                    indices_flat = indices.view(indices.size(0), -1).cpu().numpy()
                    latent_vectors.append(indices_flat)
            
            labels.append(label.numpy())
    
    # concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # use t-SNE for dimensionality reduction
    
    # for VQ-VAE indices, convert to a more suitable representation
    if configuration.config['model_type'].lower() == 'vqvae':

        # use one-hot encoding for the indices
        latent_vectors_reshaped = latent_vectors.reshape(latent_vectors.shape[0], -1)
        
        # if too high-dimensional, use PCA first
        if latent_vectors_reshaped.shape[1] > 50:
            pca = PCA(n_components=50)
            latent_vectors_reshaped = pca.fit_transform(latent_vectors_reshaped)

    else:
        latent_vectors_reshaped = latent_vectors
    
    # apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_vectors_reshaped)
    
    # plot t-SNE visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)
        
    plt.colorbar(scatter, label='Class')
    plt.title(f"{configuration.config['model_type'].upper()} Latent Space ({precision_mode} precision)")
    plt.tight_layout()
    
    ensure_dir(configuration.config['output_dir'])
    plt.savefig(f"{configuration.config['output_dir']}/{configuration.config['model_type']}_{precision_mode}_latent_tsne.png")
    plt.close() 