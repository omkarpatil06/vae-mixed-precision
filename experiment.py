import time
import torch
import gc
import configuration
import numpy as np
import torch.optim as optim

from model import create_model, get_loss_function
from utilities import load_data
from train import train_epoch, evaluate 
from utilities import set_seed, ensure_dir, visualize_reconstructions, generate_samples, visualize_latent_space

def run_training(run_id, precision_mode, seed=42):
    use_mixed_precision = (precision_mode == 'mixed')
    
    print(f"\n{'='*20} Run {run_id}: {precision_mode} precision {'='*20}")
    
    # set seed for reproducibility
    set_seed(seed + run_id)  # different seed for each run 
    
    # load data with appropriate normalization for the activation function
    train_loader, test_loader = load_data()
    
    # create the appropriate model based on configuration
    model = create_model().to(configuration.config['device'])
    
    # get the appropriate loss function
    loss_function = get_loss_function()
    
    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=configuration.config['learning_rate'])
    
    # learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configuration.config['lr_step_size'], gamma=configuration.config['lr_gamma']) if configuration.config['lr_scheduler'] else None
    
    # for tracking metrics
    train_metrics = []
    test_metrics = []
    
    # total training start time
    total_start_time = time.time()
    
    # initial evaluation
    print("Initial evaluation...")
    initial_metrics = evaluate(model, test_loader, loss_function, use_mixed_precision)
    test_metrics.append(initial_metrics)
    
    # training loop
    for epoch in range(1, configuration.config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{configuration.config['num_epochs']}")
        
        # train for one epoch
        train_metric = train_epoch(model, train_loader, optimizer, epoch, loss_function, use_mixed_precision)
        train_metrics.append(train_metric)
        
        # test the model
        test_metric = evaluate(model, test_loader, loss_function, use_mixed_precision)
        test_metrics.append(test_metric)
        
        # update learning rate
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
        
        # visualize reconstructions periodically
        if epoch % 5 == 0 or epoch == configuration.config['num_epochs']:
            _ = visualize_reconstructions(model, test_loader, epoch, precision_mode, use_mixed_precision)
    
    # final evaluation with the best model
    print("Final evaluation...")
    final_metrics = evaluate(model, test_loader, loss_function, use_mixed_precision)
    
    # visualize reconstructions for the final model
    recon_metrics = visualize_reconstructions(model, test_loader, "final", precision_mode, use_mixed_precision)
    
    # generate samples
    if configuration.config['model_type'].lower() == 'vae':
        generate_samples(model, 10, configuration.config['latent_dim'], use_mixed_precision)

    else:  # VQ-VAE
        generate_samples(model, 10, configuration.config['embedding_dim'], use_mixed_precision)
    
    # visualize latent space
    visualize_latent_space(model, test_loader, precision_mode, use_mixed_precision)
    
    # total training time
    total_training_time = time.time() - total_start_time
    
    # save model if required
    if configuration.config['save_model']:
        ensure_dir(configuration.config['output_dir'])
        torch.save(model.state_dict(), f"{configuration.config['output_dir']}/{configuration.config['model_type']}_{precision_mode}_run{run_id}.pt")
    
    # create result summary
    result = {
        'run_id': run_id,
        'model_type': configuration.config['model_type'],
        'final_activation': configuration.config['final_activation'],
        'precision_mode': precision_mode,
        'epochs_completed': len(train_metrics),
        'total_training_time': total_training_time,
        'final_train_loss': train_metrics[-1]['loss'],
        'final_test_loss': final_metrics['loss'],
        'final_mse': final_metrics['mse'],
        'final_psnr': final_metrics['psnr'],
        'final_ssim': final_metrics['ssim'],
        'avg_train_time_per_image': np.mean([m['time_per_image'] for m in train_metrics]),
        'avg_inference_time_per_image': final_metrics['inference_time_per_image'],
        'peak_gpu_memory_mb': max([item['gpu_reserved_mb'] for m in train_metrics for item in m['memory_usage']], default=0),
        'avg_batch_time': np.mean([t for m in train_metrics for t in m['batch_times']]),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'recon_metrics': recon_metrics
    }
    
    # add model-specific metrics
    if configuration.config['model_type'].lower() == 'vqvae' and 'perplexity' in final_metrics:
        result['perplexity'] = final_metrics['perplexity']
    
    # clean up GPU memory
    model.cpu()
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def run_experiment():

    # initialize results dictionary
    results = {'full': [],
               'mixed': []}
    
    # base seed
    base_seed = 42
    
    # run full precision experiments
    for run_id in range(configuration.config['num_runs']):
        result = run_training(run_id, precision_mode='full', seed=base_seed)
        results['full'].append(result)
    
    # run mixed precision experiments if available
    for run_id in range(configuration.config['num_runs']):
        result = run_training(run_id, precision_mode='mixed', seed=base_seed)
        results['mixed'].append(result)
    
    # save results
    ensure_dir(configuration.config['output_dir'])
    
    return results