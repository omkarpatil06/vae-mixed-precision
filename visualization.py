import configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import ensure_dir

def plot_comparison_boxplots(results_df):
    
    # set the style
    sns.set_theme(style="whitegrid")
    
    # metrics to plot
    metrics_to_plot = [
        ('total_training_time', 'Total Training Time (s)'),
        ('avg_train_time_per_image', 'Training Time per Image (ms)'),
        ('avg_inference_time_per_image', 'Inference Time per Image (ms)'),
        ('final_mse', 'Mean Squared Error'),
        ('final_psnr', 'PSNR (dB)'),
        ('final_ssim', 'SSIM')
    ]
    
    # add model-specific metrics
    if configuration.config['model_type'].lower() == 'vqvae' and 'perplexity' in results_df.columns:
        metrics_to_plot.append(('perplexity', 'Codebook Perplexity'))
    
    # adjust figure size based on number of metrics
    num_metrics = len(metrics_to_plot)
    nrows = (num_metrics + 2) // 3  # ceiling division
    ncols = min(3, num_metrics)
    
    # create a figure with multiple subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*5))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, (metric, title) in enumerate(metrics_to_plot):
        if i < len(axes):
            ax = axes[i]
            sns.boxplot(x='precision', y=metric, data=results_df, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Precision Mode')
            
            # Add individual data points
            sns.stripplot(x='precision', y=metric, data=results_df, 
                         ax=ax, color='black', alpha=0.5, jitter=True)
            
            # Add percentage improvement text
            if 'mixed' in results_df['precision'].unique():
                full_mean = results_df[results_df['precision'] == 'full'][metric].mean()
                mixed_mean = results_df[results_df['precision'] == 'mixed'][metric].mean()
                
                if metric in ['final_psnr', 'final_ssim', 'perplexity']:  # Higher is better
                    pct_change = (mixed_mean - full_mean) / full_mean * 100
                    change_txt = f"+{pct_change:.1f}%" if pct_change > 0 else f"{pct_change:.1f}%"
                else:  # Lower is better
                    pct_change = (full_mean - mixed_mean) / full_mean * 100
                    change_txt = f"-{pct_change:.1f}%" if pct_change > 0 else f"{pct_change:.1f}%"
                
                ax.text(0.5, 0.05, change_txt, transform=ax.transAxes, 
                       ha='center', va='bottom', fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # hide any unused subplots
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    ensure_dir(configuration.config['output_dir'])
    plt.savefig(f"{configuration.config['output_dir']}/{configuration.config['model_type']}_metrics_boxplots.png", dpi=300)
    plt.close()

def plot_loss_curves(results):

    # check if we have mixed precision results
    if 'mixed' not in results or not results['mixed']:
        print("Mixed precision results not available for loss curve plotting")
        return
    
    # extract epoch data from results
    full_train_loss = []
    full_test_loss = []
    mixed_train_loss = []
    mixed_test_loss = []
    
    # parse training metrics from all runs
    for precision, runs in results.items():
        
        for run_idx, run in enumerate(runs):
            train_metrics = run.get('train_metrics', [])
            test_metrics = run.get('test_metrics', [])
                
            train_loss = [m['loss'] for m in train_metrics]
            test_loss = [m['loss'] for m in test_metrics]
            
            if precision == 'full':
                full_train_loss.append(train_loss)
                full_test_loss.append(test_loss)
            else:
                mixed_train_loss.append(train_loss)
                mixed_test_loss.append(test_loss)
    
    # plot individual runs instead of averages if there are issues
    plt.figure(figsize=(12, 8))
    
    # plot each run separately
    for i, train_loss in enumerate(full_train_loss):
        test_loss = full_test_loss[i]
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, 'b-', alpha=0.3, label='Full Train' if i == 0 else None)
        
        if len(test_loss[1:]) == len(epochs):
            plt.plot(epochs, test_loss[1:], 'b--', alpha=0.3, label='Full Test' if i == 0 else None)
        else:
            test_epochs = range(1, len(test_loss[1:]) + 1)
            plt.plot(test_epochs, test_loss[1:], 'b--', alpha=0.3, label='Full Test (mismatched)' if i == 0 else None)
    
    for i, train_loss in enumerate(mixed_train_loss):
        test_loss = mixed_test_loss[i]
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, 'r-', alpha=0.3, label='Mixed Train' if i == 0 else None)
        
        if len(test_loss[1:]) == len(epochs):
            plt.plot(epochs, test_loss[1:], 'r--', alpha=0.3, label='Mixed Test' if i == 0 else None)
        else:
            test_epochs = range(1, len(test_loss[1:]) + 1)
            plt.plot(test_epochs, test_loss[1:], 'r--', alpha=0.3, label='Mixed Test (mismatched)' if i == 0 else None)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{configuration.config['model_type'].upper()} Training and Test Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ensure_dir(configuration.config['output_dir'])
    plt.savefig(f"{configuration.config['output_dir']}/{configuration.config['model_type']}_loss_curves.png", dpi=300)
    plt.close()

def plot_batch_time_distribution(results):

    # extract batch times from all runs
    full_batch_times = []
    mixed_batch_times = []
    
    for precision, runs in results.items():
        for run in runs:
            batch_times = np.concatenate([m['batch_times'] for m in run['train_metrics']])
            if precision == 'full':
                full_batch_times.extend(batch_times)
            else:
                mixed_batch_times.extend(batch_times)
    
    # check if we have mixed precision results
    if not mixed_batch_times:
        print("Mixed precision results not available for batch time distribution")
        return
    
    # create data frame for batch times
    batch_times_df = pd.DataFrame({
        'Full Precision': full_batch_times,
        'Mixed Precision': mixed_batch_times if mixed_batch_times else [np.nan]
    })
    
    # plot batch time distributions
    plt.figure(figsize=(12, 6))
    
    # KDE plot
    sns.kdeplot(data=batch_times_df, fill=True, alpha=0.5)
    
    plt.xlabel('Batch Time (s)')
    plt.ylabel('Density')
    plt.title(f"{configuration.config['model_type'].upper()} Batch Time Distribution Comparison")
    
    # add mean lines
    full_mean = np.mean(full_batch_times)
    plt.axvline(full_mean, color='blue', linestyle='--', 
               label=f'Full Mean: {full_mean:.4f}s')
    
    if mixed_batch_times:
        mixed_mean = np.mean(mixed_batch_times)
        plt.axvline(mixed_mean, color='orange', linestyle='--', 
                   label=f'Mixed Mean: {mixed_mean:.4f}s')
        
        # add speedup text
        speedup = (full_mean - mixed_mean) / full_mean * 100
        plt.text(0.5, 0.95, f"Speedup: {speedup:.1f}%", transform=plt.gca().transAxes, 
                ha='center', va='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ensure_dir(configuration.config['output_dir'])
    plt.savefig(f"{configuration.config['output_dir']}/{configuration.config['model_type']}_batch_time_distribution.png", dpi=300)
    plt.close()

def plot_memory_usage(memory_df):

    # check if we have both precision types
    if 'precision' not in memory_df.columns or 'mixed' not in memory_df['precision'].unique():
        print("Mixed precision results not available for memory usage visualization")
        return
    
    # plot memory usage comparison
    plt.figure(figsize=(12, 6))
    
    # box plot of GPU memory
    sns.boxplot(x='precision', y='gpu_reserved_mb', data=memory_df)
    
    # add individual points with jitter
    sns.stripplot(x='precision', y='gpu_reserved_mb', data=memory_df, 
                 color='black', alpha=0.3, jitter=True)
    
    plt.xlabel('Precision Mode')
    plt.ylabel('GPU Memory Reserved (MB)')
    plt.title(f"{configuration.config['model_type'].upper()} GPU Memory Usage Comparison")
    
    # add percentage difference text
    full_mean = memory_df[memory_df['precision'] == 'full']['gpu_reserved_mb'].mean()
    mixed_mean = memory_df[memory_df['precision'] == 'mixed']['gpu_reserved_mb'].mean()
    pct_reduction = (full_mean - mixed_mean) / (full_mean + 1e-6) * 100
    
    plt.text(0.5, 0.05, f"Memory Reduction: {pct_reduction:.1f}%", 
            transform=plt.gca().transAxes, 
            ha='center', va='bottom', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    ensure_dir(configuration.config['output_dir'])
    plt.savefig(f"{configuration.config['output_dir']}/{configuration.config['model_type']}_memory_usage.png", dpi=300)
    plt.close()

def plot_model_quality_comparison(results_df):

    # only create comparison if we have both precision types
    if 'mixed' not in results_df['precision'].unique():
        print("Mixed precision results not available for image quality visualization")
        return
    
    # quality metrics to visualize
    metrics = ['final_mse', 'final_psnr', 'final_ssim']
    metric_names = {
        'final_mse': 'Mean Squared Error',
        'final_psnr': 'PSNR (dB)',
        'final_ssim': 'SSIM'
    }
    
    # add model-specific metrics
    if configuration.config['model_type'].lower() == 'vqvae' and 'perplexity' in results_df.columns:
        metrics.append('perplexity')
        metric_names['perplexity'] = 'Codebook Perplexity'
    
    # group by precision and calculate mean
    precision_means = results_df.groupby('precision')[metrics].mean().reset_index()
    
    # create bar chart
    plt.figure(figsize=(14, 7))
    
    # number of metrics to plot
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.35
    
    # extract values
    full_values = precision_means[precision_means['precision'] == 'full'][metrics].values.flatten()
    mixed_values = precision_means[precision_means['precision'] == 'mixed'][metrics].values.flatten()
    
    # create bar chart
    plt.bar(x - width/2, full_values, width, label='Full Precision')
    plt.bar(x + width/2, mixed_values, width, label='Mixed Precision')
    
    # add percentage difference text
    for i, metric in enumerate(metrics):
        full_val = full_values[i]
        mixed_val = mixed_values[i]
        
        if metric in ['final_psnr', 'final_ssim', 'perplexity']:  # Higher is better
            pct_change = (mixed_val - full_val) / full_val * 100
            change_txt = f"+{pct_change:.1f}%" if pct_change > 0 else f"{pct_change:.1f}%"
        else:  # lower is better
            pct_change = (full_val - mixed_val) / full_val * 100
            change_txt = f"-{pct_change:.1f}%" if pct_change > 0 else f"{pct_change:.1f}%"
        
        # determine position of text
        y_pos = min(full_val, mixed_val) / 2
        plt.text(i, y_pos, change_txt, ha='center', va='center', fontweight='bold')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title(f"{configuration.config['model_type'].upper()} Quality Metrics Comparison")
    plt.xticks(x, [metric_names[m] for m in metrics])
    plt.legend()
    
    plt.tight_layout()
    ensure_dir(configuration.config['output_dir'])
    plt.savefig(f"{configuration.config['output_dir']}/{configuration.config['model_type']}_quality_comparison.png", dpi=300)
    plt.close()