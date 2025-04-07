import os
import torch
import pandas as pd
import configuration
from experiment import run_experiment
from visualization import plot_comparison_boxplots, plot_loss_curves, plot_batch_time_distribution, plot_memory_usage

def main():
    output_dir = f"precision_comparison_{configuration.config['model_type']}_{configuration.config['final_activation']}"
    os.makedirs(output_dir, exist_ok=True)

    # print configuration
    print("\nConfiguration:")
    for key, value in configuration.config.items():
        print(f"{key}: {value}")
    
    # print system information
    print("\nSystem Information:")
    print(f"  Device: {configuration.config['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # run the experiment
    results = run_experiment()
    
    # flatten results for visualization
    flat_results = []
    for precision, runs in results.items():
        for run in runs:
            run['precision'] = precision  # Add precision mode to each run
            flat_results.extend([run])
    
    # create DataFrame for visualization
    results_df = pd.DataFrame(flat_results)
    
    # create visualizations
    if 'mixed' in results and len(results['mixed']) > 0:
        # plot comparison boxplots
        plot_comparison_boxplots(results_df)
        
        # plot loss curves
        plot_loss_curves(results)
        
        # plot batch time distribution
        plot_batch_time_distribution(results)
        
        # extract and prepare memory usage data for plotting
        memory_data = []
        for precision, runs in results.items():
            for run in runs:
                for train_metric in run['train_metrics']:
                    for usage in train_metric['memory_usage']:
                        usage['precision'] = precision
                        memory_data.append(usage)
        
        # plot memory usage if data is available
        if memory_data:
            memory_df = pd.DataFrame(memory_data)
            plot_memory_usage(memory_df)
    
    print(f"\nEvaluation complete! All results saved to '{configuration.config['output_dir']}' directory.")

if __name__ == "__main__":
    main()