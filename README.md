# VAE Mixed Precision Training Evaluation

This project evaluates the impact of precision training on variational autoencoders (VAEs) and vector quantised VAEs (VQVAEs) using the MNIST, Fashion-MNIST, and CIFAR10 dataset. It compares training with full precision (FP32) and mixed precision (FP16/FP32) across multiple metrics including training speed, memory usage, and reconstruction quality.

## Features

- Comprehensive comparison of full vs. mixed precision training
- Multiple independent runs for statistical significance
- Training and inference time measurements
- Memory usage tracking
- Image quality metrics (MSE, PSNR, SSIM)
- Detailed visualizations and analysis

## Requirements

- Python 3.6+
- PyTorch 1.7.0+
- CUDA-capable GPU for mixed precision training
- See `requirements.txt` for complete dependencies

## Project Structure

```
├── configuration.py        # Configuration settings
├── model.py                # VAE and VQVAE model definition
├── utilities.py            # Utility functions
├── train.py                # Training and evaluation functions
├── visualization.py        # Visualization functions
├── experiment.py           # Experiment runner
├── main_file.py            # Main script
├── requirements.txt        # Dependencies
└── precision_comparison/   # Output directory for results
```

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/vae-mixed-precision.git
cd vae-mixed-precision
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the evaluation with default settings:

```
python main.py
```

The code will:
1. Train VAE models with both full and mixed precision (if available)
2. Run multiple independent trials
3. Generate comprehensive visualizations
4. Perform statistical analysis
5. Create a detailed report

## Configuration

Adjust parameters in `configuration.py` to customize the evaluation:

- `data` : Select the dataset for training
- `model_type` : Select the model type for training
- `batch_size`: Batch size for training
- `num_epochs`: Maximum number of training epochs
- `learning_rate`: Initial learning rate
- `latent_dim`: Dimensionality of the latent space
- `num_runs`: Number of independent trials
- `patience`: Early stopping patience
- `lr_scheduler`: Whether to use learning rate scheduling

## Output

All results are saved to the `precision_comparison` directory:

- Model checkpoints
- Training and evaluation metrics
- Comparative visualizations
- Statistical analysis

## Metrics Evaluated

1. **Speed Metrics**:
   - Total training time
   - Training time per image
   - Inference time per image
   - Batch processing time distribution

2. **Memory Usage**:
   - Peak GPU memory usage

3. **Quality Metrics**:
   - Mean Squared Error (MSE)
   - Peak Signal-to-Noise Ratio (PSNR)
   - Structural Similarity Index (SSIM)

4. **Training Dynamics**:
   - Loss curves
   - Latent space visualization
   - Sample generation quality

## License

MIT

## Acknowledgments

- PyTorch for the deep learning framework
- MNIST, Fashion-MNIST, and CIFAR10 dataset creators
