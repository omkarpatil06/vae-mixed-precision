import torch

# --- available activation functions --- #
ACTIVATION_FUNCTIONS = {
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'relu': torch.relu
}

config = {
    # --- training parameters --- # options: "mnist", "fashion_mnist", "cifar10"
    'data': 'fashion_mnist',
    'batch_size': 128,
    'learning_rate': 1e-3,  # mnist lr: 1e-3, cifar10 lr: 1e-4
    'num_epochs': 50,

    # --- model architecture --- # options: "vae", "vqvae"
    'model_type': "vqvae",
    'latent_dim': 32,   # mnist latent dim: 32, cifar10 latent dim: 128
    'beta': 1.0, # weight for the KL divergence

    # vq-vae specific parameters
    'num_embeddings': 512, # mnist num_embeddings: 512, cifar10 num_embeddings: 1024
    'embedding_dim': 64, # mnist embedding_dim: 64, cifar10 embedding_dim: 128
    'commitment_cost': 0.25,
    'decay': 0.99,

    # --- final activation function --- # options: "sigmoid", "tanh", "relu"
    'final_activation': "sigmoid", # mnist final_activation: "sigmoid", cifar10 final_activation: "tanh"

    # --- experiment parameters --- #
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_runs': 3,
    'log_interval': 15,
    'save_model': True,
    'output_dir': 'precision_comparison',

    # --- learning rate scheduler --- #
    'lr_scheduler': True,
    'lr_step_size': 10,
    'lr_gamma': 0.5,}