import torch
import configuration
import torch.nn as nn
import torch.nn.functional as F

# --- variational autoencoder --- #
class VAE(nn.Module):

    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.final_activation_fn = configuration.ACTIVATION_FUNCTIONS[configuration.config['final_activation']]
        
        # encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.flatten_size = 4 * 4 * 128
        
        # latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # decoder
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # crop to match original size (28x28)
        self.crop = nn.ZeroPad2d((0, -4, 0, -4))
        
    def encode(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        
        x = x.view(-1, self.flatten_size)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(-1, 128, 4, 4)
        
        x = F.leaky_relu(self.bn4(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.deconv2(x)), 0.2)
        x = self.deconv3(x)
        
        x = self.crop(x)
        
        # apply the specified final activation
        x = self.final_activation_fn(x)
        
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# --- vector quantized variational autoencoder --- #
class VQVAE(nn.Module):

    def __init__(self, in_channels, embedding_dim, num_embeddings, commitment_cost, decay=0.99):
        super(VQVAE, self).__init__()
        
        self.final_activation_fn = configuration.ACTIVATION_FUNCTIONS[configuration.config['final_activation']]
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(0.2)
        )
        
        # vector quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        x = self.decoder(z)
        x = self.final_activation_fn(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices

class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # initialize embedding vectors
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # convert inputs to [batch_size, height, width, embedding_dim]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # find nearest embedding for each input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # quantize by retrieving the embeddings
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # return to original shape
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])

class CIFARVAE(nn.Module):

    def __init__(self, latent_dim, in_channels=3):
        super(CIFARVAE, self).__init__()
        
        self.final_activation_name = configuration.config['final_activation']
        self.final_activation_fn = configuration.ACTIVATION_FUNCTIONS[configuration.config['final_activation']]
        self.in_channels = in_channels
        
        # encoder - adapted for 32x32 RGB images
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 16x16
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 8x8
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 4x4
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 2x2
        self.bn4 = nn.BatchNorm2d(256)
        
        # size after convolutions: 2x2x256
        self.flatten_size = 2 * 2 * 256
        
        # latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # decoder
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)
        
        # transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 4x4
        self.bn5 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 8x8
        self.bn6 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 16x16
        self.bn7 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)  # 32x32
        
    def encode(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        x = x.view(-1, self.flatten_size)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(-1, 256, 2, 2)
        
        x = F.leaky_relu(self.bn5(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.deconv3(x)), 0.2)
        x = self.deconv4(x)
        
        # Apply the final activation
        x = self.final_activation_fn(x)
        
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class CIFARVQVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_embeddings, commitment_cost, decay=0.99):
        super(CIFARVQVAE, self).__init__()
        
        self.final_activation_name = configuration.config['final_activation']
        self.final_activation_fn = configuration.ACTIVATION_FUNCTIONS[configuration.config['final_activation']]
        
        # encoder for 32x32 images
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1)  # 8x8
        )
        
        # vector quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay)
        
        # decoder for 32x32 images
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 32x32
        )
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        x = self.decoder(z)
        x = self.final_activation_fn(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices

# --- create model --- #
def create_model():

    if configuration.config['model_type'].lower() == 'vae':
        if configuration.config['data'].lower() == 'cifar10':
            return CIFARVAE(latent_dim=configuration.config['latent_dim'], in_channels=3)
        else:
            return VAE(latent_dim=configuration.config['latent_dim'],)
    
    elif configuration.config['model_type'].lower() == 'vqvae':
        if configuration.config['data'].lower() == 'cifar10':
            return CIFARVQVAE(in_channels=3,
                             embedding_dim=configuration.config['embedding_dim'],
                             num_embeddings=configuration.config['num_embeddings'],
                             commitment_cost=configuration.config['commitment_cost'],
                             decay=configuration.config['decay'],)
        else:
            return VQVAE(in_channels=1,
                    embedding_dim=configuration.config['embedding_dim'],
                    num_embeddings=configuration.config['num_embeddings'],
                    commitment_cost=configuration.config['commitment_cost'],
                    decay=configuration.config['decay'],)

# get loss function
def get_loss_function():
    if configuration.config['model_type'].lower() == 'vae':

        def vae_loss(recon_x, x, mu, logvar, beta=1.0):
            # binary cross entropy loss for sigmoid activation
            if configuration.config['final_activation'] == 'sigmoid' and configuration.config['data'].lower() != 'cifar10':
                recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
            
            else: # MSE loss for other activations
                recon_loss = F.mse_loss(recon_x, x, reduction='sum')
            
            # KL divergence
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # total loss
            return recon_loss + beta * kld, recon_loss, kld
            
        return vae_loss
    
    elif configuration.config['model_type'].lower() == 'vqvae':
        
        def vqvae_loss(recon_x, x, vq_loss):
                
            if configuration.config['final_activation'] == 'sigmoid' and configuration.config['data'].lower() != 'cifar10':
                # binary cross entropy loss for sigmoid activation
                recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)

            else:  # MSE loss for other activations
                recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
            
            # total loss
            total_loss = recon_loss + vq_loss
            
            return total_loss, recon_loss, vq_loss
            
        return vqvae_loss