import torch
import torch.nn.functional as F
from torch.cuda import device
from helpers import extract
from tqdm import tqdm
from math import pi

def linear_beta_schedule(beta_start, beta_end, timesteps):

    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):

    i = torch.linspace(0.0, timesteps, timesteps +1,device="cuda")
    f_t = torch.cos((pi/2)*(((i/timesteps) + s)/(1+s))) ** 2
    alpha = f_t/ f_t[0]
    beta = 1 - (alpha[1:]/alpha[:-1])
    beta = torch.clamp(beta, 0, 0.0200)
    return beta

def sigmoid_beta_schedule(beta_start, beta_end, timesteps,s_limit=10):

    s_limit = s_limit
    i = torch.linspace(0.0, timesteps, timesteps,device="cuda")
    beta = beta_start + (torch.sigmoid(-s_limit + (((2 * i)/timesteps) * s_limit) ) * (beta_end - beta_start))
    return beta

class Diffusion:
    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):

        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device
        
        self.betas = get_noise_schedule(self.timesteps).to(device)
        self.alpha_prod = torch.cumprod(1 - self.betas, dim=0).to(device)
        
        
        self.sqrt_alpha_prod = torch.sqrt(self.alpha_prod).to(device)
        self.sqrt_minus1_alpha_prod = torch.sqrt(1-self.alpha_prod).to(device)
        
        
        self.alpha = 1 - self.betas.to(device)
        self.sqrt_alpha = torch.sqrt(self.alpha).to(device)
        self.sqrt_by1_alpha = 1/self.sqrt_alpha
        self.sqrt_beta_by1_minus1_alpha = self.betas /self.sqrt_minus1_alpha_prod
        self.sqrt_beta = torch.sqrt(self.betas).to(device)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index,y=None):
        estimated_noise = model(x, t,y)
        beta_t_index = self.betas[t_index].to(self.device)
        sqrt_beta_t_index = torch.sqrt(beta_t_index).to(self.device)
        alpha_t_index = 1-self.betas[t_index].to(self.device)
        alpha_t_index_prev = 1 - self.betas[t_index -1].to(self.device)
        sqrt_by1_alpha = 1/torch.sqrt(alpha_t_index)
        sqrt_beta_by1_minus1_alpha = beta_t_index/self.sqrt_minus1_alpha_prod
        first_half = sqrt_by1_alpha(x - (sqrt_beta_by1_minus1_alpha * estimated_noise))
        if t > 0:
            x_t = first_half + (torch.sqrt(beta_t_index * (1- self.sqrt_minus1_alpha_prod[t_index-1]/self.sqrt_minus1_alpha_prod[t_index])) * torch.randn_like(x).to(self.device))
        else:
            x_t = first_half
        
        return x_t
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3,y=None,class_free_guidance= False,w = 0):
        
        if not isinstance(image_size, int):
            raise TypeError(f"Expected image_size to be an int, but got {type(image_size).__name__}")
        
        x_t = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        for i in reversed(range(self.timesteps)):
            
            t = torch.full((batch_size,), i, device=self.device).long()
            if y is not None:
                y = torch.tensor([y], device=self.device)
            estimated_noise = model(x_t, t,None)
            estimated_noise_classified = model(x_t, t,y)
            estrimated_combined_noise = ((1+w)*estimated_noise_classified) - (w * estimated_noise)
            if class_free_guidance == True:
                estimated_noise =estrimated_combined_noise
            first_half = self.sqrt_by1_alpha[i] * (x_t -(self.sqrt_beta_by1_minus1_alpha[i] * estimated_noise))
            if i>0:
                x_t = first_half  + (self.sqrt_beta[i]* torch.randn_like(x_t).to(self.device) )
            else:
                x_t = first_half
        
        return x_t
    
    def q_sample(self, x_zero, t, noise=None):
        
        if noise is None:
            noise = torch.randn_like(x_zero).to(self.device)
        
        sqrt_alpha_prod = (self.sqrt_alpha_prod[t][:, None, None, None]).to(self.device)  
        sqrt_minus1_alpha_prod = (self.sqrt_minus1_alpha_prod[t][:, None, None, None]).to(self.device)  
        x_t = (sqrt_alpha_prod *  x_zero.to(self.device)) + (sqrt_minus1_alpha_prod * noise)
        return x_t
    
    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1",y=None):
        if noise is None:
            noise = torch.randn_like(x_zero).to(self.device)
        x_t = self.q_sample(x_zero, t, noise)
        estimated_noise = denoise_model(x_t, t,y)
        if loss_type == 'l1':
            
            loss = torch.mean(torch.abs(estimated_noise - noise))
        elif loss_type == 'l2':
            
            loss = torch.mean((estimated_noise - noise) ** 2)
        else:
            raise NotImplementedError()
        return loss
