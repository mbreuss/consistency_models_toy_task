
import torch
import math
import torch.nn as nn
import numpy as np


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def append_zero(action):
    return torch.cat([action, action.new_zeros([1])])



def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_linear(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an linear noise schedule."""
    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return append_zero(sigmas)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GaussianFourierEmbedding(nn.Module):
    
    def __init__(self, time_embed_dim, device):
        super().__init__()
        self.t_dim = time_embed_dim 
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_embed_dim),
            nn.Linear(time_embed_dim, 2*time_embed_dim),
            nn.Mish(),
            nn.Linear(2*time_embed_dim, time_embed_dim)
        ).to(device)
    
    def forward(self, t):
        return self.embed(t)


def ema_eval_wrapper(func):
    def wrapper(self, *args, **kwargs):
        # Swap model parameters with EMA parameters
        model_state_dict = self.model.state_dict()
        ema_state_dict = self.ema_params
        self.model.load_state_dict(ema_state_dict)
        
        # Call the original function
        result = func(self, *args, **kwargs)
        
        # Swap the parameters back to the original model
        self.model.load_state_dict(model_state_dict)
        return result
    return wrapper



def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from a lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()


def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an uniform distribution."""
    return torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value

def rand_discrete(shape, values, device='cpu', dtype=torch.float32):
    probs = [1/len(values)] * len(values) # set equal probability for all values
    return torch.tensor(np.random.choice(values, size=shape, p=probs), device=device, dtype=dtype)

def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


def rand_split_log_normal(shape, loc, scale_1, scale_2, device='cpu', dtype=torch.float32):
    """Draws samples from a split lognormal distribution."""
    n = torch.randn(shape, device=device, dtype=dtype).abs()
    u = torch.rand(shape, device=device, dtype=dtype)
    n_left = n * -scale_1 + loc
    n_right = n * scale_2 + loc
    ratio = scale_1 / (scale_1 + scale_2)
    return torch.where(u < ratio, n_left, n_right).exp()