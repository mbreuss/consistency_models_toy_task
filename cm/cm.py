import math 
from functools import partial
import copy 

import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

from .utils import *
from .networks.mlps import ConistencyScoreNetwork



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


class ConsistencyModel(nn.Module):

    def __init__(
            self, 
            sampler_type: str,
            sigma_data: float,
            sigma_min: float,
            sigma_max: float,
            conditioned: bool,
            device: str,
            lr: float = 1e-4,
            rho: int = 7,
            t_steps: int = 100,
            t_steps_min: int = 2,
            ema_rate: float = 0.999,
            target_ema_start: float = 0.95,
            n_sampling_steps: int = 10,
            use_karras_noise_conditioning: bool = True,     
            adapted_karras_noise_conditioning: bool = False,
            sigma_sample_density_type: str = 'loglogistic',
            pre_train_diffusion_on_discrete: bool = False # 'uniform' # 'loglogistic'
    ) -> None:
        super().__init__()
        self.ema_rate = ema_rate
        self.model = ConistencyScoreNetwork(
            x_dim=1,
            hidden_dim=128,
            time_embed_dim=4,
            cond_dim=1,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=1,
            device=device,
            cond_conditional=conditioned
        ).to(device)
        self.target_model = ConistencyScoreNetwork(
            x_dim=1,
            hidden_dim=128,
            time_embed_dim=4,
            cond_dim=1,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=1,
            device=device,
            cond_conditional=conditioned
        ).to(device)
        self.ema_params = copy.deepcopy(self.model.state_dict())
        # now make sure the target model has the same parameters as the model at the beginning
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        # and make sure the target model is not trainable
        self.target_model.requires_grad_(False)
        self.target_ema_start = target_ema_start
        self.device = device
        # use the score wrapper 
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.t_steps = t_steps
        self.t_steps_min = t_steps_min
        self.n_sampling_steps = n_sampling_steps
        self.pre_train_diffusion_on_discrete = pre_train_diffusion_on_discrete
        self.use_karras_noise_conditioning = use_karras_noise_conditioning
        self.sigma_sample_density_type = sigma_sample_density_type
        self.optimizer = torch.optim.AdamW(self.model.get_params(), lr=lr, weight_decay=5e-4, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.RAdam(self.model.get_params(), lr=lr, weight_decay=1e-6)
        self.epochs = 0
        # in their paper they dont mention the c_in and c_noise conidtioning however in their code they use it
        self.adapted_karras_noise_conditioning = adapted_karras_noise_conditioning
        
    def consistency_wrapper(self, model, x, cond, t):
        """
        Performs the consistency wrapper for the given model, x, cond, and t.
        Based on the conditioning from EDM Karras et al. 2022 and adapted for Consistency Models from Song et al. (2023).
        Returns the scaled output of the model.
        
        Args:
        - model (nn.Module): The model to wrap with consistency conditioning.
        - x (torch.Tensor): The input tensor to the model.
        - cond (torch.Tensor): The conditional tensor for the model.
        - t (torch.Tensor): The time tensor for the model.
        
        Returns:
        - scaled_output (torch.Tensor): The output of the model after consistency scaling.
        """
        t = t.to(self.device)
        if len(t.shape) < 2:
            t = t.unsqueeze(1)
        c_skip = self.sigma_data**2 / (
            (t - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (t - self.sigma_min)
            * self.sigma_data
            / (t**2 + self.sigma_data**2) ** 0.5
        )
        # these two are not mentioned in the paper but they use it in their code
        c_in = 1 / (t**2 + self.sigma_data**2) ** 0.5
        if self.use_karras_noise_conditioning:
            t = 0.25 * torch.log(t + 1e-40)
            
        
        consistency_output = model(c_in * x, cond, t)
        scaled_output = c_out * consistency_output + c_skip * x
        return scaled_output
    
    def diffusion_wrapper(self, model, x, cond, t):
        """
        Performs the diffusion wrapper for the given model, x, cond, and t.
        Based on the conditioning from EDM Karras et al. 2022.

        Args:
            model (torch.nn.Module): The neural network model to be used for the diffusion process.
            x (torch.Tensor): The input tensor to the model.
            cond (torch.Tensor): The conditioning tensor to be used during the diffusion process.
            t (float): The time step for the diffusion process.

        Returns:
            torch.Tensor: The scaled output tensor after applying the diffusion wrapper to the model.
        """
        c_skip = self.sigma_data**2 / (
            t ** 2 + self.sigma_data**2
        )
        c_out = (
            t * self.sigma_data / (t**2 + self.sigma_data**2) ** 0.5
        )
        # these two are not mentioned in the paper but they use it in their code
        c_in = 1 / (t**2 + self.sigma_data**2) ** 0.5
        
        t = 0.25 * torch.log(t + 1e-40)
        diffusion_output = model(c_in * x, cond, t)
        scaled_output = c_out * diffusion_output + c_skip * x
        
        return scaled_output
    
    def get_diffusion_scalings(self, sigma):
        """
        Computes the scaling factors for diffusion training at a given time step sigma.

        Args:
        - self: the object instance of the model
        - sigma (float or torch.Tensor): the time step at which to compute the scaling factors
        
        , where self.sigma_data: the data noise level of the diffusion process, set during initialization of the model

        Returns:
        - c_skip (torch.Tensor): the scaling factor for skipping the diffusion model for the given time step sigma
        - c_out (torch.Tensor): the scaling factor for the output of the diffusion model for the given time step sigma
        - c_in (torch.Tensor): the scaling factor for the input of the diffusion model for the given time step sigma

        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in
    
    def sample_discrete_timesteps(self, i, N=100):
        """
        Samples a numpy array of length `N` containing `discrete` diffusion timesteps.

        Args:
            i (int): Index of the current timestep.
            N (int): Total number of timesteps.

        Returns:
            np.ndarray: Array of shape `(N,)` containing discrete timesteps sampled based on the current index `i`.
        """
        t = self.sigma_max ** (1 / self.rho) +  i / (N - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        ) 
        return (t ** self.rho).astype(np.float32)
    
    def sample_seq_timesteps(self, N=100, type='karras'):
        """
        Generates a sequence of N timesteps for the given type.

        Args:
        - self: the object instance of the model
        - N (int): the number of timesteps to generate
        - type (str): the type of sequence to generate, either 'karras', 'linear', or 'exponential'

        Returns:
        - t (torch.Tensor): the generated sequence of timesteps of shape (N,)

        The method generates a sequence of timesteps for the given type using one of the following functions:
        - get_sigmas_karras: a function that generates a sequence of timesteps using the Karras et al. schedule
        - get_sigmas_linear: a function that generates a sequence of timesteps linearly spaced between sigma_min and sigma_max
        - get_sigmas_exponential: a function that generates a sequence of timesteps exponentially spaced between sigma_min and sigma_max
        where,
        - self.sigma_min, self.sigma_max: the minimum and maximum timesteps, set during initialization of the model
        - self.rho: the decay rate for the Karras et al. schedule, set during initialization of the model
        - self.device: the device on which to generate the timesteps, set during initialization of the model

        """
        if type == 'karras':
            t = get_sigmas_karras(N, self.sigma_min, self.sigma_max, self.rho, self.device)
        elif type == 'linear':
            t = get_sigmas_linear(N, self.sigma_min, self.sigma_max, self.device)
        elif type == 'exponential':
            t = get_sigmas_exponential(N, self.sigma_min, self.sigma_max, self.device)
        else:
            raise NotImplementedError('Chosen Scheduler is implemented!')
        return t
    
    def compute_discrete_timesteps(self, train_step, max_steps):
        """
        Computes the discrete timesteps to use during sampling, based on the current training step and maximum number
        of steps. 

        Args:
            train_step (int): The current training step, used to adjust the timesteps.
            max_steps (int): The maximum number of steps for which to compute timesteps.

        Returns:
            np.ndarray: A 1D array of shape (max_steps,), containing the discrete timesteps to use during sampling.
        """
        scales = np.ceil(
            np.sqrt(
                (train_step / max_steps) * ((self.t_steps + 1) ** 2 - self.t_steps_min**2)
                + self.t_steps_min**2
            )
            - 1
        ).astype(np.int32)
        scales = np.maximum(scales, 1)
        scales = scales + 1
        return scales
    
    def train_step(self, x, cond, train_step, max_steps):
        """
        Performs a training step for the Consistecy Policy.

        Args:
            x (torch.Tensor): The input data to train on.
            cond (torch.Tensor): The conditional data for the input.
            train_step (int): The current training step.
            max_steps (int): The maximum number of training steps.

        Returns:
            float: The loss for this training step.
        """
        self.model.train()
        self.target_model.train()
        x = x.to(self.device)
        cond = cond.to(self.device)
        self.optimizer.zero_grad()
        
        # next generate the discrete timesteps
        t_steps = self.compute_discrete_timesteps(train_step, max_steps)
        t = torch.tensor([self.sample_discrete_timesteps(i, t_steps) for i in range(t_steps)]).to(self.device)
        
        t_idx = torch.randint(0, t_steps - 1, (x.shape[0], 1), device=self.device)
        # define the two discrete timesteps for the consistency loss
        t1 = t[t_idx]
        t2 = t[t_idx + 1]
        # compute the loss
        loss = self.consistency_loss(x, cond, t1, t2)
        
        loss.backward()
        self.optimizer.step()
        # first we update the ema weights of self.model
        self._update_ema_weights()

        # next update the target model
        # compute the ema rate for the target model update 
        ema_rate = self.compute_ema_rate(t_steps)
        # print(f'ema rate: {ema_rate}')
        with torch.no_grad():
            update_ema(self.target_model.parameters(), self.model.parameters(), rate=ema_rate)
        
        return loss.item()
    
    def continuous_train_step(self, x, cond, t_steps, max_steps):
        """
        Performs a training step for the Consistecy Policy.

        Args:
            x (torch.Tensor): The input data to train on.
            cond (torch.Tensor): The conditional data for the input.
            train_step (int): The current training step.
            max_steps (int): The maximum number of training steps.

        Returns:
            float: The loss for this training step.
        """
        self.model.train()
        self.target_model.train()
        x = x.to(self.device)
        cond = cond.to(self.device)
        self.optimizer.zero_grad()
        
        # next generate the discrete timesteps
        t_chosen = self.make_sample_density()(shape=(len(x),), device=self.device)
        loss = self.continuous_consistency_loss(x, cond, t_chosen)
        
        loss.backward()
        self.optimizer.step()
        # first we update the ema weights of self.model
        self._update_ema_weights()

        # next update the target model
        # compute the ema rate for the target model update 
        ema_rate = self.compute_ema_rate(t_steps)
        # print(f'ema rate: {ema_rate}')
        with torch.no_grad():
            update_ema(self.target_model.parameters(), self.model.parameters(), rate=ema_rate)
        
        return loss.item()
    
    def continuous_consistency_loss(self, x, cond, t):
        """
        Continuous consistency loss as defined in Song et al. 2023 Eq. (33).
        """
        noise = torch.randn_like(x) 
        x_t = x + noise * append_dims(t, x.ndim)
        x_0_model = self.consistency_wrapper(self.model, x_t, cond, t)
        
        
        # with torch.no_grad():
        # next we compute the jacobian vector product of model output with respect to the input x_t
        # Ensure that x_t and t have the requires_grad attribute set to True
        x_t.requires_grad_(True)
        t.requires_grad_(True)
        
        x_0_target = self.consistency_wrapper(self.target_model, x_t, cond, t)
        
        # compute the jacobian vector product of model output with respect to the input x_t
        jvp_x, jvp_t = self.compute_jacobian_vector_products(x_0_target, x_t, t)

        jvp_t = jvp_t.unsqueeze(1)
        # we use the uniform weight function thus is is not explicitly defined here 
        # this is equation (33) in the paper from Song et al. 2023
        consistency_loss = torch.mean( (x_0_model.T * (jvp_t + jvp_x * noise).detach()) )
        
        return consistency_loss
    
    @staticmethod
    def compute_jacobian_vector_products(x_0_target, x_t, t):
        # Compute the Jacobian-vector product with respect to x_t
        jvp_x, = torch.autograd.grad(
            x_0_target, x_t, grad_outputs=torch.ones_like(x_0_target), retain_graph=True
        )

        # Compute the Jacobian-vector product with respect to t
        jvp_t, = torch.autograd.grad(
            x_0_target, t, grad_outputs=torch.ones_like(x_0_target), retain_graph=True
        )

        return jvp_x, jvp_t  
    
    def diffusion_train_step(self,  x, cond, train_step, max_steps):
        """
        Computes the training loss and performs a single update step for the score-based model.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, dim)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)

        Returns:
        - loss.item() (float): the scalar value of the training loss for this batch

        """
        self.model.train()
        self.target_model.train()
        x = x.to(self.device)
        cond = cond.to(self.device)
        self.optimizer.zero_grad()
        if self.pre_train_diffusion_on_discrete:
        # next generate the discrete timesteps
            t_steps = self.compute_discrete_timesteps(train_step, max_steps)
            t = torch.tensor([self.sample_discrete_timesteps(i, t_steps) for i in range(t_steps)]).to(self.device)
            t_idx = torch.randint(0, t_steps - 1, (x.shape[0], 1), device=self.device)
            # compute the loss
            t_chosen = t[t_idx]
        else:
            t_chosen = self.make_sample_density()(shape=(len(x),), device=self.device)
        loss = self.diffusion_loss(x, cond, t_chosen)
        loss.backward()
        self.optimizer.step()
        # first we update the ema weights of self.model
        self._update_ema_weights()
        return loss.item()

    def _update_ema_weights(self):
        """
        Updates the exponential moving average (EMA) weights of the model.

        Args:
        - self: the object instance of the model

        The method performs the following steps:
        1. Gets the state dictionary of the self.model.
        2. Updates the EMA weights for each parameter by computing the weighted average between the current EMA weights
        and the current model weights, using the EMA rate parameter.
        3. Sets the new EMA weights for each parameter in self.ema_params.

        Note: the EMA rate parameter and the EMA weights dictionary self.ema_params are set during initialization of the model.
        """
        state_dict = self.model.state_dict()
        for key, ema_param in self.ema_params.items():
            ema_param.data = (1 - self.ema_rate) * ema_param.data + self.ema_rate * state_dict[key].data

    def compute_ema_rate(self, t_steps):
        """
        Compute the ema rate based on the scheduler proposed in Song et al. 2023. paper
        This is used to update the target model and not the weights of the model.
        """
        c = -np.log(self.target_ema_start) * self.t_steps_min
        target_ema = np.exp(-c / (t_steps + 1e-44))
        return target_ema
    
    def eval_step(self, x, cond):
        """
        Eval step method to compute the loss for the action prediction.
        """
        self.model.eval()
        self.target_model.eval()
        x = x.to(self.device)
        cond = cond.to(self.device)
        # next generate the discrete timesteps
        t = [self.sample_discrete_timesteps(i) for i in range(self.t_steps)]
        # compute the loss
        x_T = torch.randn_like(x) * self.sigma_max
        pred_x = self. sample(x_T, cond, t)
        loss = torch.nn.functional.mse_loss(pred_x, x)
        return loss
    
    def consistency_loss(self, x, cond, t1, t2):
        """
        Compute the consistency training loss for the given input tensor x, conditioning tensor cond, and the two 
        discrete timesteps t1 and t2. This method generates a noisy version of x and uses the consistency model to 
        denoise it at timestep t1. Then, it uses the ground truth action to compute the gradient at timestep t2 
        to update x. Finally, it uses the target model to denoise x at timestep t2 and computes the MSE loss 
        between the denoised versions of x at the two timesteps.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim].
            cond (torch.Tensor): Conditioning tensor of shape [batch_size, cond_dim].
            t1 (torch.Tensor): First discrete timestep tensor of shape [batch_size, 1].
            t2 (torch.Tensor): Second discrete timestep tensor of shape [batch_size, 1].

        Returns:
            torch.Tensor: Consistency loss tensor of shape [].
        """
        noise = torch.randn_like(x)
        x_1 = x + noise * t1
        x_1_denoised = self.consistency_wrapper(self.model, x_1, cond, t1)
        
        # compute the loss for the second timestep
        # with the ground truth action to compute the gradient
        # x_2 = self.euler_update_step(x_1, t1, t2, x)
        x_2 = x + noise * t2

        with torch.no_grad():
            x_2_denoised = self.consistency_wrapper(self.target_model, x_2, cond, t2)
            
        return torch.nn.functional.mse_loss(x_1_denoised, x_2_denoised).mean()
    
    def diffusion_loss(self, x, cond, t):
        """
        Computes the diffusion training loss for the given model, input, condition, and time.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, channels, height, width)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)
        - t (torch.Tensor): the time step tensor of shape (batch_size,)

        Returns:
        - loss (torch.Tensor): the diffusion training loss tensor of shape ()

        The diffusion training loss is computed based on the following equation from Karras et al. 2022:
        loss = (model_output - target)^2.mean()
        where,
        - noise: a tensor of the same shape as x, containing randomly sampled noise
        - x_1: a tensor of the same shape as x, obtained by adding the noise tensor to x
        - c_skip, c_out, c_in: scaling tensors obtained from the diffusion scalings for the given time step
        - t: a tensor of the same shape as t, obtained by taking the natural logarithm of t and dividing it by 4
        - model_output: the output tensor of the model for the input x_1, condition cond, and time t
        - target: the target tensor for the given input x, scaling tensors c_skip, c_out, c_in, and time t
        """
        noise = torch.randn_like(x)
        x_1 = x + noise * append_dims(t, x.ndim)
        c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4
        model_output = self.model(x_1 * c_in, cond, t)
        target = (x - c_skip * x_1) / c_out
        return (model_output - target).pow(2).mean()
        
    @torch.no_grad()
    @ema_eval_wrapper
    def sample_singlestep(self, x_shape, cond, return_seq=False):
        """
        Samples a single step from the trained consistency model. 
        If return_seq is True, returns a list of sampled tensors, 
        otherwise returns a single tensor. 
        
        Args:
        - x_shape (tuple): the shape of the tensor to be sampled.
        - cond (torch.Tensor or None): the conditional tensor.
        - return_seq (bool, optional): whether to return a list of sampled tensors (default False).
        
        Returns:
        - (torch.Tensor or list): the sampled tensor(s).
        """
        sampled_x = []
        self.model.eval()
        if cond is not None:
            cond = cond.to(self.device)

        x = torch.randn_like(x_shape).to(self.device) * self.sigma_max * 1.5
        sampled_x.append(x)
        x = self.consistency_wrapper(self.model, x, cond, torch.tensor([self.sigma_max]))
        sampled_x.append(x)
        if return_seq:
            return sampled_x
        else:
            return x
    
    @torch.no_grad()
    @ema_eval_wrapper
    def sample_multistep(self, x_shape, cond, n_sampling_steps=None, return_seq=False):
        """
        Samples from the consistency model in multiple steps with the sampler
        proposed by Song et al. (2023)
        
        Args:
        - x_shape (tuple or list): The shape of the input tensor.
        - cond (torch.Tensor): The conditional tensor.
        - n_sampling_steps (int, optional): The number of sampling steps to perform. If not provided, defaults to
            `self.n_sampling_steps`.
        - return_seq (bool, optional): Whether to return the entire sequence of samples, or only the last one. If
            True, returns a list of samples; otherwise, returns a single tensor.
        
        Returns:
        - (torch.Tensor or list of torch.Tensor): The sampled tensor(s).
        """
        self.model.eval()
        if cond is not None:
            cond = cond.to(self.device)
        x = torch.randn_like(x_shape).to(self.device) * self.sigma_max * 1.5
        sampled_x = []
        if n_sampling_steps is None:
            n_sampling_steps = self.n_sampling_steps
        
        # sample the sequence of timesteps
        ts = self.sample_seq_timesteps(N=n_sampling_steps, type='linear')
        # ts = self.sample_seq_timesteps(N=n_sampling_steps, type='linear')
        # make the initial prediction at the maximum sigma
        sampled_x.append(x)
        x = self.consistency_wrapper(self.model, x, cond, torch.tensor([self.sigma_max]))
        sampled_x.append(x)
        # iterate over the remaining timesteps
        for t in ts[1:]:
            t = torch.clip(t, self.sigma_min, self.sigma_max)
            z = torch.randn_like(x)
            x = x + torch.sqrt(t**2 - self.sigma_min**2) * z
            x = self.consistency_wrapper(self.model, x, cond, torch.tensor([t]))
            
            sampled_x.append(x)
        if return_seq:
            return sampled_x
        else:
            return x
    
    @torch.no_grad()
    @ema_eval_wrapper
    def sample_diffusion_euler(self, x_shape, cond, n_sampling_steps=None, return_seq=False):
        """
        Sample from the pre-trained diffusion model using the Euler method. This method is used for sanity checking 
        the learned diffusion model. It generates a sequence of samples by taking small steps from one sample to the next. 
        At each step, it generates a new noise from a normal distribution and combines it with the previous sample 
        to get the next sample.
        
        Parameters:
        - x_shape (torch.Tensor): Shape of the input tensor to the model.
        - cond (torch.Tensor): Conditional information for the model.
        - n_sampling_steps (int, optional): Number of sampling steps to take. Defaults to None.
        - return_seq (bool, optional): Whether to return the full sequence of samples or just the final one. 
                                        Defaults to False.
                                        
        Returns:
        - x (torch.Tensor or List[torch.Tensor]): Sampled tensor from the model. If `return_seq=True`, it returns
                                                a list of tensors, otherwise it returns a single tensor.
        """
        self.model.eval()
        if cond is not None:
            cond = cond.to(self.device)
        x = torch.randn_like(x_shape).to(self.device) * self.sigma_max * 1.5
        # x = torch.linspace(-4, 4, len(x_shape)).view(len(x_shape), 1).to(self.device)

        sampled_x = []
        if n_sampling_steps is None:
            n_sampling_steps = self.n_sampling_steps
        
        # sample the sequence of timesteps
        sigmas = self.sample_seq_timesteps(N=n_sampling_steps, type='exponential')
        sampled_x.append(x)
        # iterate over the remaining timesteps
        for i in trange(len(sigmas) - 1, disable=True):
            denoised = self.diffusion_wrapper(self.model, x, cond, sigmas[i])
            x = self.euler_update_step(x, sigmas[i], sigmas[i+1], denoised)
            sampled_x.append(x)
        if return_seq:
            return sampled_x
        else:
            return x
        
    def euler_update_step(self, x, t1, t2, x0):
        """
        Computes a single update step from the Euler sampler with a ground truth value.

        Parameters:
        x (torch.Tensor): The input tensor.
        t1 (torch.Tensor): The initial timestep.
        t2 (torch.Tensor): The final timestep.
        x0 (torch.Tensor): The ground truth value used to compute the Euler update step.

        Returns:
        torch.Tensor: The output tensor after taking the Euler update step.
        """
        denoiser = x0

        d = (x - denoiser) / append_dims(t1, x.ndim)
        samples = x + d * append_dims(t2 - t1, x.ndim)

        return samples
    
    def update_target_network(self):
        """"
        Initializes the target model weights to be the same as the model weights.

        The method updates the `self.target_model` attribute with the same weights as the `self.model`
        attribute. The `self.ema_params` attribute is updated with a deep copy of the `self.model.state_dict()`
        attribute. The `self.target_model` attribute is loaded with a deep copy of the `self.model.state_dict()`
        attribute.

        Returns:
        None
        """
        ema_state_dict = copy.deepcopy(self.ema_params)
        self.model.load_state_dict(ema_state_dict)
        # next we overwrite the target model state dict
        self.ema_params = copy.deepcopy(self.model.state_dict())
        # now make sure the target model has the same parameters as the model at the beginning
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
    
    def make_sample_density(self):
        """
        Returns a function that generates random timesteps based on the chosen sample density.

        Args:
        - self: the object instance of the model

        Returns:
        - sample_density_fn (callable): a function that generates random timesteps

        The method returns a callable function that generates random timesteps based on the chosen sample density.
        The available sample densities are:
        - 'lognormal': generates random timesteps from a log-normal distribution with mean and standard deviation set
                    during initialization of the model also used in Karras et al. (2022)
        - 'loglogistic': generates random timesteps from a log-logistic distribution with location parameter set to the
                        natural logarithm of the sigma_data parameter and scale and range parameters set during initialization
                        of the model
        - 'loguniform': generates random timesteps from a log-uniform distribution with range parameters set during
                        initialization of the model
        - 'uniform': generates random timesteps from a uniform distribution with range parameters set during initialization
                    of the model
        - 'v-diffusion': generates random timesteps using the Variational Diffusion sampler with range parameters set during
                        initialization of the model
        - 'discrete': generates random timesteps from the noise schedule using the exponential density
        - 'split-lognormal': generates random timesteps from a split log-normal distribution with mean and standard deviation
                            set during initialization of the model
        """
        sd_config = []
        
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(rand_discrete, values=sigmas)
        else:
            raise ValueError('Unknown sample density type')
    
    def gradient_field(self, x, t, cond=None):
        """
        Computes the complete gradient field of the diffusion model with respect to the input x and time t.

        Args:
            x (torch.Tensor): The input tensor for which to compute the gradient field.
            t (float or torch.Tensor): The time at which to compute the gradient field.
            cond (torch.Tensor, optional): The conditional information to use in the computation. Defaults to None.

        Returns:
            torch.Tensor: A tensor representing the gradient field of the diffusion model with respect to x and t.
        """
        x_shape = x.shape
        x = torch.tensor(x, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t = torch.tensor(t, dtype=torch.float32, device=self.device).reshape(-1, 1)
        denoised = self.diffusion_wrapper(self.model, x, cond, t)
        d = (denoised-x ) / append_dims(t, x.ndim)
        d = d.reshape(x_shape)
        return d
    
    

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = ((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho) # [:-1]
    return sigmas.to(device)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]