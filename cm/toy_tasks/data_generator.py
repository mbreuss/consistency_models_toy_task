
import torch 
import numpy as np
from torch.distributions import Normal
import torch
from torch.distributions.normal import Normal


class DataGenerator:
    def __init__(self, dist_type: str):
        self.dist_type = dist_type
        self.func_mapping = {
            "two_gmm_1D": (self.two_gmm_1D, self.two_gmm_1D_log_prob),
            "uneven_two_gmm_1D": (self.uneven_two_gmm_1D, self.uneven_two_gmm_1D_log_prob),
            "three_gmm_1D": (self.three_gmm_1D, self.three_gmm_1D_log_prob),
            "single_gaussian_1D": (self.single_gaussian_1D, self.single_gaussian_1D_log_prob),
        }
        if self.dist_type not in self.func_mapping:
            raise ValueError("Invalid distribution type")
        self.sample_func, self.log_prob_func = self.func_mapping[self.dist_type]

    def generate_samples(self, num_samples: int):
        """
        Generate `num_samples` samples and labels using the `sample_func`.
        
        Args:
            num_samples (int): Number of samples to generate.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the generated samples and labels.
        """
        samples, labels = self.sample_func(num_samples)
        return samples, labels
    
    def compute_log_prob(self, samples, exp: bool = False):
        """
        Compute the logarithm of probability density function (pdf) of the given `samples`
        using the `log_prob_func`. If `exp` is True, return exponentiated log probability.
        
        Args:
            samples (np.ndarray): Samples for which pdf is to be computed.
            exp (bool, optional): If True, return exponentiated log probability.
                Default is False.
        
        Returns:
            np.ndarray: Logarithm of probability density function (pdf) of the given `samples`.
                If `exp` is True, exponentiated log probability is returned.
        """
        return self.log_prob_func(samples, exp=exp)

    @staticmethod
    def two_gmm_1D(num_samples,):
        """
        Generates `num_samples` samples from a 1D mixture of two Gaussians with equal weights.
        
        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two torch tensors containing the generated
            samples and binary labels indicating which Gaussian component the sample is from.
        """
        g1 = Normal(loc=-1.5, scale=0.3)
        g2 = Normal(loc=1.5, scale=0.3)
        mixture_probs = torch.ones(num_samples) * 0.5
        is_from_g1 = torch.bernoulli(mixture_probs).bool()
        samples = torch.where(is_from_g1, g1.sample((num_samples,)), g2.sample((num_samples,)))
        return samples, is_from_g1.int()

    @staticmethod
    def uneven_two_gmm_1D(num_samples, w1=0.7):
        """
        Generates `num_samples` samples from a 1D mixture of two Gaussians with weights `w1` and `w2`.
        
        Args:
            num_samples (int): Number of samples to generate.
            w1 (float, optional): Weight of first Gaussian component. Default is 0.7.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two torch tensors containing the generated
            samples and binary labels indicating which Gaussian component the sample is from.
        """
        g1 = Normal(loc=-1.5, scale=0.3)
        g2 = Normal(loc=1.5, scale=0.2)
        mixture_probs = torch.tensor([w1, 1-w1])
        is_from_g1 = torch.bernoulli(mixture_probs.repeat(num_samples, 1)).view(num_samples, -1).bool().squeeze()
        
        samples_g1 = g1.sample((num_samples, 1))
        samples_g2 = g2.sample((num_samples, 1))
        samples = torch.where(is_from_g1, samples_g1, samples_g2).squeeze()

        return samples, is_from_g1.int()
    
    @staticmethod
    def single_gaussian_1D(num_samples):
        """
        Generates `num_samples` samples from a 1D Gaussian distribution.
        
        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two torch tensors containing the generated
            samples and binary labels indicating which Gaussian component the sample is from.
            Since there is only one Gaussian component, all labels will be zero.
        """
        g1 = Normal(loc=1, scale=0.2)
        samples = g1.sample((num_samples, 1))
        return samples, torch.zeros(num_samples).int()

    @staticmethod
    def three_gmm_1D(num_samples):
        """
        Generates `num_samples` samples from a 1D mixture of three Gaussians with equal weights.
        
        Args:
            num_samples (int): Number of samples to generate.
            exp (bool, optional): If True, return exponentiated log probability. Default is False.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two torch tensors containing the generated
            samples and integer labels indicating which Gaussian component the sample is from.
        """
        g1 = Normal(loc=-1.5, scale=0.2)
        g2 = Normal(loc=0, scale=0.2)
        g3 = Normal(loc=1.5, scale=0.2)
        mixture_probs = torch.ones(3) / 3
        component_assignments = torch.multinomial(mixture_probs, num_samples, replacement=True)
        samples = torch.zeros(num_samples, 1)
        
        g1_mask = (component_assignments == 0)
        g2_mask = (component_assignments == 1)
        g3_mask = (component_assignments == 2)
        
        samples[g1_mask] = g1.sample((g1_mask.sum(), )).view(-1, 1)
        samples[g2_mask] = g2.sample((g2_mask.sum(), )).view(-1, 1)
        samples[g3_mask] = g3.sample((g3_mask.sum(), )).view(-1, 1)
        
        return samples, component_assignments.int()

    @staticmethod
    def two_gmm_1D_log_prob(z, exp=False):
        """
        Computes the logarithm of the probability density function (pdf) of a 1D mixture of two Gaussians
        with equal weights at the given points `z`.
        
        Args:
            z (torch.Tensor): Points at which to compute the pdf.
            exp (bool, optional): If True, return exponentiated log probability. Default is False.
        
        Returns:
            torch.Tensor: Logarithm of probability density function (pdf) of a 1D mixture of two Gaussians
            with equal weights at the given points `z`. If `exp` is True, exponentiated log probability
            is returned.
        """
        g1 = Normal(loc=-1.5, scale=0.3)
        g2 = Normal(loc=1.5, scale=0.3)
        f = torch.log(0.5 * (g1.log_prob(z).exp() + g2.log_prob(z).exp()))
        if exp:
            return torch.exp(f)
        else:
            return f
    
    @staticmethod
    def uneven_two_gmm_1D_log_prob(z, w1=0.7, exp=False):
        """
        Computes the logarithm of the probability density function (pdf) of a 1D mixture of two Gaussians
        with weights `w1` and `w2` at the given points `z`.
        
        Args:
            z (torch.Tensor): Points at which to compute the pdf.
            w1 (float, optional): Weight of first Gaussian component. Default is 0.7.
            exp (bool, optional): If True, return exponentiated log probability. Default is False.
        
        Returns:
            torch.Tensor: Logarithm of probability density function (pdf) of a 1D mixture of two Gaussians
            with weights `w1` and `w2` at the given points `z`. If `exp` is True, exponentiated log probability
            is returned.
        """
        g1 = Normal(loc=-1.5, scale=0.3)
        g2 = Normal(loc=1.5, scale=0.2)
        f = torch.log(w1 * g1.log_prob(z).exp() + (1 - w1) * g2.log_prob(z).exp())
        if exp:
            return torch.exp(f)
        else:
            return f

    @staticmethod
    def three_gmm_1D_log_prob(z,  exp=False):
        """
        Computes the logarithm of the probability density function (pdf) of a 1D mixture of three Gaussians
        with equal weights at the given points `z`.
        
        Args:
            z (torch.Tensor): Points at which to compute the pdf.
            exp (bool, optional): If True, return exponentiated log probability. Default is False.
        
        Returns:
            torch.Tensor: Logarithm of probability density function (pdf) of a 1D mixture of three Gaussians
            with equal weights at the given points `z`. If `exp` is True, exponentiated log probability
            is returned.
        """
        g1 = Normal(loc=-1.5, scale=0.2)
        g2 = Normal(loc=0, scale=0.2)
        g3 = Normal(loc=1.5, scale=0.2)
        f = torch.log(1/3 * (g1.log_prob(z).exp() + g2.log_prob(z).exp() + g3.log_prob(z).exp()))
        if exp:
            return torch.exp(f)
        else:
            return f

    @staticmethod
    def single_gaussian_1D_log_prob(z, exp=False):
        """
        Computes the logarithm of the probability density function (pdf) of a 1D Gaussian
        distribution at the given points `z`.
        
        Args:
            z (torch.Tensor): Points at which to compute the pdf.
            exp (bool, optional): If True, return exponentiated log probability. Default is False.
        
        Returns:
            torch.Tensor: Logarithm of probability density function (pdf) of a 1D Gaussian
            distribution at the given points `z`. If `exp` is True, exponentiated log probability
            is returned.
        """
        g = Normal(loc=1, scale=0.2)
        f = g.log_prob(z)
        if exp:
            return torch.exp(f)
        else:
            return f
    

