
import torch 
import numpy as np
from torch.distributions import Normal
from cm.cm import ConsistencyModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde, norm
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import torch
from torch.distributions.normal import Normal
from cm.cm import ConsistencyModel


def get_test_samples(model, n_samples, sampling_method, n_sampling_steps):
    """
    Obtain the test samples from the given model, based on the specified sampling method and diffusion sampling type.

    Args:
        model (object): Model to be used for sampling (ConsistencyModel or Beso).
        n_samples (int): Number of samples to be taken.
        sampling_method (str, optional): Method to be used for sampling ('multistep', 'onestep', or 'euler').
        n_sampling_steps (int, optional): Number of sampling steps. Defaults to 10.

    Returns:
        test_samples (list): List of test samples obtained from the given model.
    """
    if sampling_method == 'multistep':
        return model.sample_multistep(torch.zeros((n_samples, 1)), None, return_seq=True, n_sampling_steps=n_sampling_steps)
    elif sampling_method == 'onestep':
        return model.sample_singlestep(torch.zeros((n_samples, 1)), None, return_seq=True)
    elif sampling_method == 'euler':
        return model.sample_diffusion_euler(torch.zeros((n_samples, 1)), None, return_seq=True, n_sampling_steps=n_sampling_steps)
    else:
        raise ValueError('sampling_method must be either multistep, onestep or euler')



def plot_main_figure(
    fn, 
    model, 
    n_samples, 
    train_epochs, 
    sampling_method='euler',
    x_range=[-4, 4], 
    n_sampling_steps = 10,
    save_path='/home/moritz/code/cm_1D_Toy_Task/plots'
):  
    """
    Plot the main figure for the given model and sampling method.
    Args:
    fn (callable): Target function to be plotted.
    model (object): Model to be used for sampling (ConsistencyModel or Beso).
    n_samples (int): Number of samples to be taken.
    train_epochs (int): Number of training epochs.
    sampling_method (str, optional): Method to be used for sampling ('multistep', 'onestep', or 'euler'). Defaults to False.
    x_range (list, optional): Range of x values to be plotted. Defaults to [-5, 5].
    n_sampling_steps (int, optional): Number of sampling steps. Defaults to 10.
    save_path (str, optional): Directory to save the plot. Defaults to '/home/moritz/code/cm_1D_Toy_Task/plots'.

    Raises ValueError: If the sampling_method is not one of the specified options ('multistep', 'onestep', or 'euler').
    """
    test_samples = get_test_samples(model, n_samples, sampling_method, n_sampling_steps)
    test_samples = [x.detach().cpu().numpy() for x in test_samples]
    test_samples = np.stack(test_samples, axis=1)

    x_test = np.linspace(x_range[0], x_range[1], n_samples)
    target_fn = fn(torch.tensor(x_test), exp=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax1.set_xlim(*x_range)
    ax2.set_xlim(*x_range)
    ax3.set_xlim(*x_range)

    # Plot target distribution
    ax1.plot(x_test, target_fn, color='black', label='Target Distribution')

    # Plot predicted distribution
    kde = gaussian_kde(test_samples[:, -1, 0], bw_method=0.1)
    predicted_distribution = kde(x_test)
    ax1.plot(x_test, predicted_distribution, label='Predicted Distribution')

    # Create a LineCollection to show colors on the predicted distribution line
    points = np.array([x_test, predicted_distribution]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(predicted_distribution.min(), predicted_distribution.max()))
    lc.set_array(predicted_distribution)
    lc.set_linewidth(2)

    ax1.add_collection(lc)
    stepsize = np.linspace(0, 1, model.n_sampling_steps)
    # stepsize = cm.get_noise_schedule(model.n_sampling_steps, noise_schedule_type='exponential').flip(0)
    # ax2.set_ylim(-0.1, 1.1)
    if sampling_method == 'onestep':
        n_sampling_steps = 1
        stepsize = np.linspace(0, 1, 2)
        ax2.quiver(test_samples[:, 0].reshape(-1),
                    stepsize[0] * np.ones(n_samples),
                    test_samples[:, 1].reshape(-1) - test_samples[:, 0].reshape(-1),
                    stepsize[1] * np.ones(n_samples) - stepsize[0] * np.ones(n_samples),
                    angles='xy', scale_units='xy', scale=1,
                    width=0.001
                    )
    else:
        n_sampling_steps = model.n_sampling_steps
        for i in range(1, n_sampling_steps):
            ax2.quiver(test_samples[:, i - 1].reshape(-1),
                    stepsize[i - 1] * np.ones(n_samples),
                    test_samples[:, i].reshape(-1) - test_samples[:, i-1].reshape(-1),
                    stepsize[i] * np.ones(n_samples) - stepsize[i - 1] * np.ones(n_samples),
                    angles='xy', scale_units='xy', scale=1,
                    width=0.001
                    )
    ax2.set_yticks([stepsize.min(), stepsize.max()])
    ax2.set_ylim(stepsize.min(), stepsize.max())
    
    mu = 0  # mean
    sigma = model.sigma_max  # standard deviation

    # Compute the PDF values for x_test
    prob_samples = norm.pdf(x_test, loc=mu, scale=sigma)
    # Create a LineCollection to show colors on the normal distribution line
    points = np.array([x_test, prob_samples]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(prob_samples.min(), prob_samples.max()))
    lc.set_array(prob_samples)
    lc.set_linewidth(2)

    ax3.add_collection(lc)
    ax3.set_ylim(0, 0.5)

    # ... (previous code remains unchanged)
    ax2.set_xticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax3.set_yticks([])
    ax2.set_yticklabels(['T', '0'])
    ax2.tick_params(axis='y', labelsize=16)
    # ax2.set_yticks('log')
    plt.subplots_adjust(hspace=0)
    plt.savefig(save_path + '/cm_' + sampling_method + f'_epochs_{train_epochs}.png', bbox_inches='tight', pad_inches=0.1)    
    
    print('Plot saved!')
    
