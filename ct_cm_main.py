
from tqdm import tqdm

from cm.cm import ConsistencyModel
from cm.toy_tasks.data_generator import DataGenerator
from cm.visualization.vis_utils import plot_main_figure

"""
Continuous training of the consistency model on a toy task.
For better performance, one can pre-training the model with the karras diffusion objective
and then use the weights as initialization for the consistency model.
"""

if __name__ == "__main__":

    device = 'cpu'
    use_pretraining = True
    n_sampling_steps = 20
    
    cm = ConsistencyModel(
        lr=1e-4,
        sampler_type='onestep',
        sigma_data=0.5,
        sigma_min=0.01,
        sigma_max=1,
        conditioned=False,
        device='cuda',
        rho=7,
        t_steps_min=200,
        t_steps=500,
        ema_rate=0.999,
        n_sampling_steps=n_sampling_steps,
        use_karras_noise_conditioning=True,    
    )
    train_epochs = 2000
    # chose one of the following toy tasks: 'three_gmm_1D' 'uneven_two_gmm_1D' 'two_gmm_1D' 'single_gaussian_1D'
    data_manager = DataGenerator('three_gmm_1D') 
    samples, cond = data_manager.generate_samples(10000)
    samples = samples.reshape(-1, 1).to(device)
    pbar = tqdm(range(train_epochs))
    
    # Pretraining with karras diffusion objective if desired
    if use_pretraining:
        for i in range(train_epochs):
            cond = cond.reshape(-1, 1).to(device)        
            loss = cm.diffusion_train_step(samples, cond, i, train_epochs)
            pbar.set_description(f"Step {i}, Loss: {loss:.8f}")
            pbar.update(1)

        # plot the results of the pretraining diffusion model to compare with the consistency model
        plot_main_figure(
            data_manager.compute_log_prob, 
            cm, 
            1000, 
            train_epochs, 
            sampling_method='euler', 
            x_range=[-4, 4], 
            save_path='./plots',
            n_sampling_steps=n_sampling_steps,
        )
        
        cm.update_target_network()
        pbar = tqdm(range(train_epochs))
    
    # Continuous training for the consistency model
    for i in range(train_epochs):
        cond = cond.reshape(-1, 1).to(device)        
        loss = cm.continuous_train_step(samples, cond, i, train_epochs)
        pbar.set_description(f"Step {i}, Loss: {loss:.8f}")
        pbar.update(1)
    
    # Plotting the results of the training
    # We do this for the one-step and the multi-step sampler to compare the results
    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        100, 
        train_epochs, 
        sampling_method='onestep', 
        x_range=[-4, 4], 
        save_path='./plots'
    )
    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        100, 
        train_epochs, 
        sampling_method='multistep', 
        n_sampling_steps=n_sampling_steps,
        x_range=[-4, 4], 
        save_path='./plots'
    )
    print('done')