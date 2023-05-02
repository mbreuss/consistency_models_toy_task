
from tqdm import tqdm

from cm.cd_cm import CDConsistencyModel
from cm.toy_tasks.data_generator import DataGenerator
from cm.visualization.vis_utils import plot_main_figure


"""
Discrete consistency distillation training of the consistency model on a toy task.
We train a diffusion model and the consistency model at the same time and iteratively 
update the weights of the consistency model and the diffusion model.
"""

if __name__ == "__main__":

    device = 'cpu'
    n_sampling_steps = 10
    simultanous_training = False
    cm = CDConsistencyModel(
        lr=1e-4,
        sampler_type='onestep',
        simultanous_training=simultanous_training,
        sigma_data=0.5,
        sigma_min=0.05,
        sigma_max=1,
        conditioned=False,
        device='cuda',
        rho=7,
        t_steps_min=100,
        t_steps=100,
        ema_rate=0.999,
        n_sampling_steps=n_sampling_steps,
        use_karras_noise_conditioning=True,    
    )
    train_epochs = 20002
    # chose one of the following toy tasks: 'three_gmm_1D' 'uneven_two_gmm_1D' 'two_gmm_1D' 'single_gaussian_1D'
    data_manager = DataGenerator('two_gmm_1D')
    samples, cond = data_manager.generate_samples(10000)
    samples = samples.reshape(-1, 1).to(device)
    pbar = tqdm(range(train_epochs))
    
    if not simultanous_training:
        # First pretrain the diffusion model and then train the consistency model
        for i in range(train_epochs):
            cond = cond.reshape(-1, 1).to(device)        
            diff_loss = cm.diffusion_train_step(samples, cond, i, train_epochs)
            pbar.set_description(f"Step {i}, Diff Loss: {diff_loss:.8f}")
            pbar.update(1)
    
    # Train the consistency model either simultanously with the diffusion model or after pretraining
    for i in range(train_epochs):
        cond = cond.reshape(-1, 1).to(device)        
        diff_loss, cd_loss = cm.train_step(samples, cond, i, train_epochs)
        if simultanous_training:
            pbar.set_description(f"Step {i}, CD Loss: {cd_loss:.8f}, Diff Loss: {diff_loss:.8f}")
        else:
            pbar.set_description(f"Step {i}, CD Loss: {cd_loss:.8f}")
        pbar.update(1)
    
    # Plotting the results of the training
    # We do this for the one-step and the multi-step sampler to compare the results
    plot_main_figure(
            data_manager.compute_log_prob, 
            cm, 
            200, 
            train_epochs, 
            sampling_method='euler', 
            n_sampling_steps=n_sampling_steps,
            x_range=[-4, 4], 
            save_path='./plots/cd'
        )
    
    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        200, 
        train_epochs, 
        sampling_method='onestep', 
        x_range=[-4, 4], 
        save_path='./plots/cd'
    )
    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        200, 
        train_epochs, 
        sampling_method='multistep', 
        n_sampling_steps=n_sampling_steps,
        x_range=[-4, 4], 
        save_path='./plots/cd'
    )
            
    print('done')