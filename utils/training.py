import os

from torch._C import _nccl_init_rank, dtype
if os.getcwd()[-4:] != 'wgan':
    message = 'Run the file from the the root dir:\n'
    message += 'cd wgan\n'
    message += 'python train.py'
    raise Exception(message)

import time
import torch
from torch.utils.tensorboard import SummaryWriter

"""
    Name: load_tensorboard_writer
    Description: 
    Inputs:
        >>
    Outputs:
        >>
"""
def load_tensorboard_writer(hyperparams, dataset_name, disc_norm, gen_norm):
    tensorboard_dir = os.getcwd()+'/runs/tensorboard/'
    weights_dir = os.getcwd()+'/runs/weights/'
    model_logs_dir = os.getcwd()+f'/runs/tensorboard/{dataset_name}_lr{hyperparams.lr}_e{hyperparams.total_epochs}_zDim{hyperparams.z_dim}_genLayersNorm{gen_norm}_discLayersNorm{disc_norm}/'
    model_weights_dir = os.getcwd()+f'/runs/weights/{dataset_name}_lr{hyperparams.lr}_e{hyperparams.total_epochs}_zDim{hyperparams.z_dim}_genLayersNorm{gen_norm}_discLayersNorm{disc_norm}/'

    if not os.path.isdir(os.getcwd()+'/runs'):
        os.mkdir(os.getcwd()+'/runs')
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
    if not os.path.isdir(model_logs_dir):
        os.mkdir(model_logs_dir)
    if not os.path.isdir(model_weights_dir):
        os.mkdir(model_weights_dir)

    writer = SummaryWriter(log_dir=model_logs_dir)
    return writer, model_weights_dir

"""
    Name: train_discriminator
    Description: Trains the discriminator model for one epoch.
    Inputs:
        >> disc: Torch model for the discriminator.
        >> gen: Torch model for the generator.
        >> disc_opt: Torch optimizer for the critic.
        >> real_imgs: Torch tensor containing the images with shape (B, CH, H, W).
        >> z_dim: Int value that expresses the total elements for the noise tensor.
        >> n_critic: Int value that expresses the total iters for the critic to do every epoch.
        >> clip_val: Float value that expresses the maximum and minimum value for the critic's weights.
        >> device: Device where the training is taking place.
    Outputs:
        >> mean_disc_loss: Mean discrimination loss value.
        >> mean_real_disc_outputs: Mean critic output value when real data is feeded to the network.
        >> mean_fake_disc_outputs: Mean critic output value when fake data is feeded to the network.
"""
def train_discriminator(disc, gen, disc_opt, real_imgs, z_dim, n_critic, clip_val, device):
    batch_size = real_imgs.size()[0]
    real_disc_outputs = torch.zeros((n_critic, batch_size),dtype=torch.float32)
    fake_disc_outputs = torch.zeros((n_critic, batch_size),dtype=torch.float32)
    losses_disc = torch.zeros((n_critic, batch_size),dtype=torch.float32)
    # The discriminator is more trained than the generator
    for i in range(n_critic):
        disc.zero_grad()

        noise = torch.rand(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = gen(noise)

        real_disc_output = disc(real_imgs).reshape(-1)
        fake_disc_output = disc(fake_imgs.detach()).reshape(-1)
        loss_disc = -1*(real_disc_output.mean() - fake_disc_output.mean()) # -1 as we want to maximize disc loss 

        # Backprop and actualize discriminator weights
        loss_disc.backward()
        disc_opt.step()

        # clip critic weights between -0.01, 0.01
        for p in disc.parameters():
            p.data.clamp_(-clip_val, clip_val)
        
        real_disc_outputs[i] = real_disc_output
        fake_disc_outputs[i] = fake_disc_output
        losses_disc[i]       = loss_disc

    return losses_disc.mean().item(), real_disc_outputs.mean().item(), fake_disc_outputs.mean().item()

"""
    Name: train_generator
    Description: 
    Inputs:
        >>
    Outputs:
        >>
"""
def train_generator(disc,gen,gen_opt,z_dim,batch_size,device):
    noise = torch.rand(batch_size,z_dim,1,1).to(device)
    fake_imgs = gen(noise)
    gen.zero_grad()

    ## Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
    fake_disc_output = disc(fake_imgs).reshape(-1)
    # In the paper it appears a -1. As we want to minimize it is ok to 
    loss_generator = -1.0*fake_disc_output.mean()
    loss_generator.backward()
    gen_opt.step()

    return loss_generator, fake_disc_output.mean().item()

"""
    train_one_epoch

    Description: Trains WGAN  one epoch.
    Inputs:
        >> train_dataloader: Torch Dataloader containing the training data.
        >> disc: Torch model for the discriminator.
        >> gen: Torch model for the generator.
        >> disc_opt: Discriminator optimizer.
        >> gen_opt: Generator optimizer.
        >> hyperparms: Hyperparms object containing the hyperparms values.
        >> epoch: (Int) Current epoch of the training.
        >> total_train_baches: (Int) Total number of batches.
        >> device: Device where the training will take place.
        >> writer: Tensorboard object to track training values and state.
        >> Step: (Int) Current step of the training. It is needed for the writer.
    Outputs:
        >> Step: Updated step value.
"""
def train_one_epoch(train_dataloader,disc,gen,disc_opt,gen_opt,hyperparms,epoch,total_train_baches,device,writer,step):
    for batch_idx, (real_imgs, _) in enumerate(train_dataloader):
        batch_init_time = time.perf_counter()

        # Data to device and to proper data type
        real_imgs = real_imgs.to(device)

        ## Train discriminator
        loss_disc, d_x, d_gx1 = train_discriminator(disc,gen,disc_opt,real_imgs,hyperparms.z_dim,hyperparms.critic_iters, hyperparms.clip_val, device) 

        ## Train generator
        loss_gen, d_gx2 = train_generator(disc,gen,gen_opt,hyperparms.z_dim,hyperparms.batch_size, device)

        batch_final_time = time.perf_counter()
        batch_exec_time = batch_final_time - batch_init_time
        
        if batch_idx % hyperparms.test_after_n_epochs == 0 and batch_idx !=0:
            # To be honest, in GANs the loss does not say much 
            print(f'Epoch {epoch}/{hyperparms.total_epochs} - Batch {batch_idx}/{total_train_baches} - Loss D {loss_disc:.6f} - Loss G {loss_gen:.6f} - D(x): {d_x:.6f} - D(G(x))_1: {d_gx1:.6f} - D(G(x))_2: {d_gx2:.6f} - Batch time {batch_exec_time:.6f} s.')
            writer.add_scalars( f'Loss/', {'Gen': loss_gen, 'Disc': loss_disc}, step)
            writer.add_scalars( f'Disc val/', {'D(x)': d_x, 'D(G(x))_1': d_gx1, 'D(G(x))_2': d_gx2}, step)
            step=step+1
    return step
