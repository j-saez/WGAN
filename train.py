import os
if os.getcwd()[-4:] != 'wgan':
    message = 'Run the file from the the root dir:\n'
    message += 'cd dc_gan\n'
    message += 'python train.py'
    raise Exception(message)

############
## IMPORT ##
############

import time
import torch
from utils.params                   import Params
from utils.datasets                 import load_dataset, get_dataset_transforms
from torch.utils.data.dataloader    import DataLoader
from models.discriminator           import Discriminator
from models.generator               import Generator
from torch.optim                    import RMSprop
from utils.training                 import load_tensorboard_writer, train_one_epoch

###########################
## CONSTANTS AND GLOBALS ##
###########################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} to train the model.".format(DEVICE))

###########################
## Classes and functions ##
###########################

DATASETS_CHS = {'mnist': 1}

##########
## Main ##
##########

if __name__ == '__main__':

    print("DCGAN training loop")

    # Load params
    print("Loading params")
    hyperparms, dataparams = Params().get_params()
    print("\tParams loaded.")

    # Load the data
    print("Loading dataset")
    img_size = (dataparams.img_size,dataparams.img_size)
    transforms = get_dataset_transforms(img_size, DATASETS_CHS[dataparams.dataset_name])
    train_dataset = load_dataset(dataparams.dataset_name, transforms)
    train_dataloader = DataLoader(train_dataset,hyperparms.batch_size,shuffle=True)
    print("\tDataset loaded.")

    # Load the model
    print("Loading the models")
    (img_h, img_w) = img_size
    disc_norm = [False,True,True,True]
    discriminator = Discriminator(DATASETS_CHS[dataparams.dataset_name],img_h,img_w,norm_layer_output=disc_norm,weights_file=hyperparms.disc_pretrained_weights).to(DEVICE)
    gen_norm = [True,True,True,True]
    generator = Generator(hyperparms.z_dim, img_chs=DATASETS_CHS[dataparams.dataset_name], norm_layer_output=gen_norm,weights_file=hyperparms.gen_pretrained_weights).to(DEVICE)
    print("\tModels loaded.")

    # Define optimizer and loss function
    print("Selecting optimizer")
    discriminator_optimizer = RMSprop(discriminator.parameters(), hyperparms.lr)
    generator_optimizer = RMSprop(generator.parameters(), hyperparms.lr)
    print("\tDone.")

    writer, weigths_folder = load_tensorboard_writer(hyperparms, dataparams.dataset_name, disc_norm, gen_norm)
    total_train_baches = int(len(train_dataset) / hyperparms.batch_size)
    fixed_noise = torch.rand(hyperparms.batch_size, hyperparms.z_dim,1,1).to(DEVICE)
    writer.add_images(f'Generated_images', generator(fixed_noise)[:16,:,:].detach().to('cpu').numpy(), 0)
    step = 0

    gen_epoch = int(hyperparms.gen_pretrained_weights.split('_')[-1][:-3]) if hyperparms.gen_pretrained_weights != None else 0
    disc_epoch = int(hyperparms.disc_pretrained_weights.split('_')[-1][:-3]) if hyperparms.disc_pretrained_weights != None else 0

    # Training loop
    print('\n\nStart of the training process.\n')
    for epoch in range(hyperparms.total_epochs):

        epoch_init_time = time.perf_counter()

        step = train_one_epoch(train_dataloader,discriminator,generator,discriminator_optimizer,generator_optimizer,hyperparms,epoch,total_train_baches,DEVICE,writer,step)
        epoch_exec_time = epoch_init_time - time.perf_counter()

        if epoch % hyperparms.test_after_n_epochs == 0:
            # Test model
            with torch.no_grad():
                test_generated_imgs = generator(fixed_noise)[:16,:,:,:].to('cpu')
                writer.add_images(f'Generated_images', test_generated_imgs.numpy(), epoch+1)
                step = step + 1

                gen_epoch = epoch + 1 if hyperparms.gen_pretrained_weights == None else gen_epoch + epoch + 1
                torch.save(generator.state_dict(), weigths_folder+f'Generator_epoch_{gen_epoch}.pt')

                disc_epoch = epoch + 1 if hyperparms.disc_pretrained_weights == None else disc_epoch + epoch + 1
                torch.save(discriminator.state_dict(), weigths_folder+f'Discriminator_epoch_{disc_epoch}.pt')
    print('Training finished.')
