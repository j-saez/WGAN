import torch
import torch.nn as nn
from utils.layers import conv_block
from utils.weights import init_weights

class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 img_chs,
                 norm_layer_output=[True,True,True,True,],
                 features=64,
                 weights_file=None):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            conv_block(         z_dim,       features*16, kernel_size=4, stride=1, padding=0, norm=norm_layer_output[0], activation=True, discriminator=False),
            conv_block(         features*16, features*8,  kernel_size=4, stride=2, padding=1, norm=norm_layer_output[1], activation=True, discriminator=False),
            conv_block(         features*8,  features*4,  kernel_size=4, stride=2, padding=1, norm=norm_layer_output[2], activation=True, discriminator=False),
            conv_block(         features*4,  features*2,  kernel_size=4, stride=2, padding=1, norm=norm_layer_output[3], activation=True, discriminator=False),
            nn.ConvTranspose2d( features*2,  img_chs,     kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # We want outputs to be between [-1,1]
        )
        if weights_file == None:
            init_weights(self.model)
        else:
            print(f'Loading generator model weights from {weights_file}')
            pretrained_dict = torch.load(weights_file)
            pretrained_dict = {key.replace("model.", ""): value for key, value in pretrained_dict.items()}
            self.model.load_state_dict(pretrained_dict)
        return


    def forward(self,x):
        if x.size()[-2:] != (1,1):
            raise Exception('X size must be (B,Z,1,1).')
        return self.model(x)
