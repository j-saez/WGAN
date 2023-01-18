import torch.nn as nn
from utils.layers import conv_block
from utils.weights import init_weights

class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 img_chs,
                 norm_layer_output=[True,True,True,True,]):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            conv_block(z_dim,1024, kernel_size=4, stride=1, padding=0, norm=norm_layer_output[0], activation=True, discriminator=False),
            conv_block( 1024, 512, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[1], activation=True, discriminator=False),
            conv_block( 512,  256, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[2], activation=True, discriminator=False),
            conv_block( 256,  128, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[3], activation=True, discriminator=False),
            nn.ConvTranspose2d(128,img_chs,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh() # We want outputs to be between [-1,1]
        )
        init_weights(self.model)
        return


    def forward(self,x):
        if x.size()[-2:] != (1,1):
            raise Exception('X size must be (B,Z,1,1).')
        return self.model(x)
