import torch
import torch.nn as nn
from utils.layers import conv_block
from utils.weights import init_weights

class Discriminator(nn.Module):
    def __init__(self,
                 img_chs,
                 img_h,
                 img_w,
                 norm_layer_output=[False,True,True,True,],
                 features=64,
                 weights_file=None):
        super(Discriminator, self).__init__()
        self.img_h = img_h
        self.img_w = img_w

        self.model = nn.Sequential(
            conv_block( img_chs,    features,   kernel_size=4, stride=2, padding=1, norm=norm_layer_output[0], activation=True, discriminator=True),
            conv_block( features,   features*2, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[1], activation=True, discriminator=True),
            conv_block( features*2, features*4, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[2], activation=True, discriminator=True),
            conv_block( features*4, features*8, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[3], activation=True, discriminator=True),
            nn.Conv2d(  features*8, 1,          kernel_size=4, stride=2, padding=0, bias=True),
        )
        if weights_file == None:
            init_weights(self.model)
        else:
            print(f'Loading discriminator model weights from {weights_file}')
            pretrained_dict = torch.load(weights_file)
            print(pretrained_dict.keys())
            pretrained_dict = {key.replace("model.", ""): value for key, value in pretrained_dict.items()}
            print(pretrained_dict.keys())
            self.model.load_state_dict(pretrained_dict)
        return

    def forward(self,x):
        if x.size()[-2:] != (self.img_h,self.img_w):
            raise Exception(f'X size must be (B,img_chs,{self.img_h},{self.img_w}).')
        return self.model(x)
