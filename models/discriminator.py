import torch.nn as nn
from utils.layers import conv_block
from utils.weights import init_weights

class Discriminator(nn.Module):
    def __init__(self,
                 img_chs,
                 img_h,
                 img_w,
                 norm_layer_output=[True,True,True,True,]):
        super(Discriminator, self).__init__()
        self.img_h = img_h
        self.img_w = img_w

        self.model = nn.Sequential(
            conv_block(img_chs, 128, kernel_size=4, stride=2, padding=0, norm=norm_layer_output[0], activation=True, discriminator=True),
            conv_block(    128, 256, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[1], activation=True, discriminator=True),
            conv_block(    256, 512, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[2], activation=True, discriminator=True),
            conv_block(    512,1024, kernel_size=4, stride=2, padding=1, norm=norm_layer_output[3], activation=True, discriminator=True),
            nn.Conv2d(    1024,   1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid() # We want outputs to be between [0,1]
        )
        init_weights(self.model)
        return

    def forward(self,x):
        if x.size()[-2:] != (self.img_h,self.img_w):
            raise Exception(f'X size must be (B,Z,{self.img_h},{self.img_w}).')
        return self.model(x)
