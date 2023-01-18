import torch.nn as nn

def conv_block(in_ch,out_ch,kernel_size,stride,padding,norm,activation,discriminator):
    layer = nn.Sequential()
    if discriminator:
        layer.append(nn.Conv2d(in_ch, out_ch, kernel_size,stride,padding,bias=False))
    else: # Generator
        layer.append(nn.ConvTranspose2d(in_ch,out_ch,kernel_size,stride,padding,bias=False))
    if norm:
        layer.append(nn.BatchNorm2d(out_ch))
    if activation:
        layer.append(nn.LeakyReLU(0.2))
    return layer
