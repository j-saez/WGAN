import torch.nn as nn

"""
    Name:        init_weights 
    Description: Initialize model's weights following a normal distribution mean 0.0 and standard deviation of 0.02.
    Inputs:
                 >> model: Pytorch model
    Outputs: None
"""
def init_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
    return
