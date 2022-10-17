import torch.nn as nn


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


import torch
input = torch.Tensor(4, 1, 28, 28)
model = ProtoNet()
output = model(input)
# print(output.shape)
# print('====')

x_dim = 1
hid_dim = 64
z_dim = 64

ProtoNetPC = nn.Sequential(
    conv_block(x_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, z_dim),

    # flatten
    nn.Flatten()

)

# input = torch.Tensor(4, 1, 28, 28)
# model = ProtoNetPC
# output = model(input)
# print(output.shape)

# class ProtoNetPC(nn.Module):
#     '''
#     Model as described in the reference paper,
#     source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
#     '''
#     def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
#         super(ProtoNetPC, self).__init__()
#         self.encoder = nn.Sequential(
#             conv_block(x_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, z_dim),
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x.view(x.size(0), -1)
