
from net.model import AirNet

import torch
from torch import nn

from option import options as opt

degrad_patch1 = torch.randn(3,3,224,224)

degrad_patch2 = torch.randn(4,3,224,224)
net = AirNet(opt)
restored = net(x_query=degrad_patch1, x_key=degrad_patch2)
