from torch import nn

from net import fre
from net.encoder import CBDE
# from net.DGRN import DGRN
from net.restoration import PromptIR


class AirNet(nn.Module):
    def __init__(self, opt):
        super(AirNet, self).__init__()

        # Restorer
        self.R = PromptIR()

        # Encoder
        self.E = CBDE(opt)

    def forward(self, x_query, x_key):
        if self.training:
            fea, logits, labels = self.E(x_query, x_key)

            restored = self.R(x_query, fea)

            return restored, logits, labels
        else:
            fea = self.E(x_query, x_query)

            restored = self.R(x_query, fea)

            return restored
