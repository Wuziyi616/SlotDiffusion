import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips


class VQLPIPSLoss(nn.Module):

    def __init__(self, percept_loss_w=1.0):
        super().__init__()

        self.perceptual_weight = percept_loss_w
        if self.perceptual_weight > 0.:
            self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
            for p in self.perceptual_loss.parameters():
                p.requires_grad = False

    def forward(
        self,
        quant_loss,
        x,
        recon,
    ):
        x = x.contiguous()
        recon = recon.contiguous()
        # L1 loss + LPIPS
        if self.perceptual_weight > 0:
            recon_loss = torch.abs(x - recon).mean()
        # simple L2 loss
        else:
            recon_loss = F.mse_loss(x, recon)
        if self.perceptual_weight > 0:
            if len(x.shape) == 5:
                x = x.flatten(0, 1)
                recon = recon.flatten(0, 1)
            percept_loss = self.perceptual_loss(x, recon).mean()
        else:
            percept_loss = torch.tensor(0.).type_as(quant_loss)

        loss_dict = {
            'quant_loss': quant_loss,
            'recon_loss': recon_loss,
            'percept_loss': percept_loss,
        }
        return loss_dict
