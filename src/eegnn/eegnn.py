import torch 
import torch.nn as nn
import torch.nn.functional as F
from .layers import EELayer, MLP
from .functions.sine import Sine

class EEGNN(nn.Module):
    def __init__(
        self,
        d_in_nfeats,
        d_in_efeats,
        d_h_efeats,
        d_h_nfeats,
        n_layers,
        batch_norm=True,
    ):
        super(EEGNN, self).__init__()
        self.activation = Sine()

        # Edge-Enhanced Layers
        self.layers = nn.ModuleList()
        self.layers.append(
            EELayer(d_in_nfeats, d_in_efeats, d_h_nfeats, d_h_efeats, batch_norm=batch_norm)
        )
        for _ in range(n_layers-1):
            self.layers.append(
                EELayer(d_h_nfeats, d_h_efeats, d_h_nfeats, d_h_efeats, batch_norm=batch_norm)
            )
        self.MLP_n = MLP(num_layers=2, input_dim=d_h_nfeats, hidden_dim=d_h_nfeats*2, output_dim=d_h_nfeats,mid_activation=Sine(), mid_batch_norm=False, bn_affine=False)
        self.MLP_e = MLP(num_layers=2, input_dim=d_h_efeats, hidden_dim=d_h_efeats*2, output_dim=d_h_efeats, mid_activation=Sine(), mid_batch_norm=False, bn_affine=False)
        

    def forward(self, g, feats):
        nfeats, efeats = feats
        # node updating & edge updating
        for i, layer in enumerate(self.layers):
            nfeats, efeats = layer(g, nfeats, efeats)
        nfeats, efeats = self.MLP_n(nfeats), self.MLP_e(efeats)
        return nfeats, efeats