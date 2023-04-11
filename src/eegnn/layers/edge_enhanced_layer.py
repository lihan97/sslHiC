import sys
sys.path.append('..')
import torch
from torch import nn
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from .mlp import MLP
from src.eegnn.functions.agg_ops import weighted_mean, weighted_std, weighted_max, weighted_min
from src.eegnn.functions.sine import Sine

class EELayer(nn.Module):
    def __init__(self, d_in_nfeats, d_in_efeats, d_out_nfeats, d_out_efeats, batch_norm):
        super(EELayer, self).__init__()
        self.d_out_nfeats = d_out_nfeats
        self.d_out_efeats = d_out_efeats
        self.batch_norm = batch_norm
        self.activation = Sine()

        self.forwtrans_edge_e = MLP(num_layers=2, input_dim=d_in_efeats, hidden_dim=d_out_efeats, output_dim=d_out_efeats, mid_activation=self.activation, mid_batch_norm=batch_norm)
        self.forwtrans_node_e = MLP(num_layers=2, input_dim=d_out_nfeats, hidden_dim=d_out_nfeats, output_dim=d_out_nfeats, mid_activation=self.activation, mid_batch_norm=batch_norm)
        self.posttrans_edge = MLP(num_layers=2, input_dim=d_out_efeats+d_out_nfeats, hidden_dim=d_out_efeats, output_dim=d_out_efeats, mid_activation=self.activation,mid_batch_norm=batch_norm)
        
        self.forwtrans_edge_n = MLP(num_layers=2, input_dim=d_in_efeats, hidden_dim=d_out_efeats, output_dim=d_out_efeats, mid_activation=self.activation, mid_batch_norm=batch_norm)
        self.forwtrans_node_n = MLP(num_layers=2, input_dim=d_in_nfeats, hidden_dim=d_out_nfeats, output_dim=d_out_nfeats, mid_activation=self.activation, mid_batch_norm=batch_norm)
        self.posttrans_node = MLP(num_layers=2, input_dim=4*d_out_efeats+d_out_nfeats,hidden_dim=d_out_nfeats, output_dim=d_out_nfeats, mid_activation=self.activation, mid_batch_norm=batch_norm)
        
        if self.batch_norm:
            self.batch_norm_edge = nn.BatchNorm1d(d_out_efeats)
            self.batch_norm_node = nn.BatchNorm1d(d_out_nfeats)

    def reduce_func(self, nodes):
        h = nodes.mailbox['he']
        a = h[:,:,-1].reshape((h.shape[0], h.shape[1],1))
        h = h[:,:,:-1].reshape((h.shape[0], h.shape[1],h.shape[2]-1))
        h = torch.cat([weighted_mean(h,a), weighted_min(h,a), weighted_max(h,a), weighted_std(h,a)], dim=-1)
        return {'hv_update': h}

    def apply_posttrans_edge(self, edges):
        h = torch.cat(
            [edges.dst['hv']+edges.src['hv'], edges.data['he']], dim=-1)
        h = self.posttrans_edge(h)
        if self.batch_norm:
            h = self.batch_norm_edge(h)
        h = self.activation(h)
        return {'he': h}

    def update_edges(self, g, node_feats, edge_feats):
        g = g.local_var()
        # forward transformation
        g.ndata['hv'] = self.forwtrans_node_e(node_feats)
        g.edata['he'] = self.forwtrans_edge_e(edge_feats)
        # updating
        g.apply_edges(self.apply_posttrans_edge)
        return g.edata['he']
    
    def update_nodes(self, g, node_feats, edge_feats):
        g = g.local_var()
        # forward transformation
        g.ndata['hv'] = self.forwtrans_node_n(node_feats)
        g.edata['he'] = self.forwtrans_edge_n(edge_feats)
        # information aggregation
        g.srcdata.update({'ft': g.ndata['hv']})
        g.dstdata.update({'ft': g.ndata['hv']})
        g.apply_edges(fn.u_dot_v('ft', 'ft', 'logit'))
        a = edge_softmax(g, g.edata['logit']) / self.d_out_efeats**0.5
        g.edata['he'] = torch.cat([g.edata['he'], a], dim=-1)
        g.update_all(fn.copy_e('he','he'), self.reduce_func)
        # post transformation
        h = torch.cat([g.ndata['hv_update'], g.ndata['hv']],dim=-1)
        h = self.posttrans_node(h)
        if self.batch_norm:
            h = self.batch_norm_node(h)
        h = self.activation(h)
        return h
        
    def forward(self, g, node_feats, edge_feats):
        # node updating
        node_feats = self.update_nodes(g, node_feats, edge_feats)
        # edge updating
        edge_feats = self.update_edges(g, node_feats, edge_feats)
        return node_feats, edge_feats