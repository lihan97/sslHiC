import os
import numpy as np
from numpy import linalg
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import torch
import statsmodels.stats.multitest as smm
from .eegnn.eegnn import EEGNN
from src.config import config_dict
from src.data.utils import prepare_graphs
def cosine_similarity(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta
def load_model(model_path, config):
    model = EEGNN(
        d_in_nfeats=config['d_in_nfeats'], d_in_efeats=config['d_in_efeats'], d_h_efeats=config['d_h_efeats'], d_h_nfeats=config['d_h_nfeats'], n_layers=config['n_layers'], batch_norm=config['batch_norm']
    )
    state_dict = torch.load(f"{os.path.split(os.path.realpath(__file__))[0]}/{model_path}")
    model.load_state_dict(state_dict)
    model.eval()
    return model
def get_reproducibility_score(m1, m2, resol, complete=True):
    model_config = config_dict['rep'][resol]
    model_path = model_config['model_path']
    model = load_model(model_path, model_config)
    g1, g2  = prepare_graphs(
        m1, m2, complete=complete
    )
    g1_feats, g2_feats = (g1.ndata['hv'],g1.edata['he']),(g2.ndata['hv'],g2.edata['he'])
    g1_embeds, _ = model(g1, g1_feats)
    g2_embeds, _ = model(g2, g2_feats)

    g1_embeds = g1_embeds.detach().numpy()
    g2_embeds = g2_embeds.detach().numpy()

    coefs = []
    g1_in_degrees = g1.in_degrees().numpy()
    g2_in_degrees = g2.in_degrees().numpy()
    for i, (e1, e2) in enumerate(zip(g1_embeds, g2_embeds)):
        coefs.append(cosine_similarity(e1,e2)*(g1_in_degrees[i]+g2_in_degrees[i])/(np.sum(g1_in_degrees)+np.sum(g2_in_degrees)))
    
    return np.sum(coefs)

def get_DCIs(m1, m2, resol, filter_zero_interactions=False):
    """
    500kb: sigma=0.8, 50kb: sigma=1.0, 10kb: sigma=1.0
    """
    model_config = config_dict['dci'][resol]
    model_path = model_config['model_path']
    sigma = model_config['sigma']
    model = load_model(model_path, model_config)
    
    g1, g2 = prepare_graphs(
        m1, m2, complete=True
    )
    g1_feats, g2_feats = (g1.ndata['hv'],g1.edata['he']),(g2.ndata['hv'],g2.edata['he'])
    _, g1_embeds = model(g1, g1_feats)
    _, g2_embeds = model(g2, g2_feats)

    g1_embeds = g1_embeds.detach().numpy()
    g2_embeds = g2_embeds.detach().numpy()

    n_nodes, d_h = g1.number_of_nodes(), g1_embeds.shape[-1]
    g1_embeds = g1_embeds.reshape(n_nodes,n_nodes,d_h)
    g2_embeds = g2_embeds.reshape(n_nodes,n_nodes,d_h)
    for i in range(g1_embeds.shape[-1]):
        g1_embeds[:,:,i] = gaussian_filter(g1_embeds[:,:,i],sigma=sigma)
        g2_embeds[:,:,i] = gaussian_filter(g2_embeds[:,:,i], sigma=sigma)
    g1_embeds = g1_embeds.reshape(-1,d_h)
    g2_embeds = g2_embeds.reshape(-1,d_h)

    nonzero_ids = np.logical_or(m1.toarray().reshape(-1)!=0, m2.toarray().reshape(-1)!=0)
    # p_values
    dif_scores = []
    if filter_zero_interactions:
        for emb1, emb2 in zip(g1_embeds[nonzero_ids], g2_embeds[nonzero_ids]):
            dif_scores.append(linalg.norm(emb1-emb2,ord=2))
    else:
        for emb1, emb2 in zip(g1_embeds, g2_embeds):
            dif_scores.append(linalg.norm(emb1-emb2,ord=2))

    params = norm.fit(dif_scores)
    p_vals = norm.cdf(
        dif_scores,
        loc=params[0],
        scale=params[1]
    )
    np.nan_to_num(p_vals, copy=False, posinf=1, neginf=1, nan=1)
    p_vals = 1-p_vals
    _, corrected_p = smm.multipletests(p_vals, method='fdr_bh')[:2]
    
    if filter_zero_interactions:
        final_p = np.ones(n_nodes**2)
        final_p[nonzero_ids] = corrected_p
    else: 
        final_p = corrected_p
    return 1-final_p.reshape(n_nodes, n_nodes)



