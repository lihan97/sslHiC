import cooler
import numpy as np
from scipy import sparse as sps
import dgl
import torch
from scipy.io import mmread

def featurize_graph(g): 
    g.edata['dist'] = (torch.abs((g.edges()[0] - g.edges()[1])).to(torch.float)/g.number_of_nodes())
    g.edata['he'] = torch.stack([
        g.edata['freq'],
        g.edata['dist']
    ],dim=-1)
    g.edata['freq'] = g.edata['freq']
    g.edata['dist'] = g.edata['dist']
    g.ndata['hv'] = (g.nodes()/g.number_of_nodes()).unsqueeze(-1)
    return g

def log10_min_max_norm(up_mat):
    log_m = np.log10(1+up_mat.todense())
    sm = (log_m - 0) / (np.max(log_m)-0)
    return sps.coo_matrix(sm)

def mat_to_graph(sps_mat, complete=True):
    sps_up_mat = sps.triu(sps_mat, k=0)
    sorted_indexs = np.argsort(-sps_up_mat.data)
    size = len(sps_up_mat.data)
    cutoff = int(size/2 * 0.001)
    sps_up_mat.data[sorted_indexs[:cutoff]] = sps_up_mat.data[sorted_indexs[cutoff]]
    sps_up_mat = log10_min_max_norm(sps_up_mat)
    if complete:
        sps_up_mat = sps.coo_matrix(sps_up_mat.toarray().astype(np.float32) + 1e-6, dtype=np.float32)
    sps_down_mat = sps.triu(sps_up_mat,1).transpose()
    sps_mat = sps_up_mat + sps_down_mat
    g = dgl.from_scipy(sps_mat, eweight_name='freq')
    g.edata['freq'] = g.edata['freq'].to(torch.float32)
    return g

def cool_to_mats(path, chroms):
    c = cooler.Cooler(path)
    mats = []
    for chrom in chroms:
        mat = c.matrix(balance=False,sparse=True).fetch(chrom)
        mat.data[(np.isnan(mat.data))] = 0
        mats.append(mat)
    return mats
def mtx_to_mat(path):
    return sps.coo_matrix(mmread(path))
def npz_to_mat(path, key):
    return np.load(path,allow_pickle=True)[key]
def npy_to_mat(path):
    return np.load(path,allow_pickle=True)
def cool_to_graphs(path, chroms,complete=True):
    c = cooler.Cooler(path)
    graphs = []
    for chrom in chroms:
        mat = c.matrix(balance=False, sparse=True).fetch(chrom)
        g = mat_to_graph(mat, complete=complete)
        assert g.number_of_nodes() == mat.shape[0]
        graphs.append(g)
    return graphs

def setdiff2d(a,b):
    a1_rows = a.view([('', a.dtype)] * a.shape[1])
    a2_rows = b.view([('', b.dtype)] * b.shape[1])
    c = np.setdiff1d(a1_rows, a2_rows).view(a.dtype).reshape(-1, a.shape[1])
    return c

def intersect2d(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def align_mats(mat1, mat2):
    mat1 = mat1.astype(np.float32)
    mat2 = mat2.astype(np.float32)
    assert mat1.shape == mat2.shape
    e1 = np.stack(np.where(mat1 != 0), axis=1)
    e2 = np.stack(np.where(mat2 != 0), axis=1)
    inter = intersect2d(e1, e2)
    g2_new_edges = setdiff2d(e1, inter)
    
    g1_new_edges = setdiff2d(e2, inter)
    if len(g2_new_edges) >= 0:
        mat2[g2_new_edges[:,0],g2_new_edges[:,1]] = 1e-6
    if len(g1_new_edges) >= 0:
        mat1[g1_new_edges[:,0],g1_new_edges[:,1]] = 1e-6
    return mat1, mat2

def prepare_graphs(mat1, mat2, complete=True):
    mat1_dense = np.array(mat1.todense(),dtype=np.float32)
    mat2_dense = np.array(mat2.todense(),dtype=np.float32)
    mat1_dense = mat1_dense*(np.sum(mat2_dense)/np.sum(mat1_dense))
    if not complete:
        mat1_dense, mat2_dense = align_mats(mat1_dense, mat2_dense)
    g1 = mat_to_graph(mat1_dense, complete=complete)
    g2 = mat_to_graph(mat2_dense, complete=complete)    
    g1 = featurize_graph(g1)
    g2 = featurize_graph(g2)

    return g1, g2