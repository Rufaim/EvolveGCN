import tensorflow as tf
import numpy as np
import scipy.sparse as sp



def convert_scipy_CRS_space_to_tensor(sparce_mat: sp.csr.csr_matrix):
    sparce_mat_coo = sparce_mat.tocoo()
    indices = np.transpose(np.array([sparce_mat_coo.row, sparce_mat_coo.col]))
    return tf.SparseTensor(indices, sparce_mat_coo.data, sparce_mat_coo.shape)


def normalize_adjencency_mat(adj_mat: sp.csr.csr_matrix):
    assert len(adj_mat.shape) == 2

    d = np.array(adj_mat.sum(axis=-1))[...,0]
    d = np.sqrt(d)
    d = sp.diags(d).tocsr()
    a = adj_mat + sp.eye(adj_mat.shape[-1],dtype=adj_mat.dtype)
    return d @ a @ d
