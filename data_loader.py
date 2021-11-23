import numpy as np
import pandas as pd
import os
import networkx as nx

from utils import normalize_adjencency_mat, convert_scipy_CRS_space_to_tensor



class EllipticDatasetLoader(object):
    def __init__(self, datadir_path, test_portion=0.3, filter_unknown=False, local_features_only=True):
        self.filter_unknown = filter_unknown

        classes_csv = os.path.join(datadir_path, 'elliptic_txs_classes.csv')
        edgelist_csv = os.path.join(datadir_path, 'elliptic_txs_edgelist.csv')
        features_csv = os.path.join(datadir_path, 'elliptic_txs_features.csv')

        classes = pd.read_csv(classes_csv, index_col='txId')  # labels are 'unknown', '1'(illicit), '2'
        edgelist = pd.read_csv(edgelist_csv)
        features = pd.read_csv(features_csv, header=None, index_col=0)  # features of the transactions
        data = pd.concat([classes,features],axis=1)

        num_features = features.shape[1]
        timesteps = np.unique(features[1])

        graph = nx.DiGraph()
        feature_idx = [i+2 for i in range(93+72)]
        if local_features_only:
            feature_idx = feature_idx[:94]
        for tx_idx, features in data.iterrows():
            graph.add_node(tx_idx,label=str(features["class"]),timestamp=features[1],features=features[feature_idx])
        graph.add_edges_from(edgelist.values)

        train_timesteps, test_timesteps = np.split(timesteps,[int(timesteps.shape[0]*(1-test_portion))])
        self.train_graphs = []
        self.train_triples = []
        self.test_graphs = []
        self.test_triples = []
        one_hot = np.eye(2,dtype=np.float32) if self.filter_unknown else np.eye(3,dtype=np.float32)
        class_converter = {"1":0, "2":1, "unknown":2}
        for comp in nx.weakly_connected_components(graph):
            sg = graph.subgraph(comp)
            if self.filter_unknown:
                sg.remove_nodes_from(classes[classes["class"] == "unknown"].index.tolist())

            ts = np.unique([d for _,d in sg.nodes(data="timestamp")])
            if ts.shape[0] > 1:
                raise RuntimeError("incorrect division on timestamps")

            nodes, targets = [], []
            for _, d in sg.nodes(data=True):
                targets.append(class_converter[d["label"]])
                nodes.append(d["features"].values.astype(np.float32))
            nodes = np.vstack(nodes)
            targets = one_hot[np.array(targets)]
            adjacency_mat = normalize_adjencency_mat(nx.adjacency_matrix(sg).astype(np.float32))
            adjacency_mat = convert_scipy_CRS_space_to_tensor(adjacency_mat)

            if ts[0] in train_timesteps:
                self.train_graphs.append(sg)
                self.train_triples.append((nodes,targets, adjacency_mat))
            else:
                self.test_graphs.append(sg)
                self.test_triples.append((nodes, targets, adjacency_mat))

        if len(self.train_graphs) + len(self.test_graphs) != timesteps.shape[0]:
            raise RuntimeError("number of generated graphs goes not match number of timestamps")

    @property
    def num_classes(self):
        return 3

    def test_batch_iterator(self):
        for i in range(len(self.test_graphs)):
            g = self.test_graphs[i]
            n, t, adj = self.test_triples[i]
            yield g, n, t, adj

    def train_batch_iterator(self):
        idx = np.arange(len(self.train_graphs))
        np.random.shuffle(idx)
        for i in idx:
            g = self.train_graphs[i]
            n, t, adj = self.train_triples[i]
            yield g, n, t, adj
