import datetime
import math
import os
import random
from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse


class Graph(object):

    def __init__(self):
        self.nx_graph = nx.Graph()
        self.katz = None
        self.embeddings = None
        self.emb_cols = []
        self.min_date = self.max_date = None
        self.walks = None
        self.dates = False

    def load_embeddings(self, f_name, dims=128):
        """
        Load embeddings from 'f_name'.  If embedding exists for a node, update it with the new embedding.
        """
        emb_df = pd.read_csv(f_name, sep=' ', skiprows=1, header=None, index_col=None)
        if not self.embeddings:
            self.embeddings = {}
        for i in range(0, emb_df.shape[0]):
            key = emb_df.iloc[i, 0]
            if str(key) in '</s>':
                continue
            emb = np.array(emb_df.iloc[i, 1: dims + 1].tolist())
            emb = emb.astype(float)
            self.embeddings[int(key)] = emb
        self.make_emb_cols(dims)

    def save_embeddings(self, f_name, dim):
        """
        Save current embeddings
        """
        out_file = open(f_name, 'w')
        if not self.embeddings:
            print("No embeddings to save!")
            return
        out_file.write("%s %s%s" % (len(self.embeddings), dim, os.linesep))
        for node, embedding in self.embeddings.items():
            out_file.write("%s " % node)
            d = 0
            for x in embedding:
                out_file.write("%s " % str(x))
                d += 1
                if d < dim:
                    out_file.write(" ")
            out_file.write(os.linesep)
        out_file.flush()
        out_file.close()

    def make_emb_cols(self, dims):
        """
        Generate column names for embedding dimensions
        """
        self.emb_cols = []
        for j in range(1, dims + 1):
            self.emb_cols.append('dw' + str(j - 1))

    def generate_embeddings_with_prev(self, old_emb, dims):
        """
        Given previous embeddings determine reasonable embeddings for nodes that are incident to an edge, but don't
        already have an embedding.
        """
        self.embeddings = old_emb
        for node in self.nx_graph.nodes_iter():
            if self.nx_graph.degree(node) == 0:
                continue
            if node not in self.embeddings:
                nbr_vecs = []
                for nbr in self.nx_graph.neighbors(node):
                    if nbr in self.embeddings:
                        nbr_vecs.append(self.embeddings[nbr])

                if len(nbr_vecs):
                    self.embeddings[node] = np.mean(nbr_vecs, axis=0)
                else:
                    self.embeddings[node] = self._rand_vec(dims)

    def save_betas(self, f_name, degree_graph=None):
        """
        Save per-node beta file.  Use degree_graph to choose smart betas.
        """
        beta_file = open(f_name, 'w')
        n_betas = 0
        for node in self.nx_graph.nodes_iter():
            if self.nx_graph.degree(node) > 0:
                n_betas += 1
        beta_file.write("%s%s" % (n_betas, os.linesep))
        for node in self.nx_graph.nodes_iter():
            if self.nx_graph.degree(node) > 0:
                beta = .8
                # if degree_graph:
                #    beta = .8 - (.7 / degree_graph.degree(node))
                beta_file.write("%s %f%s" % (node, beta, os.linesep))
        beta_file.flush()
        beta_file.close()

    @staticmethod
    def _rand_vec(dims):
        return [(random.random() * 2 - 1) for _ in range(0, dims)]

    def save_edgelist(self, f_name):
        el = open(f_name, 'w')
        for u, v, data in self.nx_graph.edges_iter(data=True):
            el.write("%s %s %s %s" % (u, v, data['weight'], os.linesep))
            el.write("%s %s %s %s" % (v, u, data['weight'], os.linesep))
        el.flush()
        el.close()

    def save_walks(self, f_name):
        """
        Save self.walks if it exists
        """
        # todo: delete this method
        w_file = open(f_name, 'w')
        if not self.walks:
            print("No walks exist.\nYou must run perform_walks() first!")
            return
        for walk in self.walks:
            w_len = len(walk)
            for i, node in enumerate(walk):
                w_file.write("%s" % node)
                if i < (w_len - 1):
                    w_file.write(" ")
            w_file.write("%s" % os.linesep)
        w_file.flush()
        w_file.close()

    @classmethod
    def graph_with_graph(cls, graph):
        """
        Create a copy of 'graph'
        """
        new = cls()
        new.nx_graph = graph.nx_graph.copy()
        new.max_date = graph.max_date
        new.min_date = graph.min_date
        return new

    def subgraph_within_dates(self, start_date, end_date):
        """
        Get a subgraph of this graph by removing edges outside of time interval
        """
        new = Graph.graph_with_graph(self)
        for u, v in self.nx_graph.edges_iter():
            edge = new.nx_graph.edge[u][v]
            done = False
            while len(edge['timestamps']) and not done:
                top = bottom = False
                if edge['timestamps'][0] >= end_date:
                    edge['timestamps'].pop(0)
                else:
                    top = True

                if len(edge['timestamps']) and edge['timestamps'][-1] < start_date:
                    edge['timestamps'].pop(-1)
                else:
                    bottom = True

                done = top and bottom

            if not len(edge['timestamps']):
                new.nx_graph.remove_edge(u, v)

        # min = 100000000
        # max = -100000000
        # for u,v, data in new.nx_graph.edges_iter(data=True):
        #     for time in data['timestamps']:
        #         if time < min:
        #             min = time
        #         elif time > max:
        #             max = time

        new.max_date = end_date
        new.min_date = start_date
        return new

    def subgraphs_of_length(self, days=None, periods=None):
        """
        Get all temporal subgraphs of 'days' length
        Use 'periods' when graph pre split into numbered time periods
        """
        graphs = []
        if days:
            sg_length = datetime.timedelta(days=days)
        else:
            sg_length = periods

        start_date = self.min_date
        end_date = start_date + sg_length
        done = False
        while not done:
            if start_date > self.max_date:
                break
            if end_date > self.max_date:
                # end_date = self.max_date
                done = True
            print(start_date, end_date)
            new = self.subgraph_within_dates(start_date, end_date)
            if new.nx_graph.number_of_edges():
                graphs.append(new)
            start_date += sg_length
            end_date += sg_length
        return graphs

    def katz_for_pairs(self, pairs, adj_prefix, max_length=6, beta=0.1):
        """
        Calculate katz using precomputed adjacency matrices.
        <adj_prefix>2.npz is A^2
        """
        filenames = [adj_prefix + str(i) + '.npz' for i in range(2, max_length + 1)]
        n = 1
        self.katz = {}
        bs = [1.500, .891, .631, .543, .413, .420] # output from Mohler method
        for u, v in pairs:
            if u not in self.katz:
                self.katz[u] = defaultdict(float)
        for f in filenames:
            a = scipy.sparse.load_npz(f)
            # b = beta ** n
            b = bs[n-1]
            for u, v in pairs:
                self.katz[u][v] += b * a[u-1, v-1]
            n += 1
            print("Loaded %s" % f)

    def all_pairs(self):
        """
        Get all the pairs of nodes for the graph

        :return: iterator containing pairs of nodes
        """
        return chain(self.nx_graph.edges(), nx.non_edges(self.nx_graph))

    def get_sp(self, u, v):
        """
        Get the shortest path length between u and v (do not include edge between nodes if it exists)

        :param u: The id of the source node
        :param v: The id of the target node
        :return: The length of the shortest path
        """
        ed = None

        if self.nx_graph.has_edge(u, v):
            ed = self.nx_graph.edge[u][v]
            self.nx_graph.remove_edge(u, v)

        try:
            distance = nx.shortest_path_length(self.nx_graph, u, v)
        except:
            distance = 1000000

        if distance == 0:  # I check to prevent a user from mentioning him- or herself.
            distance = 1000000  # It seems to work except for one time... I might track it down later

        if ed:
            self.nx_graph.add_edge(u, v, ed)

        return distance

    def common_nbrs(self, u, v):
        """
        Find common neighbors of u, v

        :param u: A node on the graph
        :param v: Another node on the graph
        :return: List of common neighbors, Adjacency list of U, Adjacency list of V
        """
        u_adj = self.nx_graph.neighbors(u)
        v_adj = self.nx_graph.neighbors(v)
        nbrs = []
        for u in u_adj:
            if u in v_adj:
                nbrs.append(u)

        return nbrs, u_adj, v_adj

    def get_unsupported(self, u, v):
        """
        Get the features that are unsupported by directed graphs.

        :param u: The source node
        :param v: The target node
        :return: A 4-tuple (Jaccard, Adamic/Adar, Number of common neighbors, preferential attachment)
        """
        jac = adam = 0
        nbrs, u_adj, v_adj = self.common_nbrs(u, v)
        n_nbrs = len(nbrs)
        union_magn = len(u_adj) + len(v_adj)
        if union_magn:
            jac = float(n_nbrs) / float(union_magn)

        for nbr in nbrs:
            deg = self.nx_graph.degree(nbr)
            if deg > 1:
                adam += 1 / math.log(deg)

        attachment = len(u_adj) * len(v_adj)
        return jac, adam, n_nbrs, attachment

    def make_pairs_with_edges(self, label_graph, target_positive_ratio=.5, enforce_non_edge=True, enforce_has_embeddings=False):
        """
        Generate a dataframe with a fixed ratio of positives to negatives by requiring all new edges in
        label_graph to appear in the dataframe.

        :param label_graph: The graph to check for new edges
        :param target_positive_ratio: Ratio of positive to negative (default=.5)
        :return: A list of tuples containing target_positive_ratio edges to non-edges
        """

        pairs = []
        pairs_dict = defaultdict(bool)
        edges = 0

        if target_positive_ratio == 0:
            # We want all the pairs from label_graph
            # todo: do we need pairs_dict for this part
            for u, v in label_graph.nx_graph.edges_iter():
                if enforce_has_embeddings:
                    if u not in self.embeddings or v not in self.embeddings:
                        continue
                edges += 1
                pairs.append((u, v))
            for u, v in nx.non_edges(label_graph.nx_graph):
                if enforce_has_embeddings:
                    if u not in self.embeddings or v not in self.embeddings:
                        continue
                pairs.append((u, v))
            print("\t%d edges out of %d pairs" % (edges, len(pairs)))
            return pairs

        for u, v in label_graph.nx_graph.edges_iter():
            if enforce_has_embeddings and not self.embeddings:
                print("No embeddings found! Error!")
                return
            if enforce_has_embeddings:
                if u not in self.embeddings or v not in self.embeddings:
                    continue
            if (enforce_non_edge and not self.nx_graph.has_edge(u, v)) or not enforce_non_edge:
                u, v = sorted((u, v))
                if not pairs_dict[(u, v)]:
                    pairs_dict[(u, v)] = True
                    pairs.append((u, v))
                    edges += 1

        nodes = self.embeddings.keys()
        added = 0
        rejected = 0
        while float(edges) / len(pairs) > target_positive_ratio:
            u = nodes[int(random.random() * len(nodes))]
            v = nodes[int(random.random() * len(nodes))]
            if label_graph.nx_graph.has_edge(u, v) or u == v:
                rejected += 1
                continue
            if enforce_has_embeddings:
                if u not in self.embeddings or v not in self.embeddings:
                    rejected += 1
                    continue
            (u, v) = sorted((u, v))
            if not pairs_dict[(u, v)]:
                pairs_dict[(u, v)] = True
                pairs.append((u, v))
                added += 1
        return pairs

    def to_dataframe(self, pairs=False, sampling=None, label_graph=None, cheat=False, allow_hashtags=False, min_katz=0, verbose=True, katz=None):
        """
        Get a dataframe for pairs of nodes in the graph

        :param pairs: True to consider all pairs, False to consider only non-edges, or a list of tuples to use as pairs
        :param sampling: Amount to sample (default=None, do not sample)
        :param label_graph: Graph to use to generate the true labels.  Usually the next in the time series.
        :param cheat: Do not sample when label_graph has an edge for a given pair (default=False)
        :param allow_hashtags: Also predict links between users and hashtags (default=False)
        :param min_katz: Use a katz threshold to reduce numbers of pairs
        :param verbose: Display updates (default=True)
        :param katz: Precomputed katz centrality dictionary (default=None, compute katz before generating dataframe)
        :return: A pandas dataframe containing pairs and the various calculated metrics
        """
        if not sampling:
            sampling = 2

        u = []
        v = []
        has_links = []
        jac_co = []
        adam = []
        att = []
        nbrs = []
        spl = []
        katz_centralities = []
        count = 0
        labels = []
        katzes = []
        embeddings = []
        if self.embeddings:
            for _ in self.emb_cols:
                embeddings.append([])
        # degree = nx.degree(graph)

        if type(pairs) is bool and pairs:
            iter_set = self.all_pairs()
        elif type(pairs) is bool and not pairs:
            iter_set = nx.non_edges(self.nx_graph)
        else:
            iter_set = pairs

        if verbose and not katz:
            print("Precomputing katzes....")

        if not katz:
            katz = nx.katz_centrality(self.nx_graph, alpha=.005, beta=.1, tol=.00000001, max_iter=5000)

        elim = 0
        for n1, n2 in iter_set:
            if random.random() < sampling or (cheat and label_graph and label_graph.nx_graph.has_edge(n1, n2)):
                count += 1
                if verbose:
                   if count % 10000 == 0:
                       print("%d checked... " % count)
                # k_s = np.mean((katz[n1], katz[n2]))
                #if k_s < min_katz:
                #    elim += 1
                #    continue
                u.append(n1)
                v.append(n2)
                # (jaccard, adamic, n_nbrs, attachment) = self.get_unsupported(n1, n2)
                # jac_co.append(jaccard)
                # adam.append(adamic)
                # nbrs.append(n_nbrs)
                # att.append(attachment)
                # spl.append(self.get_sp(n1, n2))
                # katz_centralities.append(np.mean((katz[n1], katz[n2])))
                labels.append(label_graph.nx_graph.has_edge(n1, n2))
                #if self.katz:
                #    katzes.append(self.katz[n1][n2])
                if self.embeddings:
                    for i in range(0, len(self.emb_cols)):
                       embeddings[i].append(np.mean((self.embeddings[n1][i], self.embeddings[n2][i])))
                    # embeddings[i].append((self.embeddings[n1][i] * self.embeddings[n2][i]))


        df = pd.DataFrame()
        df['u'] = u
        df['v'] = v
        # df['jac'] = jac_co
        # df['adam'] = adam
        # df['nbrs'] = nbrs
        # df['att'] = att
        # df['spl'] = spl
        # df['katz_centrality'] = katz_centralities
        # if self.katz:
        #     df['katz'] = katzes
        if self.embeddings:
            for i, col in enumerate(self.emb_cols):
                df[col] = embeddings[i]

        if verbose:
            print("\t%d pairs checked and %d pairs in dataframe" % (count, df.shape[0]))
        df.sample(frac=1)
        return df, labels

    def to_embedding_dataframe(self, pairs, label_graph):
        cols = [c for c in self.emb_cols]
        data = []
        labels = []
        for u, v in pairs:
            labels.append(label_graph.nx_graph.has_edge(u, v))
            if len(self.embeddings[u]) != len(self.embeddings[u]):
                print("len error...")
                print self.embeddings[u]
                print self.embeddings[v]
            data.append(np.mean((self.embeddings[u], self.embeddings[v]), axis=0))
        df = pd.DataFrame(data, columns=cols)
        return df, labels
