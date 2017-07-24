from graph.graph import Graph
from graph.twittergraph import TwitterGraph as tg
from graph.fbgraph import FBGraph as fb
from graph.enrongraph import EnronGraph as eg
from graph.collabgraph import CollabGraph as cg
import numpy as np
import pandas as pd
import subprocess
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import time

class TestMachine(object):
    def __init__(self, data_root, exe_root, output_file):
        self.data_root = data_root
        self.file_names = {'southafrica': '0',
                           'fb': 'FacebookFilteredAdj_90Days_6ActiveTimes_30OutInDeg.mat',
                           'enron': 'EnronDirectedWithCc_7days.mat',
                           'collab': 'citation2Filtered.mat'}
        self.day_graphs = ['southafrica']
        self.core_nodes = None
        self.prev_embeds = None
        self.graph = None
        self.sub_graphs = None
        self.last_data = None
        self.line_path = exe_root + 'line'
        self.n2v_path = exe_root + 'node2vec'
        self.rf_path = exe_root + 'retrofit_word2vec_one'
        self.output_file = output_file
        self.times = []
        self.iters = 0
        self.output_frame = pd.DataFrame(columns=['data',
                                                  'alpha',
                                                  'beta',
                                                  'window',
                                                  'negative',
                                                  'n_walks',
                                                  'l_walks',
                                                  'period',
                                                  'n_periods',
                                                  'train',
                                                  'valid',
                                                  'test',
                                                  'v_auc',
                                                  'v_prauc',
                                                  'v_ndcg',
                                                  't_auc',
                                                  't_prauc',
                                                  't_ndcg'])

    def _load_graph(self, dataset):
        if dataset in 'southafrica':
            graph = tg.rt_graph_from_json(self.data_root, 0)
        elif dataset in 'fb':
            graph = fb.load_fb_graph(self.data_root + self.file_names[dataset])
        elif dataset in 'enron':
            graph = eg.load_enron_graph(self.data_root + self.file_names[dataset])
        elif dataset in 'collab':
            graph = cg.load_collab_graph(self.data_root + self.file_names[dataset])
        else:
            print("Error: %s is unrecognized.\n" % dataset)
            return None
        return graph

    def run_tests(self, tests):
        self.iters = 0
        start = time.clock()
        for i, test in enumerate(tests):
            print("Running test %d/%d on %s" % (i+1, len(tests), test['data']))
            result = self.run_test(test)
            if not result:
                print("Error running test: %d" % (i + 1))
        end = time.clock()
        t_time = end - start

        print("\n\n\nCompleted %d tests in %.4f seconds" % (len(tests), t_time))
        print("Average time/test: %.4f" % (t_time / len(tests)))
        print("Average time/iteration: %.4f" % (t_time / self.iters))

    def run_test(self, test):
        if not self.last_data or not self.graph or test['data'] not in self.last_data:
            graph = self._load_graph(test['data'])
            self.graph = graph
            self.last_data = test['data']
            if test['data'] in self.day_graphs:
                sgs = graph.subgraphs_of_length(days=test['days'])
                self.sub_graphs = self.copy_subgraphs(sgs)
            else:
                sgs = graph.subgraphs_of_length(periods=test['period'])
                self.sub_graphs = self.copy_subgraphs(sgs)
        else:
            print("Reusing graph")
            graph = self.graph
            sgs = self.copy_subgraphs(self.sub_graphs)
        test['n_periods'] = len(sgs)

        self.make_core_nodes(graph, sgs)
        tp = test['train']
        testp = test['test']
        vp = test['valid']
        max_period = len(sgs) - 1
        if tp[1] > max_period or testp[1] > max_period or vp[1] > max_period:
            print("Error train, test, or valid out of range")
            return False
        elif tp[0] < 1 or testp[0] < 1:
            print("Error train or test less than 1!")
            return False
        self._run_test(graph, sgs, test)
        return True
    
    def copy_subgraphs(self, sgs):
        sub_graphs = []
        for sg in sgs:
            sub_graphs.append(Graph.graph_with_graph(sg))
        return sub_graphs
    
    def _run_test(self, graph, sgs, test):
        data_folder = self.data_root + 'temp/'
        embed_file = data_folder + 'embeddings.txt'
        walk_file = data_folder + 'walks.txt'
        init_file = data_folder + 'init.txt'
        beta_file = data_folder + 'betas.txt'
        edge_file = data_folder + 'e_list.txt'

        dims = test['dims']
        print("Dims: %d" % dims)
        tr_len = test['train'][1] - test['train'][0]
        te_len = test['test'][1] - test['test'][0]
        v_len = test['valid'][1] - test['valid'][0]
        distance = 0

        emb_command = self.rf_command(walk_file,
                                      embed_file,
                                      init_file,
                                      beta_file,
                                      alpha=test['alpha'],
                                      beta=test['beta'],
                                      window=test['window'],
                                      negative=test['negative'],
                                      size=test['dims'])

        walk_command = self.n2v_command(edge_file,
                                        walk_file,
                                        p=test['p'],
                                        q=test['q'],
                                        n_walks=test['n_walks'],
                                        walk_length=test['l_walks'])
        classifier = None
        pred = None
        v_pred = None
        y_valid = None
        y_test = None
        for i, sg in enumerate(sgs):
            self.iters += 1
            cum = graph.subgraph_within_dates(sgs[0].min_date, sg.max_date).nx_graph
            sg.save_edgelist(edge_file)
            if i == 0:
                # todo: does line have a window_size?
                init_graph = graph.subgraph_within_dates(sg.min_date, sgs[test['train'][0]].min_date)
                print("Initial graph: ")
                print(init_graph.min_date, init_graph.max_date)
                init_graph.save_edgelist(edge_file)
                self.run_command(self.line_command(edge_file,
                                                   output=embed_file,
                                                   size=dims,
                                                   negative=test['negative']))
                sg.load_embeddings(embed_file, dims)
                sg.save_embeddings(init_file, dims)
                self.prev_embeds = self.store_core_embeds(sg.embeddings)
            else:
                prev = sgs[i - 1]
                if i == test['train'][0]:
                    train_graph = graph.subgraph_within_dates(sg.min_date, sgs[i + tr_len].max_date)
                    # print("Training... ")
                    # print(sg.min_date, sgs[i + tr_len].max_date)
                    train_graph.embeddings = prev.embeddings
                    train_graph.emb_cols = prev.emb_cols
                    train_pairs = prev.make_pairs_with_edges(train_graph, test['ratio'], enforce_non_edge=False,
                                                             enforce_has_embeddings=True)
                    #df_train, y_train = prev.to_dataframe(pairs=train_pairs, label_graph=train_graph, verbose=False)
                    df_train, y_train = prev.to_embedding_dataframe(train_pairs, train_graph)
                    rf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0,
                                                n_jobs=-1)
                    fields = prev.emb_cols
                    x_train = df_train.loc[:, fields]
                    classifier = rf.fit(x_train, y_train)
                if i == test['valid'][0]:
                    valid_graph = graph.subgraph_within_dates(sg.min_date, sgs[i + v_len].max_date)
                    valid_graph.embeddings = prev.embeddings
                    valid_graph.emb_cols = prev.emb_cols
                    valid_pairs = prev.make_pairs_with_edges(valid_graph, test['ratio'], enforce_non_edge=False,
                                                             enforce_has_embeddings=True)
                    df_test, y_valid = prev.to_dataframe(valid_pairs, label_graph=valid_graph, verbose=False)
                    fields = prev.emb_cols
                    x_test = df_test.loc[:, fields]
                    v_pred = classifier.predict_proba(x_test)

                if i == test['test'][0]:
                    test_graph = graph.subgraph_within_dates(sg.min_date, sgs[i + te_len].max_date)
                    # print("Testing... ")
                    # print(sg.min_date, sgs[i + te_len].max_date)
                    test_graph.embeddings = prev.embeddings
                    test_graph.emb_cols = prev.emb_cols
                    test_pairs = prev.make_pairs_with_edges(test_graph, 0, enforce_non_edge=False,
                                                            enforce_has_embeddings=True)
                    #df_test, y_test = prev.to_dataframe(test_pairs, label_graph=test_graph, verbose=True)
                    df_test, y_test = prev.to_embedding_dataframe(test_pairs, test_graph)
                    fields = prev.emb_cols
                    x_test = df_test.loc[:, fields]
                    pred = classifier.predict_proba(x_test)

                if y_test and (y_valid or test['valid'][0] < 0):
                    break
                sg.generate_embeddings_with_prev(prev.embeddings, dims)
                self.run_command(walk_command)
                self.run_command(emb_command)
                sg.load_embeddings(embed_file, dims)  # update embeddings with output from w2v
                sg.save_embeddings(init_file, dims)
                distance += self.core_movement(sg.embeddings)
                self.prev_embeds = self.store_core_embeds(sg.embeddings)

        if len(self.core_nodes) and len(sgs):
            mean_core_mvt = distance / len(sgs) / len(self.core_nodes)
        else:
            mean_core_mvt = -1

        test['movement'] = mean_core_mvt

        if y_valid:
            auc, prauc, ndcg = self._compute_metrics(v_pred, y_valid)
        else:
            auc = prauc = ndcg = -1
        test['v_auc'] = auc
        test['v_prauc'] = prauc
        test['v_ndcg'] = ndcg

        auc, prauc, ndcg = self._compute_metrics(pred, y_test)
        test['t_auc'] = auc
        test['t_prauc'] = prauc
        test['t_ndcg'] = ndcg
        print("\tAUC: %.4f PR-AUC: %.4f NDCG: %.4f" % (auc, prauc, ndcg))
        self.store_results(test)

    def store_results(self, test):
        if 'days' in test:
            d = test['days']
            del test['days']
            test['period'] = d
        self.output_frame = self.output_frame.append(test, ignore_index=True)
        self.output_frame.to_csv(self.output_file)

    def _compute_metrics(self, pred, y_true):
        auc = roc_auc_score(y_true, pred[:, 1])
        prauc = average_precision_score(y_true, pred[:, 1])
        ndcg = self.ndcg_score(y_true, pred[:, 1], k=50)
        return auc, prauc, ndcg

    def make_test(self, **kwargs):
        test = self.default_test()
        if 'train' not in kwargs or 'test' not in kwargs or 'valid' not in kwargs:
            print("train, test, valid, are required")
            return None
        if 'data' not in kwargs:
            print("You must specify a dataset.")
            return None
        elif kwargs['data'] not in self.file_names:
            print("Invalid dataset!")
            return None
        for key, value in kwargs.items():
            test[key] = value
        is_valid = self._verify_test(test)
        if is_valid:
            return test
        return None

    def _verify_test(self, test):
        if test['data'] not in self.file_names:
            print("Dataset invalid")
            return False
        if len(test['train']) == 1:
            test['train'] = (test['test'], test['test'])
        if len(test['test']) == 1:
            test['test'] = (test['test'], test['test'])
        if len(test['valid']) == 1:
            test['valid'] = (test['valid'][0], test['valid'][0])
        if 'days' in test:
            if test['data'] not in self.day_graphs:
                print("Days specified, but graph requires periods")
                return False
        elif 'days' not in test and test['data'] in self.day_graphs:
            print("%s requires days." % test['data'])
            return False
        elif 'period' not in test:
            print("You must specify period per graph")
            return False

        train = test['train']
        valid = test['valid']
        test_period = test['test']

        if train[1] >= valid[0] >= 0:  # negative valid allowed to skip validation
            print("Must validate after train period. (No overlap)")
            return False
        if test_period[0] <= train[1]:
            print("Must test after training and validation (No overlap)")
            return False

        return True

    @staticmethod
    def default_test():
        test = {'alpha': .025,
                'beta': 1,
                'n_walks': 50,
                'l_walks': 50,
                'p': 1,
                'q': 1,
                'negative': 5,
                'window': 5,
                'dims': 128,
                'ratio': .5}
        return test

    def make_core_nodes(self, graph, sgs):
        self.core_nodes = []
        self.prev_embeds = []  # subgraphs * len(core_nodes)
        for _ in sgs:
            self.prev_embeds.append([])
        for node in graph.nx_graph.nodes_iter():
            is_core = True
            for sg in sgs:
                if sg.nx_graph.degree(node) == 0:
                    is_core = False
            if is_core:
                self.core_nodes.append(node)

    def store_core_embeds(self, embed_dict):
        embeds = []
        for node in self.core_nodes:
            embeds.append(embed_dict[node])
        return embeds

    def core_movement(self, embed_dict):
        dist = 0
        for i, node in enumerate(self.core_nodes):
            dist += self.embedding_distance(self.prev_embeds[i], embed_dict[node])
        return dist

    @staticmethod
    def embedding_distance(x1, x2):
        d = 0
        for x, y in zip(x1, x2):
            try:
                d += (float(x) - float(y)) ** 2
            except:
                print("Strange error in distance...")
                print(x, y)

        return np.sqrt(d)

    def run_command(self, command):
        process = subprocess.Popen(command, stderr=subprocess.PIPE)
        err = process.communicate()
        if err[0]:
            print err

    def line_command(self, train, output, size=128, threads=16, negative=5):
        # todo: order, rho, etc...
        command = [self.line_path, "-train", train, "-output", output, "-size", str(size), "-threads", str(threads),
                   "-negative", str(negative)]
        return command

    def rf_command(self, input, output, init, beta_file, alpha=.025, size=128, window=5, sample=0, negative=5,
                   threads=16, beta=1):
        command = [self.rf_path, "-train", input, "-init", init, "-output", output,
                   "-size", str(size), "-window", str(window), "-sample", str(sample),
                   "-negative", str(negative), "-threads", str(threads), "-alpha", str(alpha), "-beta", str(beta),
                   "-cbow", '0']
        return command

    def n2v_command(self, edge_file, output, n_walks=10, walk_length=50, p=1, q=1):
        command = [self.n2v_path, '-i:' + edge_file, '-o:' + output, '-p:' + str(p), '-q:' + str(q),
                   '-r:' + str(n_walks), '-l:' + str(walk_length), '-w', '1', '-v', '1']
        return command

    def dcg_score(self, y_true, y_score, k=10, gains="exponential"):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])

        if gains == "exponential":
            gains = 2 ** y_true - 1
        elif gains == "linear":
            gains = y_true
        else:
            raise ValueError("Invalid gains option.")

        # highest rank is 1 so +2 instead of +1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def ndcg_score(self, y_true, y_score, k=10, gains="exponential"):
        best = self.dcg_score(y_true, y_true, k, gains)
        actual = self.dcg_score(y_true, y_score, k, gains)
        return actual / best
