import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import twittergraph as tg
import ditwittergraph as dtg
import networkx as nx
import pandas as pd
import numpy as np

# change these values to setup a test....
n_trials = 1
data_directory = '/Volumes/pond/Temp/twitter/'
out_file = 'delete.csv' # don't forget to change the filename
sample_amount = 0.01
katz_threshold = 0.0069
use_hashtags = True
probability_threshold = 0.01
fields = ['katz', 'att', 'jac', 'adam', 'nbrs', 'spl']
use_auc = False
directed = True
first_split = datetime.datetime(2014, 5, 5)
second_split = datetime.datetime(2014, 5, 10)

if directed:
    graph = dtg.LoadTwitterGraph(data_directory, 0, hashtags=use_hashtags)

# else:
# graph = tg.LoadTwitterGraph(data_directory, 0, hashtags=use_hashtags)


def remove_edges_after(split, g):
    new_graph = g.copy()
    for u, v in g.edges():
        for i in range(0, len(new_graph.edge[u][v]['posted'])):
            if new_graph.edge[u][v]['posted'][0] > split:
                new_graph.edge[u][v]['posted'].pop(0)
                if not new_graph.node[u]['type'] == 'hashtag' and not new_graph.node[v]['type'] == 'hashtag':
                    new_graph.edge[u][v]['n_links'] -= 1
                else:
                    if new_graph.node[u]['type'] == 'hashtag':
                        new_graph.node[u]['n_uses'] -= 1
                    else:
                        new_graph.node[v]['n_uses'] -= 1
        if len(new_graph.edge[u][v]['posted']) == 0:
            new_graph.remove_edge(u, v)
    return new_graph


g_0 = remove_edges_after(first_split, graph)
g_1 = remove_edges_after(second_split, graph)
g_2 = graph.copy()

print("Precomputing katz 1")
kz_0 = nx.katz_centrality(g_0, alpha=.005, beta=.1, tol=.00000001, max_iter=5000)
print("Precomputing katz 2")
kz_1 = nx.katz_centrality(g_1, alpha=.005, beta=.1, tol=.00000001, max_iter=5000)
results = []

n = 0
while n < n_trials:
    print("\nBeginning trial: %d" % (n + 1))
    df_train, y_train = dtg.dataframe_from_graph(g_0, pairs=False, sampling=sample_amount,
                                                 label_graph=g_1, min_katz=katz_threshold, cheat=False, verbose=True, katz=kz_0)

    df_test, y_test = dtg.dataframe_from_graph(g_1, pairs=False, sampling=sample_amount,
                                               label_graph=g_2, min_katz=katz_threshold, verbose=True, cheat=False, katz=kz_1)

    rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1, class_weight='balanced')
    x_train = df_train.loc[:, fields]
    x_test = df_test.loc[:, fields]

    classifier = rf.fit(x_train, y_train)
    pred = classifier.predict_proba(x_test)

    if use_auc:
        auc = roc_auc_score(y_test, pred[:, 1])
        results.append(auc)
        print("Trial %d AUC: %.4f" % (n + 1, auc))
    else:
        bin_pred = []
        for i in range(len(pred)):
            if pred[i, 1] > probability_threshold:
                bin_pred.append(True)
            else:
                bin_pred.append(False)

        (pr, re, fs, su) = precision_recall_fscore_support(y_test, bin_pred, average='binary')
        results.append(fs)
        print("Trial %d F-Measure: %.4f" % (n + 1, fs))
    print("After %d trials: " % (n + 1))
    print("\tMax: %.4f" % np.max(results))
    print("\tMin: %.4f" % np.min(results))
    print("\tMean: %.4f" % np.mean(results))
    print("\tStandard Deviation: %.4f" % np.std(results))
    n += 1

df_save = pd.DataFrame(data=results, columns=['results'])
df_save.to_csv(out_file)
