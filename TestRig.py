import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from twittergraph import TwitterGraph as tg
import ditwittergraph as dtg
import networkx as nx
import pandas as pd
import numpy as np


# change these values to setup a test....
n_trials = 5
data_directory = '/Volumes/pond/Temp/twitter/'
out_file = 'delete.csv' # don't forget to change the filename
sample_amount = 2
katz_threshold = 0
use_hashtags = True
probability_threshold = 0.5
fields = ['katz', 'att', 'jac', 'adam', 'nbrs', 'spl']
use_auc = True
directed = True
first_split = datetime.datetime(2014, 5, 5)
second_split = datetime.datetime(2014, 5, 10)

graph = tg.rt_graph_from_json(data_directory, 0)

g_0 = tg.tg_by_removing_edges_after_date(graph, first_split)
g_1 = tg.tg_by_removing_edges_after_date(graph, second_split)
g_2 = tg.tg_with_tg(graph)

print("Precomputing katz 1")
kz_0 = nx.katz_centrality(g_0.nx_graph, alpha=.005, beta=.1, tol=.00000001, max_iter=5000)
print("Precomputing katz 2")
kz_1 = nx.katz_centrality(g_1.nx_graph, alpha=.005, beta=.1, tol=.00000001, max_iter=5000)
results = []
prauc = []

n = 0
while n < n_trials:
    print("\nBeginning trial: %d" % (n + 1))
    train_pairs = g_0.make_pairs_with_edges(g_1, .3333)
    test_pairs = g_1.make_pairs_with_edges(g_2, .3333)

    df_train, y_train = g_0.to_dataframe(pairs=train_pairs, sampling=sample_amount,
                                         label_graph=g_1, min_katz=katz_threshold, cheat=False, verbose=False, katz=kz_0)

    df_test, y_test = g_1.to_dataframe(pairs=test_pairs, sampling=sample_amount,
                                       label_graph=g_2, min_katz=katz_threshold, verbose=False, cheat=False, katz=kz_1)

    rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1, class_weight='balanced')
    x_train = df_train.loc[:, fields]
    x_test = df_test.loc[:, fields]

    classifier = rf.fit(x_train, y_train)
    pred = classifier.predict_proba(x_test)

    if use_auc:
        auc = roc_auc_score(y_test, pred[:, 1])
        pr = average_precision_score(y_test, pred[:, 1], average='macro')
        results.append(pr)
        #prauc.append(pr)
        print("Trial %d AUC: %.4f" % (n + 1, auc))
        print("Trial %d PRAUC: %.4f" % (n+1, pr))
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
