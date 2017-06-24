import networkx as nx
import numpy as np
import random
import json
import re
import datetime
from itertools import chain
import pandas as pd
from collections import defaultdict
import math

filenames = ['SAfrica-community-relevant-restricted.json',
             'Kenya-community-relevant-restricted.json',
             'Nigeria-community-relevant-restricted.json']

tweetList = []
timeList = []
userList = []
rt_count = 0
ht_counts = defaultdict(int)
tweets = {}


def LoadTwitterGraph(directory, country, sample_amount=None, n_tweets=0, hashtags=True):
    """
    :param directory: Directory where data files are
    :param country: 0 - SA, 1 - Kenya, 2 - Nigera (do not rename files!)
    :param sample_amount: (optional) NONE - load whole graph, if specified sample by sample_amount
    :param n_tweets: (optional, ignored if sample_amount specified) load only n_tweets tweets
    :return: networkx graph object
    """
    mypath = directory + filenames[country]
    G = nx.DiGraph()

    tweetList[:] = []
    timeList[:] = []
    userList[:] = []
    userNameDict = {}

    timeFormat = "%Y-%m-%dT%H:%M:%S.%fZ"
    numTweets = 0

    with open(mypath) as f1:
        for line in f1:
            if (sample_amount and random.random() < sample_amount) or not sample_amount:

                if not sample_amount and n_tweets and numTweets > n_tweets:
                    print("Loaded %d tweets" % (numTweets - 1))
                    return G

                tweetObj = json.loads(line)
                currentTime = datetime.datetime.strptime(tweetObj['postedTime'], timeFormat)
                id2 = int(re.findall('^.*:([0-9]+)$', str(tweetObj['actor']['id']))[0])
                uName = tweetObj['actor']['preferredUsername']
                tweets[numTweets] = tweetObj['body']
                numTweets += 1

                if uName not in userNameDict:
                    userNameDict[uName] = id2
                else:
                    id2 = userNameDict[uName]

                if not id2:
                    continue

                if id2 not in G:
                    G.add_node(id2, n_tweets=1, n_mentions=0, u_name=uName, type="user")
                else:
                    G.node[id2]['n_tweets'] += 1

                for ui in tweetObj['twitter_entities']['user_mentions']:
                    id1 = ui['id']
                    name2 = ui['screen_name']
                    if not id1 or id1 == id2:
                        continue

                    if name2 not in userNameDict:
                        userNameDict[name2] = id1
                    else:
                        id1 = userNameDict[name2]

                    if id1 not in G:
                        G.add_node(id1, n_tweets=0, n_mentions=1, u_name=name2, type="user")
                    else:
                        G.node[id1]['n_mentions'] += 1

                    if not G.has_edge(id2, id1):
                        G.add_edge(id2, id1, posted=[currentTime], n_links=1, tweets=[numTweets])
                    else:
                        timeStamps = G.edge[id2][id1]['posted']
                        i = 0
                        while i < len(timeStamps) and timeStamps[i] > currentTime:
                            i += 1
                        G.edge[id2][id1]['posted'].insert(i, currentTime)
                        G.edge[id2][id1]['tweets'].insert(i, numTweets)
                        G.edge[id2][id1]['n_links'] += 1
                if hashtags:
                    for ht in tweetObj['twitter_entities']['hashtags']:
                        #ht_counts[ht['text'].lower()] += 1
                        text = ht['text'].lower()
                        ht_counts[text] += 1
                        if text not in G:
                            G.add_node(text, n_uses=1, type="hashtag")
                        else:
                            G.node[text]['n_uses'] += 1

                        if not G.has_edge(id2, text):
                            G.add_edge(id2, text, posted=[currentTime], tweets=[numTweets])
                        else:
                            timeStamps = G.edge[id2][text]['posted']
                            i = 0
                            while i < len(timeStamps) and timeStamps[i] > currentTime:
                                i += 1
                            G.edge[id2][text]['posted'].insert(i, currentTime)
                            G.edge[id2][text]['tweets'].insert(i, numTweets)
                try:
                    if (not tweetObj['body'].lower().startswith("rt")):
                        # Increment tweet count
                        tweetList.append(tweetObj['body'].lower())
                        timeList.append(currentTime)
                        userList.append(tweetObj['actor']['id'])
                        # print globalTweetCounter, tweetObj['body'].lower()
                except:
                    pass
    print("Loaded %d tweets" % numTweets)
    return G


def candidate_list(graph):
    """
    Don't use this, supposed to generate list of "edge candidates"
    Doesn't work
    """
    deg = nx.degree(graph)
    n_tweets = defaultdict(int)
    n_mentions = defaultdict(int)
    pair_dict = defaultdict(bool)
    pairs = []
    pair_count = 0
    for u in graph.nodes_iter():
        if graph.node[u]['type'] == 'user':
            n_tweets[u] = graph.node[u]['n_tweets']
            n_mentions[u] = graph.node[u]['n_mentions']
        else:
            continue
        if n_tweets[u] > 15 or n_mentions[u] > 15:
            for v in graph.nodes_iter():
                if graph.node[v]['type'] == 'hashtag':
                    continue
                if deg[v] > 0 or n_tweets[v] > 0 or n_mentions[v] > 0:
                    (u, v) = sorted((u, v))
                    if not pair_dict[(u,v)]:
                        pair_dict[(u, v)] = True
                        pairs.append((u, v))
                        pair_count += 1
    return pairs


def all_pairs(graph):
    return chain(graph.edges(), nx.non_edges(graph))


def get_sp(graph, u, v):
    ed = None

    if graph.has_edge(u, v):
        ed = graph.edge[u][v]
        graph.remove_edge(u, v)

    try:
        distance = nx.shortest_path_length(graph, u, v)
    except:
        distance = 1000000

    if distance == 0:       # I check to prevent a user from mentioning him- or herself.
        distance = 1000000  # It seems to work except for one time... I might track it down later

    if ed:
        graph.add_edge(u, v, ed)

    return distance

def get_att(graph, u, v):
    att = 0
    j = nx.preferential_attachment(graph, [(u, v)])
    for x, y, p in j:
        att = p
    return att


def common_nbrs(graph, u, v):
    u_adj = graph.neighbors(u)
    v_adj = graph.neighbors(v)
    nbrs = []
    for u in u_adj:
        if u in v_adj:
            nbrs.append(u)

    return nbrs, u_adj, v_adj

zero_count = 0

def get_unsupported(graph, u, v):
    jac = adam = n_nbrs = attachment = 0
    nbrs, u_adj, v_adj = common_nbrs(graph, u, v)
    n_nbrs = len(nbrs)
    union_magn = len(u_adj) + len(v_adj)
    if union_magn:
        jac = float(n_nbrs) / float(union_magn)

    for nbr in nbrs:
        deg = graph.degree(nbr)
        if deg > 1:
            adam += 1/math.log(deg)

    attachment = len(u_adj) * len(v_adj)
    return jac, adam, n_nbrs, attachment


def remove_degree_zero_nodes(graph):
    for node in graph.nodes():
        if graph.degree(node) == 0:
            graph.remove_node(node)


def dataframe_from_graph(graph, pairs=True, sampling=None, label_graph=None, cheat=False, allow_hashtags=False, min_katz=0, verbose=True, katz=None):
    """
    :param graph: Graph to generate the dataframe from
    :param pairs: (Optional) TRUE-Use all pairs, False-only use non-edges, or a list of tuples of pairs to use
    :param sampling: (optional) NONE - use whole graph, otherwise amount to sample
    :param label_graph: (optional) NONE - don't force any edges in, if specified all pairs with edge in label_graph will
                        be included
    :param allow_hashtags: FALSE - only allow user nodes
    :param min_degree: one node must have degree at least min_degree to be included in dataframe
    :return: pandas dataframe with pairs of nodes and their various metrics
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
    katzes = []
    count = 0
    labels = []
    # degree = nx.degree(graph)

    if type(pairs) is bool and pairs:
        iter_set = all_pairs(graph)
    elif type(pairs) is bool and not pairs:
        iter_set = nx.non_edges(graph)
    else:
        iter_set = pairs
        print("Using the pairs you provided...")

    if verbose and not katz:
        print("Precomputing katzes....")

    if not katz:
        katz = nx.katz_centrality(graph, alpha=.005, beta=.1, tol=.00000001, max_iter=5000)

    elim = 0
    for n1, n2 in iter_set:
        if random.random() < sampling or (cheat and label_graph and label_graph.has_edge(n1, n2)):
            if allow_hashtags or (graph.node[n1]['type'] != 'hashtag' and graph.node[n2]['type'] != 'hashtag'):
                count += 1
                #if verbose:
                    #if count % 1000000 == 0:
                        #print("%d checked... %d eliminated" % (count, elim))
                k_s = np.mean((katz[n1], katz[n2]))
                if k_s < min_katz:
                    elim += 1
                    continue
                u.append(n1)
                v.append(n2)
                (jaccard, adamic, n_nbrs, attachment) = get_unsupported(graph, n1, n2)
                jac_co.append(jaccard)
                adam.append(adamic)
                nbrs.append(n_nbrs)
                att.append(attachment)
                spl.append(get_sp(graph, n1, n2))
                katzes.append(np.mean((katz[n1], katz[n2])))
                labels.append(label_graph.has_edge(n1, n2))

    df = pd.DataFrame()
    df['u'] = u
    df['v'] = v
    df['jac'] = jac_co
    df['adam'] = adam
    df['nbrs'] = nbrs
    df['att'] = att
    df['spl'] = spl
    df['katz'] = katzes

    #if verbose:
        #print("%d pairs checked and %d pairs in dataframe" % (count, df.shape[0]))

    return df, labels


def dataframe_from_graph2(graph, pairs=True, sampling=None, label_graph=None, allow_hashtags=False, min_degree=0):
    """
    This is for fun.
    Don't use this method
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
    count = 0
    degree = nx.degree(graph)

    if type(pairs) is bool and pairs:
        iter_set = all_pairs(graph)
    elif type(pairs) is bool and not pairs:
        iter_set = nx.non_edges(graph)
    else:
        iter_set = pairs
        print("Using the pairs you provided...")

    for n1, n2 in iter_set:
        if random.random() < sampling:  # or (label_graph and label_graph.has_edge(n1, n2)):
            if allow_hashtags or (graph.node[n1]['type'] != 'hashtag' and graph.node[n2]['type'] != 'hashtag'):
                if count % 150000 == 0:
                    print("%d in set so far..." % count)
                shortest = get_sp(graph, n1, n2)
                count += 1
                if shortest < 10:
                    deg1 = degree[n1]
                    deg2 = degree[n2]
                    u.append(n1)
                    v.append(n2)
                    # has_links.append(graph.has_edge(n1, n2))
                    jac_co.append(get_jac(graph, n1, n2))
                    adam.append(get_adam(graph, n1, n2))
                    # att.append(get_att(graph, n1, n2))
                    att.append(deg1 * deg2)
                    nbrs.append(get_nbrs(graph, n1, n2))
                    spl.append(shortest)

    df = pd.DataFrame()
    df['u'] = u
    df['v'] = v
    # df['link'] = has_links
    df['jac'] = jac_co
    df['adam'] = adam
    df['nbrs'] = nbrs
    df['att'] = att
    df['spl'] = spl
    print("%d pairs and %d edges in dataframe" % (count, np.count_nonzero(has_links)))

    return df



def labels_for_dataframe(df, graph):
    labels = []
    for i in range(0, df.shape[0]):
        u = df.loc[i, 'u']
        v = df.loc[i, 'v']
        if graph.has_edge(u, v):
            labels.append(1)
        else:
            labels.append(0)

    return labels
