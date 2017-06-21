import networkx as nx
import numpy as np
import random
import json
import re
import datetime
from itertools import chain
import pandas as pd
from collections import defaultdict

filenames = ['SAfrica-community-relevant-restricted.json',
             'Kenya-community-relevant-restricted.json',
             'Nigeria-community-relevant-restricted.json']

tweetList = []
timeList = []
userList = []
ht_counts = defaultdict(int)


def LoadTwitterGraph(directory, country, sample_amount=None, n_tweets=0):
    mypath = directory + filenames[country]
    G = nx.Graph()

    tweetList[:] = []
    timeList[:] = []
    userList[:] = []
    userNameDict = {}

    timeFormat = "%Y-%m-%dT%H:%M:%S.%fZ"
    numTweets = 0

    with open(mypath) as f1:
        for line in f1:
            if (sample_amount and random.random() < sample_amount) or not sample_amount:
                numTweets += 1

                if not sample_amount and n_tweets and numTweets > n_tweets:
                    print("Loaded %d tweets" % (numTweets - 1))
                    return G

                tweetObj = json.loads(line)
                currentTime = datetime.datetime.strptime(tweetObj['postedTime'], timeFormat)
                id2 = int(re.findall('^.*:([0-9]+)$', str(tweetObj['actor']['id']))[0])
                uName = tweetObj['actor']['preferredUsername']

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

                    if not G.has_edge(id1, id2):
                        G.add_edge(id1, id2, posted=[currentTime], n_links=1)
                    else:
                        timeStamps = G.edge[id1][id2]['posted']
                        i = 0
                        while i < len(timeStamps) and timeStamps[i] > currentTime:
                            i += 1
                        G.edge[id1][id2]['posted'].insert(i, currentTime)
                        G.edge[id1][id2]['n_links'] += 1
                for ht in tweetObj['twitter_entities']['hashtags']:
                    #ht_counts[ht['text'].lower()] += 1
                    text = ht['text'].lower()
                    ht_counts[text] += 1
                    if text not in G:
                        G.add_node(text, n_uses=1, type="hashtag")
                    else:
                        G.node[text]['n_uses'] += 1

                    if not G.has_edge(id2, text):
                        G.add_edge(id2, text, posted=[currentTime])
                    else:
                        timeStamps = G.edge[id2][text]['posted']
                        i = 0
                        while i < len(timeStamps) and timeStamps[i] > currentTime:
                            i += 1
                        G.edge[id2][text]['posted'].insert(i, currentTime)
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

    if ed:
        graph.add_edge(u, v, ed)

    return distance


def get_jac(graph, u, v):
    jac = 0
    j = nx.jaccard_coefficient(graph, [(u, v)])
    for x, y, p in j:
        jac = p
    return jac


def get_adam(graph, u, v):
    adam = 0
    j = nx.adamic_adar_index(graph, [(u, v)])
    try:
        for x, y, p in j:
            adam = p
    except:
        adam = 0
    return adam


def get_att(graph, u, v):
    att = 0
    j = nx.preferential_attachment(graph, [(u, v)])
    for x, y, p in j:
        att = p
    return att


def get_nbrs(graph, u, v):
    nbrs = 0
    for nbr in nx.common_neighbors(graph, u, v):
        nbrs += 1

    return nbrs


def remove_degree_zero_nodes(graph):
    for node in graph.nodes():
        if graph.degree(node) == 0:
            graph.remove_node(node)


def dataframe_from_graph(graph, pairs=True, sampling=None, label_graph=None, allow_hashtags=False, min_degree=0):
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
        if random.random() < sampling or (label_graph and label_graph.has_edge(n1, n2)):
            deg1 = degree[n1]
            deg2 = degree[n2]
            if (deg1 > min_degree or deg2 > min_degree) and (allow_hashtags or (graph.node[n1]['type'] != 'hashtag' and graph.node[n2]['type'] != 'hashtag')):
                if count % 150000 == 0:
                    print("%d in set so far..." % count)
                count += 1
                u.append(n1)
                v.append(n2)
                # has_links.append(graph.has_edge(n1, n2))
                jac_co.append(get_jac(graph, n1, n2))
                adam.append(get_adam(graph, n1, n2))
                # att.append(get_att(graph, n1, n2))
                att.append(deg1 * deg2)
                nbrs.append(get_nbrs(graph, n1, n2))
                spl.append(get_sp(graph, n1, n2))

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
