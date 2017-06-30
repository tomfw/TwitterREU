import networkx as nx
import json
import re
import datetime
from collections import defaultdict
import numpy as np
from itertools import chain
import random
import math
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TwitterGraph(object):
    filenames = ['SAfrica-community-relevant-restricted.json',
                 'Kenya-community-relevant-restricted.json',
                 'Nigeria-community-relevant-restricted.json']

    def __init__(self):
        """
        Create and empty twitter graph.
        """
        self.n_tweets = 0
        self.ht_counts = defaultdict(int)
        self.tweets = {}
        self.nx_graph = nx.Graph()
        self.n_users = 0
        self.userNameDict = {}

    def _get_user_dict(self, country):
        """
        Get the user_id to graph_id dictionary for a country
        """

        filenames = ['SA_RT_UserID_Dict.csv']
        if country > len(filenames) - 1:
            self.display_error("_get_user_dict", "Don't have user dictionary for that country yet!")
            return

        df = pd.read_csv(filenames[country])
        for i in range(0, df.shape[0]):
            self.userNameDict[df.loc[i, 'twitter_id']] = df.loc[i, 'graph_id']

    @classmethod
    def rt_graph_from_json(cls, data_directory, country):
        """
        Load a retweet-based graph from JSON

        :param data_directory: Directory containing JSON files.
        :param country: Index of country to be loaded
        :return:
        """
        mypath = data_directory + cls.filenames[country]

        new = cls()
        new._get_user_dict(country)
        timeFormat = "%Y-%m-%dT%H:%M:%S.%fZ"

        times = []
        rt_count = 0
        rt2_count = 0
        with open(mypath) as f1:
            for line in f1:

                tweetObj = json.loads(line)
                currentTime = datetime.datetime.strptime(tweetObj['postedTime'], timeFormat)
                times.append(currentTime)
                tweeter_id = cls.parse_id_string(
                    str(tweetObj['actor']['id']))  # int(re.findall('^.*:([0-9]+)$', str(tweetObj['actor']['id']))[0])
                tweet_id = cls.parse_id_string(str(tweetObj['id']))
                uName = tweetObj['actor']['preferredUsername']
                new.tweets[tweet_id] = {'text': tweetObj['body'], 'type': 'post'}
                new.n_tweets += 1

                if new.n_tweets > 10:
                    pass

                if tweeter_id not in new.userNameDict:
                    new.n_users += 1
                    new.userNameDict[tweeter_id] = new.n_users
                    user_id = new.n_users
                else:
                    user_id = new.userNameDict[tweeter_id]

                if not user_id:
                    continue

                if user_id not in new.nx_graph:
                    new.nx_graph.add_node(user_id, n_tweets=1, n_mentions=0, u_name=uName, type="user")
                else:
                    new.nx_graph.node[user_id]['n_tweets'] += 1

                if tweetObj['verb'] == 'share':
                    rt_id = cls.parse_id_string(str(tweetObj['object']['id']))
                    rt_user_id = cls.parse_id_string(str(tweetObj['object']['actor']['id']))
                    if rt_user_id not in new.userNameDict:
                        new.n_users += 1
                        new.userNameDict[rt_user_id] = new.n_users
                        rt_user_id = new.n_users
                    else:
                        rt_user_id = new.userNameDict[rt_user_id]

                    if rt_user_id not in new.nx_graph:
                        new.nx_graph.add_node(rt_user_id, n_tweets=1, n_mentions=1, type="user")
                    else:
                        new.nx_graph.node[rt_user_id]['n_mentions'] += 1
                    if not new.nx_graph.has_edge(user_id, rt_user_id):
                        new.nx_graph.add_edge(user_id, rt_user_id, posted=[currentTime], tweets=[tweet_id], n_links=1)
                    else:
                        timeStamps = new.nx_graph.edge[user_id][rt_user_id]['posted']
                        i = 0
                        while i < len(timeStamps) and timeStamps[i] > currentTime:
                            i += 1
                        new.nx_graph.edge[user_id][rt_user_id]['posted'].insert(i, currentTime)
                        new.nx_graph.edge[user_id][rt_user_id]['tweets'].insert(i, tweet_id)
                        new.nx_graph.edge[user_id][rt_user_id]['n_links'] += 1
                    new.tweets[tweet_id]['type'] = 'rt'
                    new.tweets[tweet_id]['rt_id'] = rt_id
                    rt2_count += 1

        new.min_date = np.min(times)
        new.max_date = np.max(times)
        return new

    @classmethod
    def tg_with_tg(cls, tg):
        """
        Copy a twitter graph

        :param tg: Graph to be copied
        :return: Copy of the graph
        """
        new = cls()
        new.n_tweets = tg.n_tweets
        new.ht_counts = tg.ht_counts
        new.tweets = tg.tweets
        new.nx_graph = tg.nx_graph.copy()
        new.max_date = tg.max_date
        new.min_date = tg.min_date
        return new

    @classmethod
    def tg_by_removing_edges_after_date(cls, tg, date):
        """
        Create a new directed twitter graph by removing edges from a directed twitter graph.

        :param tg: The original graph
        :type tg: twittergraph
        :param date: The date past which all edges should be removed.
        :return: A new graph with all edges later than date removed
        """
        new = cls.tg_with_tg(tg)
        for u, v in tg.nx_graph.edges():
            for i in range(0, len(new.nx_graph.edge[u][v]['posted'])):
                if new.nx_graph.edge[u][v]['posted'][0] > date:
                    new.nx_graph.edge[u][v]['posted'].pop(0)
                    new.nx_graph.edge[u][v]['tweets'].pop(0)
                    new.n_tweets -= 1
                    if not new.nx_graph.node[u]['type'] == 'hashtag' and not new.nx_graph.node[v]['type'] == 'hashtag':
                        new.nx_graph.edge[u][v]['n_links'] -= 1
                        new.nx_graph.node[u]['n_tweets'] -= 1
                        new.nx_graph.node[v]['n_mentions'] -= 1
                    else:
                        if new.nx_graph.node[u]['type'] == 'hashtag':
                            new.nx_graph.node[u]['n_uses'] -= 1
                        else:
                            new.nx_graph.node[v]['n_uses'] -= 1
            if len(new.nx_graph.edge[u][v]['posted']) == 0:
                new.nx_graph.remove_edge(u, v)
        return new

    @classmethod
    def tg_from_json(cls, data_directory, country, hashtags=True):
        """
        Load a mention-based graph from JSON

        :param data_directory: Directory containing JSON files.
        :param country: Index of the country to be loaded.
        :param hashtags: Include hashtags as nodes.
        :return: a new DirectedTwitterGraph
        """
        mypath = data_directory + cls.filenames[country]

        new = cls()

        userNameDict = {}
        timeFormat = "%Y-%m-%dT%H:%M:%S.%fZ"


        times = []
        rt_count = 0
        rt2_count = 0
        with open(mypath) as f1:
            for line in f1:

                tweetObj = json.loads(line)
                currentTime = datetime.datetime.strptime(tweetObj['postedTime'], timeFormat)
                times.append(currentTime)
                id2 = cls.parse_id_string(str(tweetObj['actor']['id']))  # int(re.findall('^.*:([0-9]+)$', str(tweetObj['actor']['id']))[0])
                tweet_id = cls.parse_id_string(str(tweetObj['id']))
                uName = tweetObj['actor']['preferredUsername']
                new.tweets[tweet_id] = {'text': tweetObj['body'], 'type': 'post'}
                new.n_tweets += 1

                if new.n_tweets > 10:
                    pass

                if tweetObj['verb'] == 'share':
                    rt_id = cls.parse_id_string(str(tweetObj['object']['id']))
                    new.tweets[tweet_id]['type'] = 'rt'
                    new.tweets[tweet_id]['rt_id'] = rt_id
                    rt2_count += 1

                if uName not in userNameDict:
                    userNameDict[uName] = id2
                else:
                    id2 = userNameDict[uName]

                if not id2:
                    continue

                if id2 not in new.nx_graph:
                    new.nx_graph.add_node(id2, n_tweets=1, n_mentions=0, u_name=uName, type="user")
                else:
                    new.nx_graph.node[id2]['n_tweets'] += 1

                for ui in tweetObj['twitter_entities']['user_mentions']:
                    id1 = ui['id']
                    name2 = ui['screen_name']
                    if not id1 or id1 == id2:
                        continue

                    if name2 not in userNameDict:
                        userNameDict[name2] = id1
                    else:
                        id1 = userNameDict[name2]

                    if id1 not in new.nx_graph:
                        new.nx_graph.add_node(id1, n_tweets=0, n_mentions=1, u_name=name2, type="user")
                    else:
                        new.nx_graph.node[id1]['n_mentions'] += 1

                    if not new.nx_graph.has_edge(id2, id1):
                        new.nx_graph.add_edge(id2, id1, posted=[currentTime], n_links=1, tweets=[tweet_id])
                    else:
                        timeStamps = new.nx_graph.edge[id2][id1]['posted']
                        i = 0
                        while i < len(timeStamps) and timeStamps[i] > currentTime:
                            i += 1
                        new.nx_graph.edge[id2][id1]['posted'].insert(i, currentTime)
                        new.nx_graph.edge[id2][id1]['tweets'].insert(i, tweet_id)
                if hashtags:
                    for ht in tweetObj['twitter_entities']['hashtags']:
                        # ht_counts[ht['text'].lower()] += 1
                        text = ht['text'].lower()
                        new.ht_counts[text] += 1
                        if text not in new.nx_graph:
                            new.nx_graph.add_node(text, n_uses=1, type="hashtag")
                        else:
                            new.nx_graph.node[text]['n_uses'] += 1

                        if not new.nx_graph.has_edge(id2, text):
                            new.nx_graph.add_edge(id2, text, posted=[currentTime], tweets=[tweet_id])
                        else:
                            timeStamps = new.nx_graph.edge[id2][text]['posted']
                            i = 0
                            while i < len(timeStamps) and timeStamps[i] > currentTime:
                                i += 1
                            new.nx_graph.edge[id2][text]['posted'].insert(i, currentTime)
                            new.nx_graph.edge[id2][text]['tweets'].insert(i, tweet_id)
        new.min_date = np.min(times)
        new.max_date = np.max(times)
        return new

    @staticmethod
    def parse_id_string(id_str):
        """
        Parse a twitter id string

        :param id_str: The twitter id link string
        :return: The integer id
        """
        return int(re.findall('^.*:([0-9]+)$', id_str)[0])

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
        jac = adam = n_nbrs = attachment = 0
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

    def to_dataframe_for_pairs(self, pairs_file, sampling=None, label_graph=None):
        """
        Get a dataframe for training or testing from a graph by providing a csv of candidate pairs.

        :param pairs_file: The path to a CSV contain u, v columns having nodes to consider
        :param sampling: Amount to consider, default=None (do not sample)
        :param label_graph: Graph to check for the true labels
        :return: A pandas dataframe with columns for computed features
        """
        pairs = pd.read_csv(pairs_file)
        if not sampling:
            sampling = 2
        if not label_graph:
            self.display_error("to_dataframe_for_pairs", "you must provide a label_graph now")
            return

        u_list = []
        v_list = []
        jac_co = []
        adam = []
        att = []
        nbrs = []
        spl = []
        katz_u = []
        katz_v = []
        count = 0
        labels = []

        katz = nx.katz_centrality(self.nx_graph, alpha=.005, beta=.1, tol=.00000001, max_iter=5000)

        for u, v in zip(pairs.u, pairs.v):
            if u not in self.nx_graph or v not in self.nx_graph:
                continue
            if random.random() < sampling:
                u_list.append(u)
                v_list.append(v)
                katz_u.append(katz[u])
                katz_v.append(katz[v])

                (jaccard, adamic, n_nbrs, attachment) = self.get_unsupported(u, v)
                jac_co.append(jaccard)
                adam.append(adamic)
                nbrs.append(n_nbrs)
                att.append(attachment)
                spl.append(self.get_sp(u, v))
                labels.append(label_graph.nx_graph.has_edge(u, v))
        df = pd.DataFrame()
        df['u'] = u_list
        df['v'] = v_list
        df['jac'] = jac_co
        df['adam'] = adam
        df['nbrs'] = nbrs
        df['att'] = att
        df['spl'] = spl
        df['katz_u'] = katz_u
        df['katz_v'] = katz_v

        print("Dataframe size: %d" % df.shape[0])
        print("%d labeled edges" % np.sum(labels))

        return df, labels

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
        katzes = []
        count = 0
        labels = []
        # degree = nx.degree(graph)

        if type(pairs) is bool and pairs:
            iter_set = self.all_pairs()
        elif type(pairs) is bool and not pairs:
            iter_set = nx.non_edges(self.nx_graph)
        else:
            iter_set = pairs
            print("Using the pairs you provided...")

        if verbose and not katz:
            print("Precomputing katzes....")

        if not katz:
            katz = nx.katz_centrality(self.nx_graph, alpha=.005, beta=.1, tol=.00000001, max_iter=5000)

        elim = 0
        for n1, n2 in iter_set:
            if random.random() < sampling or (cheat and label_graph and label_graph.nx_graph.has_edge(n1, n2)):
                if allow_hashtags or (self.nx_graph.node[n1]['type'] != 'hashtag' and self.nx_graph.node[n2]['type'] != 'hashtag'):
                    count += 1
                    if verbose:
                        if count % 1000000 == 0:
                            print("%d checked... %d eliminated" % (count, elim))
                    k_s = np.mean((katz[n1], katz[n2]))
                    if k_s < min_katz:
                        elim += 1
                        continue
                    u.append(n1)
                    v.append(n2)
                    (jaccard, adamic, n_nbrs, attachment) = self.get_unsupported(n1, n2)
                    jac_co.append(jaccard)
                    adam.append(adamic)
                    nbrs.append(n_nbrs)
                    att.append(attachment)
                    spl.append(self.get_sp(n1, n2))
                    katzes.append(np.mean((katz[n1], katz[n2])))
                    labels.append(label_graph.nx_graph.has_edge(n1, n2))

        df = pd.DataFrame()
        df['u'] = u
        df['v'] = v
        df['jac'] = jac_co
        df['adam'] = adam
        df['nbrs'] = nbrs
        df['att'] = att
        df['spl'] = spl
        df['katz'] = katzes

        if verbose:
            print("%d pairs checked and %d pairs in dataframe" % (count, df.shape[0]))

        return df, labels

    def user_tweet_count(self, u_id):
        """
        Get number of tweets made by a user

        :param u_id: A user id
        :return: Number of tweets
        """
        try:
            results = self.nx_graph.node[u_id]['n_tweets']
        except:
            self.display_error("user_tweet_count", "invalid key or not user")
            results = 0
        return results

    def make_pairs_with_edges(self, label_graph, target_positive_ratio=.5):
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
        for u, v in label_graph.nx_graph.edges_iter():
            if not self.nx_graph.has_edge(u, v):
                u, v = sorted((u, v))
                if not pairs_dict[(u, v)]:
                    pairs_dict[(u, v)] = True
                    pairs.append((u, v))
                    edges += 1

        for u, v in nx.non_edges(self.nx_graph):
            if random.random() < .05:
                (u, v) = sorted((u, v))
                if not pairs_dict[(u, v)]:
                    pairs_dict[(u, v)] = True
                    pairs.append((u, v))
            if float(edges) / len(pairs) < target_positive_ratio:
                break

        print("Found %d new edges out of %d total pairs" % (edges, len(pairs)))
        return pairs

    def user_mention_count(self, u_id):
        """
        Get the number of times a user has been mentioned

        :param u_id: User id
        :return: Number of mentions
        """
        results = 0
        try:
            results = self.nx_graph.node[u_id]['n_mentions']
        except:
            self.display_error("user_mention_count", "invalid key or not user")
        return results

    def user_tweets(self, u_id):
        """
        Get a list of tweets made by a user.

        :param u_id: A user id
        :return: A list of tweet ids
        """
        tweets = []
        for u, v, data in self.nx_graph.edges(u_id, data=True):
            for t in data['tweets']:
                if t not in tweets:
                    tweets.append(t)
        return tweets

    def tweet_sentiment(self, t_id):
        """
        Get the sentiment of a given tweet.

        :param t_id: A tweet id.
        :return: A 4-tuple (positve, neutral, negative, composite)
        """
        text = self.tweet_text(t_id)
        sia = SentimentIntensityAnalyzer()
        s = sia.polarity_scores(text)
        return s['pos'], s['neu'], s['neg'], s['compound']

    def tweet_text(self, t_id):
        """
        Get the text of a tweet

        :param t_id: A tweet id
        :return: The text of the tweet
        """
        return self.tweets[t_id]['text']

    def is_hashtag(self, node):
        """
        Determine if a node is a hashtag

        :param node: A node in the nx_graph
        :return: True if the node is a hashtag
        """
        return self.nx_graph.node[node]['type'] == 'hashtag'

    def is_retweet(self, t_id):
        """
        Determine if a tweet is a retweet

        :param t_id: A tweet id
        :return: True if the tweet is a retweet
        """
        return self.tweets[t_id]['type'] == 'rt'

    def display_error(self, function, message):
        print("Error in DirectedTwitterGraph.%s:\n\t%s" % (function, message))