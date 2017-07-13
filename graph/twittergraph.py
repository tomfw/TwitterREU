import datetime
import json
import math
import os
import random
import re
from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from graph import Graph

# from node2vec import Graph


class TwitterGraph(Graph):
    filenames = ['SAfrica-community-relevant-restricted.json',
                 'Kenya-community-relevant-restricted.json',
                 'Nigeria-community-relevant-restricted.json']

    def __init__(self):
        """
        Create an empty twitter graph.
        """
        Graph.__init__(self)
        self.n_tweets = 0
        self.ht_counts = defaultdict(int)
        self.tweets = {}
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

                new.tweets[tweet_id]['poster'] = user_id

                if not user_id:
                    continue

                if user_id not in new.nx_graph:
                    new.nx_graph.add_node(user_id, n_tweets=1, n_mentions=0, u_name=uName, type="user")
                else:
                    new.nx_graph.node[user_id]['n_tweets'] += 1

                if tweetObj['verb'] == 'share':
                    rt_id = cls.parse_id_string(str(tweetObj['object']['id'])) # id of the original tweet
                    rt_user_id = cls.parse_id_string(str(tweetObj['object']['actor']['id'])) # id of the original tweeter
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
                        new.nx_graph.add_edge(user_id, rt_user_id, timestamps=[currentTime], tweets=[tweet_id], n_links=1, weight=1)
                    else:
                        timeStamps = new.nx_graph.edge[user_id][rt_user_id]['timestamps']
                        i = 0
                        while i < len(timeStamps) and timeStamps[i] > currentTime:
                            i += 1
                        new.nx_graph.edge[user_id][rt_user_id]['timestamps'].insert(i, currentTime)
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
        new = cls.graph_with_graph(tg)
        new.n_tweets = tg.n_tweets
        new.ht_counts = tg.ht_counts
        new.tweets = tg.tweets
        return new

    @classmethod
    def tg_by_removing_edges_after_date(cls, tg, date):
        """
        Create a new directed twitter graph by removing edges from a directed twitter graph.

        :param tg: The original graph
        :param date: The date past which all edges should be removed.
        :return: A new graph with all edges later than date removed
        """

        # todo: delete this method when possible
        new = cls.tg_with_tg(tg)
        for u, v in tg.nx_graph.edges():
            for i in range(0, len(new.nx_graph.edge[u][v]['timestamps'])):
                if new.nx_graph.edge[u][v]['timestamps'][0] > date:
                    new.nx_graph.edge[u][v]['timestamps'].pop(0)
                    t_id = new.nx_graph.edge[u][v]['tweets'].pop(0)
                    new.n_tweets -= 1
                    if not new.nx_graph.node[u]['type'] == 'hashtag' and not new.nx_graph.node[v]['type'] == 'hashtag':
                        new.nx_graph.edge[u][v]['n_links'] -= 1
                        if new.tweet_poster(t_id) == u:
                            new.nx_graph.node[u]['n_tweets'] -= 1
                            new.nx_graph.node[v]['n_mentions'] -= 1
                        else:
                            new.nx_graph.node[v]['n_tweets'] -= 1
                            new.nx_graph.node[u]['n_mentions'] -= 1
                    else:
                        if new.nx_graph.node[u]['type'] == 'hashtag':
                            new.nx_graph.node[u]['n_uses'] -= 1
                        else:
                            new.nx_graph.node[v]['n_uses'] -= 1
            if len(new.nx_graph.edge[u][v]['timestamps']) == 0:
                new.nx_graph.remove_edge(u, v)
        return new

    # todo: override subgraph_within_dates if twitter-specific stuff matters as n_tweets etc. is now inaccurate.

    @classmethod
    def tg_from_json(cls, data_directory, country, hashtags=True):
        """
        Load a mention-based graph from JSON

        :param data_directory: Directory containing JSON files.
        :param country: Index of the country to be loaded.
        :param hashtags: Include hashtags as nodes.
        :return: a new DirectedTwitterGraph
        """

        # this is probably broken.  fix or delete.

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
                        new.nx_graph.add_edge(id2, id1, timestamps=[currentTime], n_links=1, tweets=[tweet_id])
                    else:
                        timeStamps = new.nx_graph.edge[id2][id1]['timestamps']
                        i = 0
                        while i < len(timeStamps) and timeStamps[i] > currentTime:
                            i += 1
                        new.nx_graph.edge[id2][id1]['timestamps'].insert(i, currentTime)
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
                            new.nx_graph.add_edge(id2, text, timestamps=[currentTime], tweets=[tweet_id])
                        else:
                            timeStamps = new.nx_graph.edge[id2][text]['timestamps']
                            i = 0
                            while i < len(timeStamps) and timeStamps[i] > currentTime:
                                i += 1
                            new.nx_graph.edge[id2][text]['timestamps'].insert(i, currentTime)
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

    def tweet_poster(self, t_id):
        return self.tweets[t_id]['poster']

    def is_retweet(self, t_id):
        """
        Determine if a tweet is a retweet

        :param t_id: A tweet id
        :return: True if the tweet is a retweet
        """
        return self.tweets[t_id]['type'] == 'rt'

    def display_error(self, function, message):
        print("Error in TwitterGraph.%s:\n\t%s" % (function, message))