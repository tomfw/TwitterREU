import networkx as nx
import numpy as np
import random
import json
import re
import datetime

filenames = ['SAfrica-community-relevant-restricted.json',
             'Kenya-community-relevant-restricted.json',
             'Nigeria-community-relevant-restricted.json']

tweetList = []
timeList = []
userList = []


def LoadTwitterGraph(directory, country, sample_amount=None, n_tweets=0):
    mypath = directory + filenames[country]
    G = nx.Graph()

    tweetList[:] = []
    timeList[:] = []
    userList[:] = []

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

                if not id2:
                    continue

                if not id2 in G:
                    G.add_node(id2, n_tweets=1, n_mentions=0)
                else:
                    G.node[id2]['n_tweets'] += 1

                for ui in tweetObj['twitter_entities']['user_mentions']:
                    id1 = ui['id']

                    if not id1:
                        continue

                    if not id1 in G:
                        G.add_node(id1, n_tweets=0, n_mentions=1)
                    else:
                        G.node[id1]['n_mentions'] += 1

                    if not G.has_edge(id1, id2):
                        G.add_edge(id1, id2, posted=currentTime, n_links=1)
                    else:
                        oldTime = G.edge[id1][id2]['posted']
                        if currentTime > oldTime:
                            newTime = currentTime
                        else:
                            newTime = oldTime

                        G.edge[id1][id2]['posted'] = newTime
                        G.edge[id1][id2]['n_links'] += 1
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
