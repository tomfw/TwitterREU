from graph import Graph
import networkx as nx
import scipy.io as sio


class CollabGraph(Graph):

    def __init__(self):
        Graph.__init__(self)

    @classmethod
    def load_collab_graph(cls, f_name):
        new = cls()
        for time in range(9, -1, -1):
            matrix = sio.loadmat(f_name)["adj"]
            sg = nx.from_numpy_matrix(matrix[:, :, time])
            for u, v in sg.edges_iter():
                if new.nx_graph.has_edge(u, v):
                    new.nx_graph.edge[u][v]['timestamps'].append(time)
                else:
                    new.nx_graph.add_edge(u, v, timestamps=[time], weight=1)
        new.min_date = 0
        new.max_date = 9
        return new
