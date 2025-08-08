import networkx as nx
import matplotlib.pyplot as plt

def build_route_graph(points, connections):
    G = nx.Graph()
    for point in points:
        G.add_node(point)
    for src, dst, weight in connections:
        G.add_edge(src, dst, weight=weight)
    return G

def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')

def plot_graph(graph, path=None):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    if path:
        edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red', width=2)
    plt.show()
