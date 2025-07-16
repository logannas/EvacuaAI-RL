import networkx as nx
import matplotlib.pyplot as plt
import logging


def create_graph(
    edges: tuple[int, int, float],
    positions: dict,
    nodes_exit: list = [],
    nodes_fire: list = [],
):
    """
    Cria e desenha um grafo com nós, arestas, e atributos visuais baseados nos nós de saída e de fogo.

    Args:
        edges (list of tuple): Lista de arestas no formato (nó1, nó2, peso).
        positions (dict): Dicionário com posições dos nós no formato {nó: {"x": valor_x, "y": valor_y}}.
        nodes_exit (list, optional): Lista de nós de saída. Defaults to None.
        nodes_fire (list, optional): Lista de nós em situação de fogo. Defaults to None.

    Returns:
        networkx.Graph: Grafo criado com os nós, arestas e atributos configurados.
    """
    # Inicializa o grafo
    g = nx.Graph()

    # Define posições dos nós
    pos = {int(node): (data["x"], data["y"]) for node, data in positions.items()}
    nx.set_node_attributes(g, pos, "pos")
    g.add_nodes_from(pos.keys())

    # Adiciona as arestas com pesos
    g.add_weighted_edges_from(edges)

    # Define o mapa de cores dos nós
    color_map = [
        "green" if node in nodes_exit else "red" if node in nodes_fire else "blue"
        for node in g
    ]

    # Desenha o grafo
    plt.figure(figsize=(8, 6))
    nx.draw(
        g, pos, node_color=color_map, with_labels=True, edge_color="gray", node_size=500
    )
    plt.gca().invert_yaxis()
    plt.title("Graph Visualization")
    plt.show()
    plt.savefig("graph.jpg")

    # Limpa o buffer do gráfico
    plt.clf()

    # Loga informações do grafo
    logging.info("Total number of nodes: %d", len(g.nodes))
    logging.info("Total number of edges: %d", len(g.edges))

    return g
