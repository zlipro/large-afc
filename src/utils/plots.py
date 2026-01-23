import matplotlib.pyplot as plt

def visualize_graph(graph, node_config):
    """
    Visualizes the graph in 2D or 3D.
    
    Parameters:
        graph: A graph object with edges.
        node_config (numpy.ndarray): A dxn matrix containing node coordinates.
    """
    d, n = node_config.shape
    
    if d == 2:
        # plt.figure()
        for i, j in graph.edges:
            plt.plot([node_config[0, i], node_config[0, j]], 
                     [node_config[1, i], node_config[1, j]], 'k-', zorder=1)
        
        plt.scatter(node_config[0, :], node_config[1, :], c='r', s=100, zorder=2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        # plt.show()
    
    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i, j in graph.edges:
            ax.plot([node_config[0, i], node_config[0, j]], 
                    [node_config[1, i], node_config[1, j]], 
                    [node_config[2, i], node_config[2, j]], 'k-')
        
        ax.scatter(node_config[0, :], node_config[1, :], node_config[2, :], c='r', s=100)
        ax.set_box_aspect([1,1,1])  # Equal aspect ratio
        # plt.axis('off')
        # plt.show()