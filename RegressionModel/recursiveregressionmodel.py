import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import copy
import networkx as nx
from collections import deque


class Point:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.rank  = -1


def create_graph():
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "D","E","F"])
    graph.add_edges_from([("A", "B"), ("B", "C"), ("B", "D"), ("B", "E"), ("A", "F"), ("D", "E")])

    return graph


def rotate_point_3D(
    rotate_point: Point, pivot_point: Point, theta_x, theta_y, theta_z
):  # rotate_point and pivot_point are both point objects, theta is the desired rotation in degrees
    temp_array = np.array(
        [
            rotate_point.x - pivot_point.x,
            rotate_point.y - pivot_point.y,
            rotate_point.z - pivot_point.z,
        ]
    )
    # the point is now centered around the origin
    rad_x = np.radians(theta_x)
    rad_y = np.radians(theta_y)
    rad_z = np.radians(theta_z)

    x = [rotate_point.x, pivot_point.x]
    y = [rotate_point.y, pivot_point.y]
    z = [rotate_point.z, pivot_point.z]

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot
    ax.plot(x, y, z, color="blue", linewidth=2)
    ax.scatter(x, y, z)

    # Labels

    transformation_matrix = np.array(
        [
            [
                np.cos(rad_y) * np.cos(rad_z),
                -np.sin(rad_y),
                np.cos(rad_y) * np.sin(rad_z),
            ],
            [
                np.cos(rad_x) * np.sin(rad_y) * np.cos(rad_z)
                + np.sin(rad_x) * np.sin(rad_z),
                np.cos(rad_x) * np.cos(rad_y),
                np.cos(rad_x) * np.sin(rad_y) * np.sin(rad_z)
                - np.sin(rad_x) * np.cos(rad_z),
            ],
            [
                np.sin(rad_x) * np.sin(rad_y) * np.cos(rad_z)
                - np.cos(rad_x) * np.sin(rad_z),
                np.sin(rad_x) * np.cos(rad_y),
                np.sin(rad_x) * np.sin(rad_y) * np.sin(rad_z)
                + np.cos(rad_x) * np.cos(rad_z),
            ],
        ]
    )

    temp_array = transformation_matrix @ temp_array
    rotate_point.x = temp_array[0] + pivot_point.x
    rotate_point.y = temp_array[1] + pivot_point.y
    rotate_point.z = temp_array[2] + pivot_point.z

    x_2 = [rotate_point.x, pivot_point.x]
    y_2 = [rotate_point.y, pivot_point.y]
    z_2 = [rotate_point.z, pivot_point.z]
    ax.plot(x_2, y_2, z_2, color="green", linewidth=2)
    ax.scatter(x_2, y_2, z_2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def align_main_joint(
    goal_point: Point, current_point: Point
):  # instead of constantly shifting, automatically align a "main pivot"
    # this could be the hip for example so that the two hips are overlaid, after this, only rotatiion are necessary

    return

def set_graph_ranks(main_joint, graph):
   
    # use the .popleft() to remove and get the next element
    # once an element is removed the edges are added 
   
    
    distances = nx.single_source_shortest_path_length(graph, source=main_joint)
    print(distances)
    cycles = nx.cycle_basis(graph)
    print(cycles)
        
    
def main():
    graph= create_graph()
    set_graph_ranks("A", graph)
    
    


if __name__ == "__main__":
    main()

