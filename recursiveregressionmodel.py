import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import copy
import networkx as nx
from collections import deque
import pose


class Point:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        


def create_graph(world_coordinates):
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(11, 33)]) # this removes the head because its not important for now.
    # see https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker for a mapping of points and their edges
    
    graph.add_edges_from([
    (18, 20), (16, 18), (16, 20), (16, 22), (14, 16), (12, 14),  # right arm
    (17, 19), (15, 17), (15, 19), (15, 21), (13, 15), (11, 13),  # left arm
    (11, 12), (12, 24), (11, 23), (23, 24),  # torso
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # right leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31)  # left leg
])  # there are 26 in total
    
    # for reference: For simplicity sake we decided to remove the head
# 0 - nose
# 1 - left eye (inner), 2 - left eye, 3 - left eye (outer), 4 - right eye (inner),
# 5 - right eye, 6 - right eye (outer), 7 - left ear, 8 - right ear,
# 9 - mouth (left), 10 - mouth (right), 11 - left shoulder, 12 - right shoulder, 
# 13 - left elbow, 14 - right elbow, 15 - left wrist, 16 - right wrist,
# 17 - left pinky, 18 - right pinky, 19 - left index, 20 - right index
# 21 - left thumb, 22 - right thumb, 23 - left hip, 24 - right hip
# 25 - left knee, 26 - right knee, 27 - left ankle, 28 - right ankle
# 29 - left heel, 30 - right heel, 31 - left foot index, 32 - right foot index

    
    # there are 4 ranks in total, points that are in a cycle all have the same rank, the torso has a rank of 0 
    # and each joint that progressively goes out has a rank of prev + 1
    rank_0 = [11,12,24,23]
    rank_1 = [14,13,26,25]
    rank_2 = [16,15,28,27]
    rank_3 = [22,20,18,21,19,17,32,30,29,31]
    
    for j in range(11, 33): # this goes through every joint
        list_of_points = []
        for i in range(world_coordinates.shape[0]): # this goes through every frame
            list_of_points.append(Point(world_coordinates[i,j,0], world_coordinates[i,j,1], world_coordinates[i,j,2]))

        graph.nodes[j]['Coords'] = list_of_points
        if j in rank_0:
            graph.nodes[j]['Rank'] = 0     
        elif j in rank_1:
            graph.nodes[j]['Rank'] = 1
        elif j in rank_2:
            graph.nodes[j]['Rank'] = 2
        elif j in rank_3:
            graph.nodes[j]['Rank'] = 3

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

    
def main():
    
    plt.close('all')

    video_file_base = "team-82-FormCorrect\SomeGuySquatting.mp4"
    video_file_user = "team-82-FormCorrect\BuffGuySquatting.mp4"
    #video_file = "team-82-FormCorrect\GuyMovingArms.mp4"
    
    load_numpy_array_base = np.load('team-82-FormCorrect\GuySquatArray.npz')
    load_numpy_array_user = np.load('team-82-FormCorrect\BuffSquatArray.npz')
    
    #has names world and edge
    world_coord_base = load_numpy_array_base['world']
    edges_array_base = load_numpy_array_base['edge']
    world_coord_user = load_numpy_array_user['world']
    edges_array_user = load_numpy_array_user['edge']

    # def create_graph(world_coordinates):
    # graph= create_graph()
    # set_graph_ranks("A", graph)
    graph_base = create_graph(world_coord_base)
    graph_user = create_graph(world_coord_user)

    nx.draw(graph_base, with_labels=True, node_color="skyblue", node_size=2000, font_size=15)
    plt.show()

    
    print(graph_base)
    print(graph_user)

    # saved this to file so wouldn't have to run every time.
    # world_coord_base, edges_array_base = pose.process_frame_without_video_output(video_file_base)
    # world_coord_user, edges_array_user = pose.process_frame_without_video_output(video_file_user)
    
    # np.savez('team-82-FormCorrect\GuySquatArray.npz', world = world_coord_base, edge = edges_array_base)
    # np.savez('team-82-FormCorrect\BuffSquatArray.npz', world = world_coord_user, edge = edges_array_user)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i in range(0,edges_array_base.shape[0]):
        for j in range(0, edges_array_base.shape[1]):
            
            x = edges_array_base[i,j, :, 0] - world_coord_base[i,12,0]
            y = edges_array_base[i,j, :, 1] - world_coord_base[i,12,1]
            z = edges_array_base[i,j, :, 2] - world_coord_base[i,12,2]
            ax.plot(x,y,z, color = 'blue')
            x1 = edges_array_user[i+1,j, :, 0] - world_coord_user[i,12,0]
            y1 = edges_array_user[i+1,j, :, 1] - world_coord_user[i,12,1]
            z1 = edges_array_user[i+1,j, :, 2] - world_coord_user[i,12,2]
            ax.plot(x1,y1,z1, color = 'red')

        plt.draw()
        plt.pause(.1)
        plt.cla()
    plt.show() 
      
        




if __name__ == "__main__":
    main()


# world_coord_array = np.array(world_coord_list)
#     edges_array = np.array(edges_array)
#     print("Edge array shape:", edges_array.shape)
#     # np.save("pose_edges.npy", edges_array)
#     cap.release()
#     cv2.destroyAllWindows()
#     return world_coord_array, edges_array