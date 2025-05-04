import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import copy
import networkx as nx
from collections import deque
import pose

# Some chatgpt used for equations of projection and rotation as well as small editing
memo = {}

class Point:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # def copy(self):
    #     # Use deepcopy to ensure nested objects are copied
    #     return copy.deepcopy(self)


def create_graph(world_coordinates):
    graph = nx.Graph()
    graph.add_nodes_from(
        [i for i in range(11, 33)]
    )  # this removes the head because its not important for now.
    # see https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker for a mapping of points and their edges

    graph.add_edges_from(
        [
            (18, 20),
            (16, 18),
            (16, 20),
            (16, 22),
            (14, 16),
            (12, 14),  # right arm
            (17, 19),
            (15, 17),
            (15, 19),
            (15, 21),
            (13, 15),
            (11, 13),  # left arm
            (11, 12),
            (12, 24),
            (11, 23),
            (23, 24),  # torso
            (24, 26),
            (26, 28),
            (28, 30),
            (28, 32),
            (30, 32),  # right leg
            (23, 25),
            (25, 27),
            (27, 29),
            (27, 31),
            (29, 31),  # left leg
        ]
    )  # there are 26 in total

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
    rank_0 = [11, 12, 24, 23]
    rank_1 = [14, 13, 26, 25]
    rank_2 = [16, 15, 28, 27]
    rank_3 = [22, 20, 18, 21, 19, 17, 32, 30, 29, 31]

    for j in range(11, 33):  # this goes through every joint
        list_of_points = []
        for i in range(world_coordinates.shape[0]):  # this goes through every frame
            list_of_points.append(
                Point(
                    world_coordinates[i, j, 0],
                    world_coordinates[i, j, 1],
                    world_coordinates[i, j, 2],
                )
            )

        graph.nodes[j]["Coords"] = list_of_points
        if j in rank_0:
            graph.nodes[j]["Rank"] = 0
        elif j in rank_1:
            graph.nodes[j]["Rank"] = 1
        elif j in rank_2:
            graph.nodes[j]["Rank"] = 2
        elif j in rank_3:
            graph.nodes[j]["Rank"] = 3

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

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x), np.cos(rad_x)],
        ]
    )

    Ry = np.array(
        [
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)],
        ]
    )

    Rz = np.array(
        [
            [np.cos(rad_z), -np.sin(rad_z), 0],
            [np.sin(rad_z), np.cos(rad_z), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation (XYZ order)
    transformation_matrix = Rz @ Ry @ Rx

    temp_array = transformation_matrix @ temp_array
    rotate_point.x = temp_array[0] + pivot_point.x
    rotate_point.y = temp_array[1] + pivot_point.y
    rotate_point.z = temp_array[2] + pivot_point.z


def determine_angle(base_in: Point, base_out: Point, user_in: Point, user_out: Point):
    line_a = np.array(
        [
            base_out.x - base_in.x,
            base_out.y - base_in.y,
            base_out.z - base_in.z,
        ]
    )

    line_b = np.array(
        [
            user_out.x - user_in.x,
            user_out.y - user_in.y,
            user_out.z - user_in.z,
        ]
    )

    dot_prod = np.dot(line_a, line_b)
    norm_a = np.linalg.norm(line_a)
    norm_b = np.linalg.norm(line_b)

    radian_out = np.arccos((dot_prod / (norm_a * norm_b)))
    return np.degrees(radian_out)


def determine_angle_vector(
    vector: np.array, base_in: Point, base_out: Point, user_in: Point, user_out: Point
):  # this version is used during the actual function
    # vector should always be a unit vector
    line_a = np.array(
        [
            base_out.x - base_in.x,
            base_out.y - base_in.y,
            base_out.z - base_in.z,
        ]
    )

    line_b = np.array(
        [
            user_out.x - user_in.x,
            user_out.y - user_in.y,
            user_out.z - user_in.z,
        ]
    )
    line_a = line_a - np.dot(line_a, vector) * vector
    line_b = line_b - np.dot(line_b, vector) * vector

    dot_prod = np.dot(line_a, line_b)
    norm_a = np.linalg.norm(line_a)
    norm_b = np.linalg.norm(line_b)

    radian_out = np.arccos((dot_prod / (norm_a * norm_b)))
    cross = np.cross(line_a, line_b)
    reference_axis = np.array([0, 1, 0])
    sign = np.sign(np.dot(cross, reference_axis))
    if sign < 0:
        radian_out *= -1
    return -np.degrees(
        radian_out
    )  # weird sign output so that it aligns with the rotate point function


def get_thetas(
    world_coord_base, world_coord_user, i, pivot_num: int, rotate_num: int
):  # for rotation, i is for what frame, pivot_num is number being rotated around, rotate_num is number being rotated
    first = Point(0, 0, 0)
    second = Point(
        world_coord_base[i, rotate_num, 0] - world_coord_base[i, pivot_num, 0],
        world_coord_base[i, rotate_num, 1] - world_coord_base[i, pivot_num, 1],
        world_coord_base[i, rotate_num, 2] - world_coord_base[i, pivot_num, 2],
    )
    mybasis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    third = Point(0, 0, 0)
    fourth = Point(
        world_coord_user[i, rotate_num, 0] - world_coord_user[i, pivot_num, 0],
        world_coord_user[i, rotate_num, 1] - world_coord_user[i, pivot_num, 1],
        world_coord_user[i, rotate_num, 2] - world_coord_user[i, pivot_num, 2],
    )

    temp = Point(
        world_coord_user[i, rotate_num, 0] - world_coord_user[i, pivot_num, 0],
        world_coord_user[i, rotate_num, 1] - world_coord_user[i, pivot_num, 1],
        world_coord_user[i, rotate_num, 2] - world_coord_user[i, pivot_num, 2],
    )
    theta_x = determine_angle_vector(mybasis[:, 0], first, second, third, fourth)

    rotate_point_3D(temp, first, theta_x, 0, 0)
    theta_y = determine_angle_vector(mybasis[:, 1], first, second, third, temp)

    rotate_point_3D(temp, first, 0, theta_y, 0)
    theta_z = determine_angle_vector(mybasis[:, 2], first, second, third, temp)
    return theta_x, theta_y, theta_z


def transform_coords(
    world_coord_base, world_coord_user
):  # the final frame count will be the minimum of the two frame counts
    new_base = np.zeros(
        (min(world_coord_base.shape[0], world_coord_user.shape[0]), 33, 3)
    )
    new_user = np.zeros(
        (min(world_coord_base.shape[0], world_coord_user.shape[0]), 33, 3)
    )
    for i in range(min(world_coord_base.shape[0], world_coord_user.shape[0])):
        theta_x, theta_y, theta_z = get_thetas(
            world_coord_base, world_coord_user, i, 24, 23
        )
        rotate_point = Point(0, 0, 0)

        for j in range(33):  # num joints
            curpoint = Point(
                world_coord_user[i, j, 0] - world_coord_user[i, 24, 0],
                world_coord_user[i, j, 1] - world_coord_user[i, 24, 1],
                world_coord_user[i, j, 2] - world_coord_user[i, 24, 2],
            )
            rotate_point_3D(curpoint, rotate_point, theta_x, theta_y, theta_z)
            new_user[i, j, :] = np.array([curpoint.x, curpoint.y, curpoint.z])

            new_base[i, j, :] = np.array(
                [
                    world_coord_base[i, j, 0] - world_coord_base[i, 24, 0],
                    world_coord_base[i, j, 1] - world_coord_base[i, 24, 1],
                    world_coord_base[i, j, 2] - world_coord_base[i, 24, 2],
                ]
            )

    return new_base, new_user


def get_dist(
    graph_base, graph_user, iter
):  # graph_base and graph_user are both graphs made with network x;;; iter is which frame
    nodes = list(graph_base.nodes())
    total_dist = 0
    for node in nodes:
        base_point = graph_base.nodes[node]["Coords"][iter]
        user_point = graph_user.nodes[node]["Coords"][iter]
        total_dist += (
            (base_point.x - user_point.x) * (base_point.x - user_point.x)
            + (base_point.y - user_point.y) * (base_point.y - user_point.y)
            + (base_point.z - user_point.z) * (base_point.z - user_point.z)
        )

    return total_dist


def compressed_dist_and_rotate(
    world_coord_base,
    temp_graph,
    i,
    theta_x,
    theta_y,
    theta_z,
    rotate_point: int,
    to_be_rotated: list,
):  # smaller function to save unnecessary overhead does the rotation and returns the distance, this returns the difference from only the joints that are moved, so the result can be negative
    initial_dist = 0
    mock_points = []
    # calculate the initial distance and create mock points
    for r in to_be_rotated:
        initial_dist += (
            (world_coord_base[i, r, 0] - temp_graph.nodes[r]["Coords"][i].x)
            * (world_coord_base[i, r, 0] - temp_graph.nodes[r]["Coords"][i].x)
            + (world_coord_base[i, r, 1] - temp_graph.nodes[r]["Coords"][i].y)
            * (world_coord_base[i, r, 1] - temp_graph.nodes[r]["Coords"][i].y)
            + (world_coord_base[i, r, 2] - temp_graph.nodes[r]["Coords"][i].z)
            * (world_coord_base[i, r, 2] - temp_graph.nodes[r]["Coords"][i].z)
        )
        mock_points.append(
            Point(
                temp_graph.nodes[r]["Coords"][i].x,
                temp_graph.nodes[r]["Coords"][i].y,
                temp_graph.nodes[r]["Coords"][i].z,
            )
        )

    r_point = Point(
        temp_graph.nodes[rotate_point]["Coords"][i].x,
        temp_graph.nodes[rotate_point]["Coords"][i].y,
        temp_graph.nodes[rotate_point]["Coords"][i].z,
    )
    new_dist = 0
    for p, r in zip(mock_points, to_be_rotated):
        rotate_point_3D(p, r_point, theta_x, theta_y, theta_z)
        new_dist += (
            (world_coord_base[i, r, 0] - p.x) * (world_coord_base[i, r, 0] - p.x)
            + (world_coord_base[i, r, 1] - p.y) * (world_coord_base[i, r, 1] - p.y)
            + (world_coord_base[i, r, 2] - p.z) * (world_coord_base[i, r, 2] - p.z)
        )

    return initial_dist - new_dist  # returns the difference


def iterative_rotation(
    world_coord_base,
    world_coord_user,
    i,
    graph_base,
    graph_user,
    rotate_point: int,
    specificity,
):  # i is for iter
    # construct what should be rotated

    
    to_be_rotated = []
    if rotate_point in memo: 
        to_be_rotated = memo[rotate_point]
    else:   
        current_rank = graph_user.nodes[rotate_point]["Rank"]
        friends = list(graph_user.neighbors(rotate_point))
        q = deque()  # this acts as a queue
        for f in friends:
            q.append(f)

        while q:
            current = q.popleft()
            if graph_user.nodes[current]["Rank"] > current_rank:
                if current not in to_be_rotated:
                    to_be_rotated.append(current)
                homies = list(graph_user.neighbors(current))
                homies = [
                    homie for homie in homies if homie not in to_be_rotated
                ]  # homie = neighbor
                for homie in homies:
                    q.append(homie)
        memo.setdefault(rotate_point, to_be_rotated)
    # print(to_be_rotated)
    # def get_thetas(world_coord_base, world_coord_user, i, pivot_num: int, rotate_num: int):
    temp_graph = copy.deepcopy(graph_user)
    r_point = temp_graph.nodes[rotate_point]["Coords"][i]

    theta_x = 0
    step = 50
    best_x = theta_x
    best_val = 0
    prev_val = 0
    # compressed_dist_and_rotate(world_coord_base, world_coord_user,i,theta_x, theta_y, theta_z, rotate_point: int, to_be_rotated: list)
    for j in range(specificity):
        theta_x += step
        current_val = compressed_dist_and_rotate(
            world_coord_base, temp_graph, i, theta_x, 0, 0, rotate_point, to_be_rotated
        )

        if current_val > best_val:  # if improved best
            best_val = current_val
            best_x = theta_x
        if current_val < prev_val:  # if worse than last time
            step *= -1 / 2
        if step < 1 / 8 and step > -1 / 8:
            step *= 8

        prev_val = current_val

    theta_x = best_x
    for point in to_be_rotated:
        rotate_point_3D(temp_graph.nodes[point]["Coords"][i], r_point, theta_x, 0, 0)

    theta_y = 0
    step = 50
    best_y = theta_y
    best_val = 0
    prev_val = 0
    # compressed_dist_and_rotate(world_coord_base, world_coord_user, i, theta_x, theta_y, theta_z, rotate_point: int, to_be_rotated: list)
    for j in range(specificity):
        theta_y += step
        current_val = compressed_dist_and_rotate(
            world_coord_base, temp_graph, i, 0, theta_y, 0, rotate_point, to_be_rotated
        )

        if current_val > best_val:  # if improved best
            best_val = current_val
            best_y = theta_y
        if current_val < prev_val:  # if worse than last time
            step *= -1 / 2
        if step < 1 / 8 and step > -1 / 8:
            step *= 8

        prev_val = current_val

    theta_y = best_y
    for point in to_be_rotated:
        rotate_point_3D(temp_graph.nodes[point]["Coords"][i], r_point, 0, theta_y, 0)

    theta_z = 0
    step = 50
    best_z = theta_z
    best_val = 0
    prev_val = 0
    # compressed_dist_and_rotate(world_coord_base, world_coord_user, i, theta_x, theta_y, theta_z, rotate_point: int, to_be_rotated: list)
    for j in range(specificity):
        theta_z += step
        current_val = compressed_dist_and_rotate(
            world_coord_base, temp_graph, i, 0, 0, theta_z, rotate_point, to_be_rotated
        )

        if current_val > best_val:  # if improved best
            best_val = current_val
            best_z = theta_z
        if current_val < prev_val:  # if worse than last time
            step *= -1 / 2
        if step < 1 / 8 and step > -1 / 8:
            step *= 8

        prev_val = current_val

    theta_z = best_z
    for point in to_be_rotated:
        rotate_point_3D(temp_graph.nodes[point]["Coords"][i], r_point, 0, 0, theta_z)

    # print("theta_x" + str(theta_x))
    # print("theta_y" + str(theta_y))
    # print("theta_z" + str(theta_z))

    # plot_compare_graph(
    #     graph_user, temp_graph, graph_base, i + 1
    # )  # 1 red, 2 blue, 3 green

    return get_dist(graph_base, temp_graph, i), theta_x, theta_y, theta_z, rotate_point


def every_rotation(
    world_coord_base, world_coord_user, i, graph_base, graph_user, specificity
):  # goes through all the rotation possibilities for a given frame
    nodes = list(graph_base.nodes())
    rank_3 = [
        22,
        20,
        18,
        21,
        19,
        17,
        32,
        30,
        29,
        31,
    ]  # remove things that cant be rotated around - fingertips
    nodes = [node for node in nodes if node not in rank_3]
    max_finder = []
    for n in nodes:
        value = iterative_rotation(
            world_coord_base,
            world_coord_user,
            i,
            graph_base,
            graph_user,
            n,
            specificity,
        )
        max_finder.append(value)
    best = min(max_finder, key=lambda x: x[0])

    output = generate_string(best)
    return output, best


def frame_iterator(
    world_coord_base, world_coord_user, graph_base, graph_user, frames, specificity
):  # returns a string for each frame on what to do
    string_list = []
    output_list = []
    for f in range(frames):
        print(f)
        total_output = every_rotation(
                world_coord_base,
                world_coord_user,
                f,
                graph_base,
                graph_user,
                specificity,
            )
        string_list.append(total_output[0])
        output_list.append(total_output[1])

    return string_list,output_list


def generate_string(
    info: tuple,
):  # creates the string output tuple contains dist, theta_x, theta_y, theta_z, node
    body_parts = {
        0: "nose",
        1: "left eye (inner)",
        2: "left eye",
        3: "left eye (outer)",
        4: "right eye (inner)",
        5: "right eye",
        6: "right eye (outer)",
        7: "left ear",
        8: "right ear",
        9: "mouth (left)",
        10: "mouth (right)",
        11: "left shoulder",
        12: "right shoulder",
        13: "left elbow",
        14: "right elbow",
        15: "left wrist",
        16: "right wrist",
        17: "left pinky",
        18: "right pinky",
        19: "left index",
        20: "right index",
        21: "left thumb",
        22: "right thumb",
        23: "left hip",
        24: "right hip",
        25: "left knee",
        26: "right knee",
        27: "left ankle",
        28: "right ankle",
        29: "left heel",
        30: "right heel",
        31: "left foot index",
        32: "right foot index",
    }

    return (
        "Rotate around "
        + str(body_parts[info[4]])
        + " joint "
        + "X-axis: "
        + str(info[1])
        + " degrees, Y-axis: "
        + str(info[2])
        + " degrees, Z-axis: "
        + str(info[3])
        + " degrees"
    )


def plot_compare_graph(graph_base, graph_user, graph_other, frames, strings):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(frames):
        for edge in graph_base.edges():

            base_point1 = graph_base.nodes[edge[0]]["Coords"][i]
            base_point2 = graph_base.nodes[edge[1]]["Coords"][i]

            user_point1 = graph_user.nodes[edge[0]]["Coords"][i]
            user_point2 = graph_user.nodes[edge[1]]["Coords"][i]

            other_point1 = graph_other.nodes[edge[0]]["Coords"][i]
            other_point2 = graph_other.nodes[edge[1]]["Coords"][i]

            x_user = [user_point1.x, user_point2.x]
            y_user = [user_point1.y, user_point2.y]
            z_user = [user_point1.z, user_point2.z]
            ax.plot(x_user, y_user, z_user, color="blue")

            x_base = [base_point1.x, base_point2.x]
            y_base = [base_point1.y, base_point2.y]
            z_base = [base_point1.z, base_point2.z]
            ax.plot(x_base, y_base, z_base, color="red")

            x_other = [other_point1.x, other_point2.x]
            y_other = [other_point1.y, other_point2.y]
            z_other = [other_point1.z, other_point2.z]
            ax.plot(x_other, y_other, z_other, color="green")

        ax.set_title(strings[i])

        plt.draw()
        plt.pause(.1)
        plt.cla()
    plt.show()


def plot_output(world_coord_base: np.array, world_coord_user: np.array, strings, outputs: list):
    # return get_dist(graph_base, temp_graph, i), theta_x, theta_y, theta_z, rotate_point ---- this is what outputs has
    new_base, new_user = transform_coords(world_coord_base, world_coord_user)
    graph_base = create_graph(new_base)
    graph_user = create_graph(new_user)

    graph_temp = copy.deepcopy(graph_user)

    for o, i in zip(outputs, range(new_base.shape[0])):
        rotate_point = o[4]
        to_be_rotated = []
        rpoint = graph_temp.nodes[o[4]]['Coords'][i]
        if rotate_point in memo: 
            to_be_rotated = memo[rotate_point]
        else:   
            current_rank = graph_user.nodes[rotate_point]["Rank"]
            friends = list(graph_user.neighbors(rotate_point))
            q = deque()  # this acts as a queue
            for f in friends:
                q.append(f)

            while q:
                current = q.popleft()
                if graph_user.nodes[current]["Rank"] > current_rank:
                    if current not in to_be_rotated:
                        to_be_rotated.append(current)
                    homies = list(graph_user.neighbors(current))
                    homies = [
                        homie for homie in homies if homie not in to_be_rotated
                    ]  # homie = neighbor
                    for homie in homies:
                        q.append(homie)
        memo.setdefault(rotate_point, to_be_rotated)

        for r in to_be_rotated:
            rotate_point_3D(graph_temp.nodes[r]['Coords'][i], rpoint, o[1],o[2],o[3])

    plot_compare_graph(graph_user, graph_temp, graph_base, new_base.shape[0], strings) # 1 red, 2 blue, 3 green


def actual_model(video_base: str, video_user: str): # CALLLLLLLLLLLLL THISSSSSSSSSSS it takes two strings which are the addresses to the mp4s 
                                                    #and returns a list of strings and a list of outputs which have the best dist, theta_x, theta_y, theta_z and best nodes
    plt.close("all")
    world_coord_base, edges_array_base = pose.process_frame_without_video_output(video_base)
    world_coord_user, edges_array_user = pose.process_frame_without_video_output(video_user)

    new_base, new_user = transform_coords(world_coord_base, world_coord_user)
    graph_base = create_graph(new_base)
    graph_user = create_graph(new_user)

    strings,outputs = frame_iterator(
        new_base, new_user, graph_base, graph_user, new_base.shape[0], 20
    )
    
    return strings, outputs # this whole thing takes like 5-6 minutes to run
        # return get_dist(graph_base, temp_graph, i), theta_x, theta_y, theta_z, rotate_point ---- this is what outputs has

def actual_model_modified(world_coord_base, world_coord_user): # CALLLLLLLLLLLLL THISSSSSSSSSSS it takes two strings which are the addresses to the mp4s 
                                                    #and returns a list of strings and a list of outputs which have the best dist, theta_x, theta_y, theta_z and best nodes
    plt.close("all")

    new_base, new_user = transform_coords(world_coord_base, world_coord_user)
    graph_base = create_graph(new_base)
    graph_user = create_graph(new_user)

    strings,outputs = frame_iterator(
        new_base, new_user, graph_base, graph_user, new_base.shape[0], 20
    )
    
    return strings, outputs # this whole thing takes like 5-6 minutes to run
        # return get_dist(graph_base, temp_graph, i), theta_x, theta_y, theta_z, rotate_point ---- this is what outputs has

def fake_actual_model(): # for testing purposes - preloads the numpy arrays from soohwans model so that this process doesn't have to occur every time. 
    load_numpy_array_base = np.load(r"team-82-FormCorrect\uploads\GuySquatArray.npz")
    load_numpy_array_user = np.load(r"team-82-FormCorrect\uploads\BuffSquatArray.npz")

    world_coord_base = load_numpy_array_base["world"]
    world_coord_user = load_numpy_array_user["world"]

    new_base, new_user = transform_coords(world_coord_base, world_coord_user)
    graph_base = create_graph(new_base)
    graph_user = create_graph(new_user)

    strings,outputs = frame_iterator(
        new_base, new_user, graph_base, graph_user, new_base.shape[0], 20
    )
    
    return strings, outputs, new_base, new_user # this whole thing takes like 5-6 minutes to run
        # return get_dist(graph_base, temp_graph, i), theta_x, theta_y, theta_z, rotate_point ---- this is what outputs has

# def load_video_to_npz(video_file_base: str, video_file_user: str):
#     world_coord_base, edges_array_base = pose.process_frame_without_video_output(video_file_base)
#     world_coord_user, edges_array_user = pose.process_frame_without_video_output(video_file_user)

#     np.savez(r'team-82-FormCorrect\uploads\GuySquatArray.npz', world = world_coord_base, edge = edges_array_base)
#     np.savez(r'team-82-FormCorrect\uploads\BuffSquatArray.npz', world = world_coord_user, edge = edges_array_user)

def main():

    plt.close("all")  # closes soohwans stuff

    video_file_base = r"team-82-FormCorrect\uploads\SomeGuySquatting.mp4"
    video_file_user = r"team-82-FormCorrect\uploads\BuffGuySquatting.mp4"
    # video_file = "team-82-FormCorrect\GuyMovingArms.mp4"

    
    strings, outputs, new_base, new_user = fake_actual_model()

    print(strings)
    print(outputs)

    plot_output(new_base, new_user, strings, outputs)
    # has names world and edge
    

    # def create_graph(world_coordinates):
    # graph= create_graph()
    # set_graph_ranks("A", graph)
    

    # nx.draw(graph_base, with_labels=True, node_color="skyblue", node_size=2000, font_size=15)
    # plt.show()

    # def recursive_rotation(world_coord_base, world_coord_user, i, graph_base, graph_user, rotate_point: int):

    # saved this to file so wouldn't have to run every time.
    
    


if __name__ == "__main__":
    main()
