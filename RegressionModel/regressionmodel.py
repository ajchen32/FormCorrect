import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation
import math


def plot_initial_end(x_goal, y_goal, x_cor, y_cor, x_end, y_end, iterations):
    plt.plot(x_goal[0:2], y_goal[0:2], marker="o", linestyle="-", color="k")
    plt.plot(x_goal[1:3], y_goal[1:3], marker="o", linestyle="-", color="k")
    plt.plot(x_cor[0:2], y_cor[0:2], marker="o", linestyle="-", color="g")
    plt.plot(x_cor[1:3], y_cor[1:3], marker="o", linestyle="-", color="g")
    plt.plot(x_end[0:2], y_end[0:2], marker="o", linestyle="-", color="r")
    plt.plot(x_end[1:3], y_end[1:3], marker="o", linestyle="-", color="r")

    plt.title(
        "Goal: Black, Initial: Green, End: Red (" + str(iterations) + " iterations)"
    )
    init = calc_dist(x_goal, y_goal, x_cor, y_cor)
    end = calc_dist(x_goal, y_goal, x_end, y_end)
    plt.xlabel(
        "Distances from user to goal in units^2 -> Init: "
        + str(init)
        + " End: "
        + str(end)
    )

    plt.savefig("team-82-FormCorrect/RegressionModel/regressionplot.png")
    plt.show()


def plot_points(
    x_goal, y_goal, x_cor, y_cor
):  # plot the goal points side by side with the current points
    plt.plot(x_goal[0:2], y_goal[0:2], marker="o", linestyle="-", color="k")
    plt.plot(x_goal[1:3], y_goal[1:3], marker="o", linestyle="-", color="k")
    plt.plot(x_cor[0:2], y_cor[0:2], marker="o", linestyle="-", color="g")
    plt.plot(x_cor[1:3], y_cor[1:3], marker="o", linestyle="-", color="g")
    plt.show()


def calc_dist(
    x_goal, y_goal, x_cor, y_cor
):  # calculate distance between each point and its relative goal point and sum the distances
    # the distances remain in their "squared form"

    x_val = x_goal - x_cor
    y_val = y_goal - y_cor
    x_dist = x_val * x_val
    y_dist = y_val * y_val
    total_dist = x_dist + y_dist

    return np.sum(total_dist)


def rotate_point(
    x_cor, y_cor, theta, pivot_idx, move_idx
):  # rotates the "move" point around the "pivot" point by theta
    # must pass function np array so that the original is modified
    # formulas
    # x′=xc​+(x−xc​)⋅cos(θ)−(y−yc​)⋅sin(θ)
    # y′=yc+(x−xc)⋅sin⁡(θ)+(y−yc)⋅cos⁡(θ)
    radians = math.radians(theta)
    temp = (
        x_cor[pivot_idx]
        + (x_cor[move_idx] - x_cor[pivot_idx]) * math.cos(radians)
        - (y_cor[move_idx] - y_cor[pivot_idx]) * math.sin(radians)
    )
    y_cor[move_idx] = (
        y_cor[pivot_idx]
        + (x_cor[move_idx] - x_cor[pivot_idx]) * math.sin(radians)
        + (y_cor[move_idx] - y_cor[pivot_idx]) * math.cos(radians)
    )
    x_cor[move_idx] = temp


def shift_points(
    x_cor, y_cor, x_samt, y_samt
):  # shifts all x_cor and y_cor by shift amount (samt)
    x_cor += x_samt
    y_cor += y_samt


def run_regression(x_goal, y_goal, x_cor, y_cor, rounds, change_array):

    # plot initial

    best_dist = calc_dist(
        x_goal, y_goal, x_cor, y_cor
    )  # looks for the best minimal distance between goal and actual coordinates
    num_commands = 4
    current_samt = change_array[0]  # starting amount that each shift takes
    current_ramt = change_array[
        1
    ]  # starting amount that each rotation completes in degrees

    record_changes = ""

    for i in range(rounds):

        did_shift = False  # checks if a shift was completed - whether or not the ramt needs to be decreased
        did_rotate = False  # checks if a rotate was completed
        current_change = "No change."

        for j in range(num_commands):
            for k in [-1, 1]:
                change_amount = current_samt
                if j == 0 or j == 1:
                    change_amount = current_ramt

                temp_return = regression_helper(
                    x_goal, y_goal, x_cor, y_cor, best_dist, k * change_amount, j
                )

                if best_dist != temp_return[1]:
                    best_dist = temp_return[1]
                    current_change = temp_return[0]
                    if j == 0 or j == 1:
                        did_rotate = True
                    else:
                        did_shift = True
        if current_change != "No change.":
            record_changes += str(i) + ". " + current_change + "\n"

        if not did_rotate:
            current_ramt -= 1
            if current_ramt < -100:
                current_ramt = 180

        if not did_shift:
            current_samt -= 1
            if current_samt < -180:
                current_samt = 100

    # with open(
    #     "team-82-FormCorrect/RegressionModel/regressionmodeloutput.txt", "w"
    # ) as file:
    #     file.write(record_changes)
    change_array[0] = current_samt
    change_array[1] = current_ramt


def regression_helper(
    x_goal, y_goal, x_cor, y_cor, best_dist, change_amount, command
):  # change_amount can be either degree or shift amount
    # command variable indicates what type of shift is occurring
    x_temp = x_cor.copy()
    y_temp = y_cor.copy()
    current_change = command_helper(x_temp, y_temp, change_amount, command)

    temp_dist = calc_dist(x_goal, y_goal, x_temp, y_temp)
    if temp_dist < best_dist:
        best_dist = temp_dist

        x_cor[:] = x_temp
        y_cor[:] = y_temp

    return current_change, best_dist


def command_helper(
    x_temp, y_temp, change_amount, command
):  # modifies x_temp and y_temp
    command_list_names = [
        "Rotate point 0 around point 1 by " + str(change_amount) + " degrees.",
        "Rotate point 2 around point 1 by " + str(change_amount) + " degrees.",
        "Shift structure by " + str(change_amount) + " on the x_axis.",
        "Shift structure by " + str(change_amount) + " on the y_axis.",
    ]

    rotate_0 = partial(
        rotate_point, theta=change_amount, pivot_idx=1, move_idx=0
    )  # command 0
    rotate_2 = partial(
        rotate_point, theta=change_amount, pivot_idx=1, move_idx=2
    )  # command 1
    shift_x = partial(shift_points, x_samt=change_amount, y_samt=0)  # command 2
    shift_y = partial(shift_points, x_samt=0, y_samt=change_amount)  # command 3

    all_the_commands = [rotate_0, rotate_2, shift_x, shift_y]
    all_the_commands[command](x_temp, y_temp)

    return command_list_names[command]


def main():

    # initialize the points
    grid_dimension = 1000
    # goal
    x_goal = np.random.randint(0, grid_dimension, 3).astype(float)
    y_goal = np.random.randint(0, grid_dimension, 3).astype(float)

    # user data
    x_cor = np.random.randint(0, grid_dimension, 3).astype(float)
    y_cor = np.random.randint(0, grid_dimension, 3).astype(float)

    x_begin = x_cor.copy()
    y_begin = y_cor.copy()

    iterations = 1000
    change_array = np.array([180, 180], dtype=float)
    run_regression(x_goal, y_goal, x_cor, y_cor, iterations, change_array)
    plot_initial_end(x_goal, y_goal, x_begin, y_begin, x_cor, y_cor, iterations)


if __name__ == "__main__":
    main()


# rotate right, #rotate left, shift right, shift up, shift left, shift down

# create a regression model that
