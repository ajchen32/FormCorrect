import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import regressionmodel as rm

grid_dimension = 1000
# goal
x_goal = np.random.randint(0, grid_dimension, 3).astype(float)
y_goal = np.random.randint(0, grid_dimension, 3).astype(float)

# user data
x_cor = np.random.randint(0, grid_dimension, 3).astype(float)
y_cor = np.random.randint(0, grid_dimension, 3).astype(float)

change_array = np.array([180, 180], dtype=float)

fig, ax = plt.subplots()
(goalline,) = ax.plot([], [], lw=2, label="Goal Line", color="blue")
(currentline,) = ax.plot([], [], lw=2, label="Current Line", color="green")
(startline,) = ax.plot([], [], lw=2, label="Start Line", color="red")
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)

ax.legend()


def init(): # resets frames

    goalline.set_data([], [])
    currentline.set_data([], [])
    return goalline, currentline  # Must return all lines as a tuple


def update(frame, framerate): # gives new frame
    rm.run_regression(x_goal, y_goal, x_cor, y_cor, framerate, change_array)
    goalline.set_data(x_goal, y_goal)
    currentline.set_data(x_cor, y_cor)

    ax.set_xlabel(str(frame))

    return goalline, currentline


def main(): # generates animation
    startline.set_data(x_cor, y_cor)

    framerate = 1

    animation = FuncAnimation(
        fig,
        update,
        frames=range(1000),
        init_func=init,
        blit=True,
        interval=20,
        fargs=(framerate,),
    )

    
    plt.show()


if __name__ == "__main__":
    main()
