# some packages are necessary to run the program.
# pip install numpy
# pip install plotly
# pip install nbformat # Mime type rendering requires nbformat>=4.2.0
# pip install matplotlib

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# define constants
rho = 28  # scaled Rayleigh number
sigma = 10  # Prandtl number
beta = 8 / 3  # geometry aspect ratio


def main():
    initial_state = [0, 1, 20]
    trajectory_eulers = eulers_method(initial_state, 25000, 0.001)
    trajectory_heuns = heuns_method(initial_state, 25000, 0.001)
    trajectory_rk4 = rk4_method(initial_state, 25000, 0.001)
    time_pt1 = 1000
    vis_diff_methods(trajectory_eulers, trajectory_heuns, trajectory_rk4, time_pt1)
    time_pt2 = 12000
    vis_diff_methods(trajectory_eulers, trajectory_heuns, trajectory_rk4, time_pt2)
    time_pt3 = 24000
    vis_diff_methods(trajectory_eulers, trajectory_heuns, trajectory_rk4, time_pt3)

    i_state_1 = [1, 1, 1]
    i_state_2 = [1.5, 1.5, 1.5]
    state = [i_state_1, i_state_2]

    trj1 = eulers_method(i_state_1, 10000, 0.005)
    trj2 = eulers_method(i_state_2, 10000, 0.005)
    trjs = [trj1, trj2]
    vis_diff_states(trjs, "euler", 2000, state)

    plot_poincare("x", "z", "y", 0, 2, 1, trajectory_rk4)
    plot_poincare("x", "y", "z", 0, 1, 2, trajectory_rk4)
    plot_poincare("y", "z", "x", 1, 2, 0, trajectory_rk4)


# define lorenz system
def lorenz(x, y, z):
    dx = sigma * (y - x)
    dy = rho * x - y - x * z
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def eulers_method(initial_state, num_time_pts, h):
    """Solves first order non-linear differential equation systems with Euler's method.
    Args:
        initial_state: initial x,y,z values
        num_time_pts: number of time points to plot
        h: next_value - current_value
    Returns:
        numpy array: trajectory of lorenz attractor
    """
    # trajectory = [[x0,x1,x2,...], [y1,y2,y3],...], [z1,z2,z3,...]]
    trajectory = np.zeros((3, num_time_pts))  # trajectory matrix
    # x, y, z = initial_values[0], initial_values[1], initial_values[2]
    x, y, z = initial_state
    for i in range(0, num_time_pts):
        trajectory[0, i] = x
        trajectory[1, i] = y
        trajectory[2, i] = z

        x_next = x + h * lorenz(x, y, z)[0]
        y_next = y + h * lorenz(x, y, z)[1]
        z_next = z + h * lorenz(x, y, z)[2]
        x, y, z = x_next, y_next, z_next
    return trajectory


def heuns_method(initial_state, num_time_pts, h):
    """Solves first order non-linear differential equation systems with Heun's method.
    Args:
        initial_state: initial x,y,z values
        num_time_pts: number of time points to plot
        h: next_value - current_value
    Returns:
        numpy array: trajectory of lorenz attractor
    """
    x, y, z = initial_state
    trajectory = np.zeros((3, num_time_pts))
    for i in range(0, num_time_pts):
        trajectory[0, i] = x
        trajectory[1, i] = y
        trajectory[2, i] = z

        # calculate mid points
        x_mid = x + h * lorenz(x, y, z)[0]
        y_mid = y + h * lorenz(x, y, z)[1]
        z_mid = z + h * lorenz(x, y, z)[2]

        # calculate slopes at initial points
        s_0x = lorenz(x, y, z)[0]
        s_0y = lorenz(x, y, z)[1]
        s_0z = lorenz(x, y, z)[2]

        # calculate slopes at midpoints
        s_mx = lorenz(x_mid, y_mid, z_mid)[0]
        s_my = lorenz(x_mid, y_mid, z_mid)[1]
        s_mz = lorenz(x_mid, y_mid, z_mid)[2]

        # update x,y,z points
        x_next = x + (h / 2) * (s_0x + s_mx)
        y_next = y + (h / 2) * (s_0y + s_my)
        z_next = z + (h / 2) * (s_0z + s_mz)

        x, y, z = x_next, y_next, z_next

    return trajectory


def rk4_method(initial_state, num_time_pts, h):
    """Solves first order non-linear differential equation systems with Runge-Kutta(4th) method.
    Args:
        initial_state: initial x,y,z values
        num_time_pts: number of time points to plot
        h: next_value - current_value
    Returns:
        numpy array: trajectory of lorenz attractor
    """
    x, y, z = initial_state
    trajectory = np.zeros((3, num_time_pts))
    for i in range(num_time_pts):
        # fill trajectory matrix
        trajectory[0, i] = x
        trajectory[1, i] = y
        trajectory[2, i] = z
        # define ODEs
        k_1x = h * lorenz(x, y, z)[0]
        k_1y = h * lorenz(x, y, z)[1]
        k_1z = h * lorenz(x, y, z)[2]
        # compute first set of intermediate values
        k_2x = h * lorenz(x + k_1x / 2, y + k_1y / 2, z + k_1z / 2)[0]
        k_2y = h * lorenz(x + k_1x / 2, y + k_1y / 2, z + k_1z / 2)[1]
        k_2z = h * lorenz(x + k_1x / 2, y + k_1y / 2, z + k_1z / 2)[2]
        # compute second set of intermediate values
        k_3x = h * lorenz(x + k_2x / 2, y + k_2y / 2, z + k_2z / 2)[0]
        k_3y = h * lorenz(x + k_2x / 2, y + k_2y / 2, z + k_2z / 2)[1]
        k_3z = h * lorenz(x + k_2x / 2, y + k_2y / 2, z + k_2z / 2)[2]
        # compute third set of intermediate values
        k_4x = h * lorenz(x + k_3x, y + k_3y, z + k_3z)[0]
        k_4y = h * lorenz(x + k_3x, y + k_3y, z + k_3z)[1]
        k_4z = h * lorenz(x + k_3x, y + k_3y, z + k_3z)[2]
        # update variables
        x_next = x + (1 / 6) * (k_1x + (2 * k_2x) + (2 * k_3x) + k_4x)
        y_next = y + (1 / 6) * (k_1y + (2 * k_2y) + (2 * k_3y) + k_4y)
        z_next = z + (1 / 6) * (k_1z + (2 * k_2z) + (2 * k_3z) + k_4z)

        x, y, z = x_next, y_next, z_next

    return trajectory


def put_markers(time_pt, trajectory):
    markers = np.zeros((3, 1))
    x = trajectory[0, time_pt]
    y = trajectory[1, time_pt]
    z = trajectory[2, time_pt]
    markers[0] = x
    markers[1] = y
    markers[2] = z
    return markers


def vis_diff_states(trj, method_name, time_pt, initial_state):
    name = method_name.capitalize()
    trj1, trj2 = trj
    st1, st2 = initial_state
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Initial state={st1}",
            f"Initial state={st2}",
        ],
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter3d"}],
        ],
        row_heights=[2],
        column_widths=[2, 2],
    )
    # add traces for 1. trajectory to the first subplot
    fig.add_trace(
        go.Scatter3d(
            x=trj1[0, :],
            y=trj1[1, :],
            z=trj1[2, :],
            mode="lines",
            line=dict(color="salmon", width=2),
            marker=dict(size=2),
            name=f"{name}",
        ),
        row=1,
        col=1,
    )

    # add markers for 1. trajectory to the first subplot
    fig.add_trace(
        go.Scatter3d(
            x=put_markers(time_pt, trj1)[0, :],
            y=put_markers(time_pt, trj1)[1, :],
            z=put_markers(time_pt, trj1)[2, :],
            mode="markers",
            marker_color="purple",
            marker_size=5,
            name=f"{name} marker",
        ),
        row=1,
        col=1,
    )

    # add traces for 2. trajectory to the first subplot
    fig.add_trace(
        go.Scatter3d(
            x=trj2[0, :],
            y=trj2[1, :],
            z=trj2[2, :],
            mode="lines",
            line=dict(color="salmon", width=2),
            marker=dict(size=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # add markers for 1. trajectory to the first subplot
    fig.add_trace(
        go.Scatter3d(
            x=put_markers(time_pt, trj2)[0, :],
            y=put_markers(time_pt, trj2)[1, :],
            z=put_markers(time_pt, trj2)[2, :],
            mode="markers",
            marker_color="purple",
            marker_size=5,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


def vis_diff_methods(trj_euler, trj_heuns, trj_rk4, time_pt):
    trace_eulers = go.Scatter3d(
        x=trj_euler[0, :],
        y=trj_euler[1, :],
        z=trj_euler[2, :],
        mode="lines",
        line=dict(color="orange", width=2),
        marker=dict(size=2),
        name="Euler",
    )
    trace_heuns = go.Scatter3d(
        x=trj_heuns[0, :],
        y=trj_heuns[1, :],
        z=trj_heuns[2, :],
        mode="lines",
        line=dict(color="green", width=2),
        marker=dict(size=2),
        name="Heun",
    )
    trace_rk4 = go.Scatter3d(
        x=trj_rk4[0, :],
        y=trj_rk4[1, :],
        z=trj_rk4[2, :],
        mode="lines",
        line=dict(color="blueviolet", width=2),
        marker=dict(size=2),
        name="Runge-Kutta(4. order)",
    )

    # add markers
    mark_euler = go.Scatter3d(
        x=put_markers(time_pt, trj_euler)[0, :],
        y=put_markers(time_pt, trj_euler)[1, :],
        z=put_markers(time_pt, trj_euler)[2, :],
        mode="markers",
        marker_color="darkorange",
        marker_size=5,
        name="Euler Marker",
    )
    mark_heuns = go.Scatter3d(
        x=put_markers(time_pt, trj_heuns)[0, :],
        y=put_markers(time_pt, trj_heuns)[1, :],
        z=put_markers(time_pt, trj_heuns)[2, :],
        mode="markers",
        marker_color="darkgreen",
        marker_size=5,
        name="Heun Marker",
    )
    mark_rk4 = go.Scatter3d(
        x=put_markers(time_pt, trj_rk4)[0, :],
        y=put_markers(time_pt, trj_rk4)[1, :],
        z=put_markers(time_pt, trj_rk4)[2, :],
        mode="markers",
        marker_color="purple",
        marker_size=5,
        name="RK4 Marker",
    )

    data = [trace_eulers, trace_heuns, trace_rk4, mark_euler, mark_heuns, mark_rk4]

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        title=f"Lorenz Attractor, Markers at t={time_pt}",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


def get_intersections(trj, axis):
    # return the points that crosses axis to get intersections
    return trj[axis]


def plot_poincare(
    ax1, ax2, intsec_ax, ax_index_1, ax_index_2, ax_index_intsec, trajectory_rk4
):
    a, b = trajectory_rk4[ax_index_1], trajectory_rk4[ax_index_2]
    fig, ax = plt.subplots(layout="constrained")
    sc = ax.scatter(
        a, b, c=get_intersections(trajectory_rk4, ax_index_intsec), cmap="viridis", s=1
    )
    ax.set_xlabel(f"{ax1}")
    ax.set_ylabel(f"{ax2}")
    fig.colorbar(mappable=sc, ax=ax, label=f"intersection at {intsec_ax} axis")
    ax.set_title(f"Poincaré map for Lorenz Attractor at {intsec_ax}=0")
    plt.show()


if __name__ == "__main__":
    main()
