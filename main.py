import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint
import matplotlib.pyplot as plt


rho = 28  # scaled Rayleigh number
sigma = 10  # Prandtl number
beta = 8 / 3  # geometry aspect ratio


# initial conditions
y0 = [-8, 8, 27]  # initial state
delta_t = 0.001  # change in time
T = 25  # endpoint in time
num_time_pts = int(T / delta_t)  # time points for trajectory
t = np.linspace(0, T, num_time_pts)  # time interval


# Lorenz System
def lorenz(t, y):
    dy = [sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]]
    return np.array(dy)


# 1. Runge-Kutta Method(4th Order)
def rk4_integrate(fun, dt, t0, y0):
    # Integration of single system
    f1 = fun(t0, y0)
    f2 = fun(t0 + dt / 2, y0 + (dt / 2) * f1)
    f3 = fun(t0 + dt / 2, y0 + (dt / 2) * f2)
    f4 = fun(t0 + dt, y0 + dt * f3)
    yout = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
    return yout


def rk4_solver(y0, dt):
    Y = np.zeros((3, num_time_pts))
    Y[:, 0] = y0
    yin = y0
    for i in range(1, num_time_pts):
        yout = rk4_integrate(lorenz, dt, t[i], yin)
        Y[:, i] = yout
        yin = yout
    return Y


# 3D Interactive Simulation using Plotly
def simulation(Y, method_name):
    x, y, z = Y
    fig = go.Figure(
        data=[go.Scatter3d(x=x, y=y, z=z, mode="lines", marker=dict(size=2))]
    )
    # Customize layout
    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        title=f"Lorenz Attractor ({method_name})",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


# Compute trajectroy and create 3D scatter plot
trajectory = rk4_solver(y0, delta_t)
sim = simulation(trajectory, "Runge-Kutta Method")


# 2. Euler's Method
# X_{n+1} = X_n + h * f(X_n, t_n)


def euler_solver(y0, delta_t):
    Y = np.zeros((3, num_time_pts))  # trajectory matrix
    Y[:, 0] = y0  # assign initial state to starting point
    y_in = y0  # initial state

    # Update trajectory as time goes on
    for i in range(1, num_time_pts):
        # X_{n+1} = X_n + h * f(X_n, t_n)
        y_out = y_in + delta_t * lorenz(t[i], y_in)
        Y[:, i] = y_out  # update trajectory matrix
        y_in = y_out  # update current state

    return Y


trajectory = euler_solver(y0, delta_t)
sim = simulation(trajectory, "Euler's Method")


# 3. Heun's Method (Improved Euler)
# X_{n+1} = X_n + (h/2) * (f(X_n, t_n) + f(X_n + h * f(X_n, t_n), t_n + h))


def heuns_solver(initial_state, delta_t):
    current_state = initial_state
    V = np.zeros((3, num_time_pts))
    V[:, 0] = initial_state
    for i in range(1, num_time_pts):
        # X_{n+1} = X_n + (h/2) * (f(X_n, t_n) + f(X_n + h * f(X_n, t_n), t_n + h))
        next_state = current_state + (delta_t / 2) * (
            lorenz(t[i], current_state)
            + lorenz(
                t[i] + delta_t, current_state + delta_t * lorenz(t[i], current_state)
            )
        )
        V[:, i] = next_state
        current_state = next_state
    return V


trajectory = heuns_solver(y0, delta_t)
sim = simulation(trajectory, "Heun's Method")


### RUN BEFORE EXECUTING ###
# %pip install numpy
# %pip install plotly
# %pip install scipy
# %pip install matplotlib
