# -*- coding: utf-8 -*-
import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Equations
def equations(y0, t) :
    theta, x = y0
    f = [x,-c*x -(g/l) * sin(theta)]
    return f
def plot_results(time, theta1, theta2) :
    plt.plot(time, theta1[:,0])
    plt.plot(time,theta2)
    s = ('Initial Angle = ' + str(initial_angle) + ' degrees')
    plt.title('Pendulum Motion: ' + s)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend(['nonlinear, damped', 'linear'], loc='lower right')

# Parameters
g = 9.81 # accel due to gravity (m/s^2)
l = 1.0 # length of pendulum (m)
delta_t = 0.025 # time step (s)
time = np.arange(0.0,10.0,delta_t) # time (s)

# Variables
c = 0.5 # damping constant

# Initial Conditions
initial_angle = 45.0
theta0 = np.radians(initial_angle)
x0 = np.radians(0.0) # Inital Velocity (rad/s)

# Solution to the nonlinear problem
theta1 = odeint(equations,[theta0, x0], time)

# Solution to linear problem
w = np.sqrt(g/l)
theta2 = [theta0 * cos(w*t) for t in time]

# plot results
plot_results(time,theta1,theta2)