# -*- coding: utf-8 -*-
import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Equations
def equations(y0, time) :
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
y=theta1[:,0]
y2=theta2

fig, (ax1, ax2) = plt.subplots(2,1,sharey=True)
s = ('Initial Angle = ' + str(initial_angle) + ' degrees')
plt.suptitle('Pendulum Motion: ' + s)
line, = ax1.plot(time, y)
line2, = ax2.plot(time, y2)
plt.xlabel('Time (s)')
fig.text(0.04, 0.5, 'Angle (rad)', va='center', rotation='vertical')
ax1.set_ylim(-1.2,1)
ax1.legend(['nonlinear, damped'], loc='lower right')
ax2.legend(['linear, undamped'], loc='lower right')
plt.subplots_adjust(top=0.94)

def update(num, time, y, y2, line,line2):
    line.set_data(time[:num], y[:num])
    line2.set_data(time[:num], y2[:num])
    return line, line2

ani = animation.FuncAnimation(fig, update, len(time), fargs=[time, y,y2, line,line2],
                              interval=10, blit=True)
plt.show()