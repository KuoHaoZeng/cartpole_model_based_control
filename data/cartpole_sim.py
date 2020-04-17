import sys

from numpy import clip
import numpy as np
from numpy import sin, cos

from scipy.integrate import solve_ivp

import cv2

class CartpoleSim:

    """
    Simulator for cartpole

    Cartpole state:
          dtheta:   [rad/s] angular velocity of pendulum
          dp:       [m/s]   velocity of cart
          theta:    [rad]   angle of pendulum (counterclockwise from negative y-axis)
          p:        [m]     position of cart

    Control:
          u:        [N]     force on cart (Newtons)
    """

    def __init__(self, 
                 g=9.81, l=0.5, m=0.5, 
                 M=0.5, b=0.1, dt=0.1,
                 max_u=10,
                 state_std=0.01,
                 seed=12345
                ):
        self.g = g                      # gravity (m/s^2)
        self.l = l                      # pendulum length (m)
        self.m = m                      # mass of pendulum (kg)
        self.M = M                      # mass of cart (kg)
        self.b = b                      # Friction between cart and ground (N/m/s)

        self.max_u = max_u              # Max amplitude of action
        self.state_std = state_std      # Standard deviation (sigma) of state noise

        self.dt = dt                    # Sampling period (s)
        self.rng = np.random.RandomState(seed)

    def dynamics(self, t, x, u):
        """ Dynamics function for cartpole: dx(t) = f(t, x(t))

            For a detailed derivation, see:
            Efficient Reinforcement Learning Using Gaussian Processes, Appendix C,
            KIT Scientific Publishing, 2010.

            @param t: not used
            @param state: a [4] numpy array
            @param u: a [1] numpy array

            @return: a [4] numpy array
        """
        u = u[0]
        u = self.max_u * clip(u, -1, 1) # Squash control

        sin_th = sin(x[2])
        cos_th = cos(x[2])

        ddtheta = ( -3*self.m*self.l*np.power(x[0],2)*sin_th*cos_th \
                    -6*(self.M+self.m)*self.g*sin_th \
                    -6*(u-self.b*x[1])*cos_th ) \
                  / ( 4*self.l*(self.M+self.m) - 3*self.m*self.l*np.power(cos_th,2) )
        ddp     = ( 2*self.m*self.l*np.power(x[0],2)*sin_th \
                   +3*self.m*self.g*sin_th*cos_th \
                   +4*u - 4*self.b*x[1] ) \
                  / ( 4*(self.M+self.m) - 3*self.m*np.power(cos_th,2) )
        dtheta  = x[0]
        dp      = x[1]

        return np.array([ddtheta, ddp, dtheta, dp])

    def step(self, x, u, noisy=True):
        """ Simulate for dt seconds from x, applying action u
        """
        temp = solve_ivp(self.dynamics, (0,self.dt), x, t_eval=(0,self.dt), args=(u,)) 
        x = temp.y[:,-1]
        if noisy:
            x = x + self.rng.normal(0, self.state_std, size=x.shape)

        return x
