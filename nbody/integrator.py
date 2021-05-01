
import numpy as np
import os
from mpi4py import MPI
from warnings import warn


def serial_timestep(x, v, a, dt):
    """ Returns the updated positions (p) and velocities (v) given accelerations
    (a) and a timestep dt, using the leap-frog algorithm.

    Args:
        x (np.array): positions (N x d)
        v (np.array)): velocities (N x d)
        a (np.array): accelerations (N x d)
        dt (float): time step

    Returns:
        [np.array, np.array]: new positions (N x d), new velocites (N x d)
    """
    
    # for the correct leapfrog condition, assume self-started
    # i.e. p = p(i)
    #      v = v(i - 1/2)
    #      a = F(p(i))
    # kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
    v1 = v + a * dt
    # drift step: x(i+1) = x(i) + v(i + 1/2) dt
    x1 = x + v1 * dt
    return x1, v1
