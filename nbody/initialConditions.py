import numpy as np


def plummer(N, dim:int, a, m=1., G=4.483e-3, seed=None):
    """Compute the positions and velocities of particles in the Plummer sphere

    Args:
        N (int): number of particles
        dim (int): dimension
        a (int): Plummer radius
        m ([int], optional): mass of particles can be a single value or an array
                of values. Defaults to 1..
        G (int, optional): Gravitational constant. Defaults to 4.483e-3.
        seed (int, optional): random generator seed.

    Returns:
        [pos, vel]: [(N x dim), (N x dim)] positions and velocities of particles in the Plummer sphere
    """

    if seed is None:
        rand = np.random
    else:
        rand = np.random.RandomState(seed=seed)
    Npart = int(N)
    pos = PlummerDist(N, dim, a, rand=rand)
    if np.size(m) == 1:
        M = N * m  # if all particles have the same mass
    else:
        M = np.sum(m)
    vel = velDist_Plummer(N, dim, pos, M, a, G, rand=rand)
    return [pos, vel]


def rand_unit_vector(d, rand=np.random):
    """ Generate random unit vector.

    Args:
        d (int): dimension of the vector
        rand (func, optional): random generate function. 
            Defaults to np.random.

    Returns:
        [np.array]: d-dimensional random unit vector (norm = 1)
    """

    phi = rand.uniform(0, np.pi*2)
    costheta = rand.uniform(-1, 1)
    theta = np.arccos(costheta)
    if d == 2:
        x = np.cos(phi) 
        y = np.sin(phi) 
        vec = np.array([x, y])
    elif d == 3:
        x = np.cos(phi)*np.sin(theta) 
        y = np.sin(phi)*np.sin(theta) 
        z = np.cos(theta)
        vec = np.array([x, y, z])
    return vec
    

# Spatial Distribution for Plummer Model
def PlummerDist(N, dim, a, rand=np.random):
    """Initializes particles with Plummer density profile.

    Args:
        N (int): number of particles
        dim (int): dimension
        a (int): Plummer radius
        rand (func, optional): random generate function. 
            Defaults to np.random.
    Returns:
        [np.array]:  (N x dim) array of positions
    """

    N = int(N)
    r = np.zeros((N))
    pos = np.zeros((N, dim))
    if dim == 3:
        for i in range(N):
            # Let enclosed mass fraction f_mi be random number between 0 and 1
            f_mi = rand.uniform(0., 1.)
            r[i] = a/np.sqrt(f_mi**(-2./3.)-1.)
            pos[i] = r[i]*rand_unit_vector(dim, rand=rand)
    elif dim == 2:
        for i in range(N):
            # Let enclosed mass fraction f_mi be random number between 0 and 1
            f_mi = rand.uniform(0., 1.)
            r[i] = a*np.sqrt(f_mi/(f_mi-1.))
            pos[i] = r[i]*rand_unit_vector(dim, rand=rand)
    return pos


# Initial velocities for particles in the Plummer model
def escapeV_Plummer(r, M, a, G):
    """Compute the escape velocity at a radius r from the center of the 
        Plummer sphere

    Args:
        r (float): radius away from the center of the Plummer sphere
        M (float): total mass of the Plummer sphere
        a (float): plummer radius
        G (float): gravitational constant

    Returns:
        [float]: escape velocity of a particle at radius r inside a Plummer sphere
    """

    pref = np.sqrt(2.*G*M/a)
    return pref*(1.+(r*r)/(a*a))**(-0.25)


def rejV_Plummer(r, dim, M, a, G, rand=np.random):
    """Uses the rejection technique to find the velocity drawn randomly
    from the velocity distribution

    Args:
        r (float): radius from the center of the Plummer sphere
        dim (int): dimension
        M (float): total mass of the Plummer sphere
        a (float): Plummer radius
        G (float): gravitational constant
        rand (func, optional): RandomState generator. Defaults to np.random.

    Returns:
        [float]: velocity of a particle
    """

    q = 0.
    gmax = 0.1  # slightly bigger than g_max = g(\sqrt(2/9)) = 0.092
    g0 = gmax
    while g0 > q*q * (1. - q*q)**(dim + .5):
        # 0 <= v <= v_esc or 0 <= q <= 1 where x = v/v_esc
        q = rand.uniform(0., 1.)
        # 0 <= g <= g_max
        g0 = rand.uniform(0., gmax)
    return q*escapeV_Plummer(r, M, a, G)


def velDist_Plummer(N, dim, r, M, a, G, rand=np.random):
    """Compute velocities from the velocity distribution of particles
    of the Plummer model.

    Args:
        N (int): total number of particles to be initialized
        dim (int): dimension
        r (float): radius from the center of the Plummer sphere
        M (float): total mass of the Plummer sphere
        a (float): Plummer radius
        G (float): gravitational constant
        rand (func, optional): RandomState generator. Defaults to np.random.

    Returns:
        [np.array]: (N, dim) array of velocities of particles
    """
    N = int(N)
    vel = np.zeros((N, dim))
    for i in range(N):
        r_abs = np.linalg.norm(r[i])
        vel_mod = rejV_Plummer(r_abs, dim, M, a, G, rand=rand)
        vel[i, :] = rand_unit_vector(dim, rand=rand)*vel_mod
    return vel


