"""
.. module:: nbody

"""

import numpy as np

class node(object):
    def __init__(self, dim, cell, particles, masses):
        """Initialize node object

        Args:
            dim (int): the dimension of the object
            cell (cell): the object with useful information of grid
            particles ([type]): [description]
            masses ([type]): [description]
        """
        super().__init__()
        self.dim = dim
        self.cell = cell
        self.center = self.box.middle() #center of node/cell
        self.com = self.center
        self.children = [] #list of children of this node
        
        self.n = 0 #number of particles 
        self.p = None #the *single* particle bound to this node. only set when we hit a leaf node
        self.pid = None
        
        self.particles = particles #positions of particles belonging to this node
        self.masses = masses #masses of particles belonging to this node
        
        self.M = np.sum(self.masses) #total mass of all particles in this node
        self.leaf = False


    def insert(self, particle, idx):
        """
        particle: (2,) numpy array of the position of a particle to be added to the current node

        This is a recursive function that will keep creating new nodes/cells/octants till it finds a node
        with 0 particles in it and then populate that node with the input particle.
        """
        if(not self.box.inside(particle)): #check if particle is in the node/cell
            return 
        
        if(self.n == 0): #no particles in this node/cell
            self.p = particle #assign particle to this node/cell
            self.com = particle #com is just the particle position 
            self.leaf = True
        else:
            if(self.n == 1): #leaf node
                self.create_children(self.box) #create children (aka subtree)
                for child in self.children:
                    child.insert(self.p, self.pid) #need to reassign the current particle bc the new particle also wants to be in this node/cell
                self.p = None #once it's reassigned, the particle belonging to this node/cell becomes none
            for child in self.children:
                child.insert(particle, idx) #iterate over the children and assign the input particle
            self.leaf = False
                
        self.pid = idx #pid of the particle assigned to the leaf
        self.update_com() #update center of mass of this node/cell once the particle is assigned
        self.n += 1 #update number of particles belonging to this node/cell.
    
        return 

    def update_com(self):
        """
        updates center of mass. p self-explanatory
        """
        self.com = np.zeros(2)
        self.com[0] = np.dot(self.particles[:,0], self.masses[:])/self.M
        self.com[1] = np.dot(self.particles[:,1], self.masses[:])/self.M
        # self.com[2] = np.dot(self.particles[:,2], self.masses[:])/self.M
        
        return 
    
    def create_children(self, box):
        """
        subdivides the current box into 8 octants and creates those children nodes. 
        """
        xhalf = self.center[0]
        yhalf = self.center[1]
        # zhalf = self.center[2]
        
        index = self.particles > self.box.middle() #will return a boolean array of shape (len(particles),3) indicating which octant the particles belong in
        
        c1_box = cell(np.array([[box.xlow, xhalf], [box.ylow, yhalf]]))
        mask = np.all(index == np.bool_([0,0]), axis=1)
        c1 = node(c1_box, self.particles[mask], self.masses[mask])
        
        c2_box = cell(np.array([[xhalf, box.xhigh], [box.ylow, yhalf]]))
        mask = np.all(index == np.bool_([1,0]), axis=1)
        c2 = node(c2_box, self.particles[mask], self.masses[mask])
        
        c3_box = cell(np.array([[box.xlow, xhalf], [yhalf, box.yhigh]]))
        mask = np.all(index == np.bool_([0,1]), axis=1)
        c3 = node(c3_box, self.particles[mask], self.masses[mask])
        
        c4_box = cell(np.array([[xhalf, box.xhigh], [yhalf, box.yhigh]]))
        mask = np.all(index == np.bool_([1,1]), axis=1)
        c4 = node(c4_box, self.particles[mask], self.masses[mask])
            
        self.children = [c1, c2, c3, c4] #assign children


class cell(object):
    def __init__(self, box, dim=2):
        """
        box: numpy array (2,2)
        """
        self.loc = np.array(box)
        
        self.xlow = self.loc[0,0]
        self.xhigh = self.loc[0,1]
        
        self.ylow = self.loc[1,0]
        self.yhigh = self.loc[1,1]
        
        self.center = np.array([(self.loc[0,0]+self.loc[0,1])/2, (self.loc[1,0]+self.loc[1,1])/2])
        self.dim = dim
     
    def __call__(self):
        return self.loc
    
    def inside(self, p):
        """
        given coordinate is inside the bounding box
        input: p is an array of x, y
        output: True or False
        """
        if (p[0] < self.xlow or p[0] > self.xhigh or p[1] < self.ylow or p[1] > self.yhigh):
            return False
        else:
            return True
        
    def middle(self):
        """
        returns center of bounding box
        """
        return np.array([((self.xlow + self.xhigh))/2., (self.ylow + self.yhigh)/2.])
    
    def bounds(self):
        """
        returns min/max values of the bounding box
        """
        return np.array([self.loc.min(), self.loc.max()])


class quadtree(object):
    """
    Inputs: 
        particles: positions of all particles in simulation (Nx3 numpy array)
        masses: masses of all particles in simulation (Nx1 numpy array)
        box: bounding box (class bbox; see above.)
    Attributes:
        particles: positions of all particles in simulation (Nx3 numpy array)
        masses: masses of all particles in simulation (Nx1 numpy array)
        box: bounding box (class bbox; see above.)
        root: octnode object which holds the root of the created tree (octnode)
        leaves: list of leaves (octnode)
        particles_dict: dictionary mapping particle index to the leaf node the particle belongs to (dictionary)
    """
    def __init__(self, particles, masses, box):
        self.particles = particles
        self.masses = masses
        self.box = box
        self.root = self.create_tree()
        self.leaves = []
        self.particle_dict = {}
        self.get_all_leaves(self.root)
        
        
    def create_tree(self):
        bl, bh = self.box.bounds()
        bb = cell([[bl, bh], [bl, bh]]) #does this make sense to do? or should i use the particle min/maxes
        root = node(bb, self.particles, self.masses)
        
        for i in range(len(self.particles)): #parallelize tree construction!!!!
            root.insert(self.particles[i], i) 
        
        return root
    
    def get_all_leaves(self, n):
        if(n.leaf): #if n is a leaf
            self.leaves.append(n) #append to leaves list
            self.particle_dict[n.pid] = n #update partcile_dict
        else:
            for c in n.children: #otherwise loop over all its children
                self.get_all_leaves(c)
            
    def accel(self, theta, particle_id, G, eps=0.1):
        """
        Description: 
            Calculate acceleration for a given particle_id in the simulation with some tolerance theta
        Inputs:
            theta: opening angle (float)
            particle_id: index of particle in sim to calculate force for (int)
            G: gravitational constant (float)
        Output:
            grad: force array (1x3)
        """
        grad = self.traverse(self.root, self.particle_dict[particle_id], theta,
                             particle_id, np.zeros(2), G, eps=eps)
        return grad
    
    def traverse(self, n0, n1, theta, idx, ret, G, eps=0.01):
        """
        given two nodes n0 and n1, and some tol theta, traverse the tree till it's far enough that you can approximate the
        node as a "particle" and add the gravitational acceleration of that particle to the ret array. n1 is the leaf node that 
        holds the particle we are calculating the accel for.
        """
        if(n0 == n1):
            return
        dr = n0.com - n1.com
        r = np.sqrt(np.sum(dr**2))
        size_of_node = n0.box.xhigh - n0.box.xlow
        if(size_of_node/r < theta or n0.leaf):
            ret += G*n0.M*dr/(r**2 + eps**2)**1.5
        else:
            for c in n0.children:
                self.traverse(c, n1, theta, idx, ret, G)
        return ret