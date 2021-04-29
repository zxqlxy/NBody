import numpy as np

class cell(object):
    def __init__(self, box, dim=2):
        """Initialize cell object

        Args:
            box (np.array): (dim,2) array
            dim (int, optional): dimension. Defaults to 2.
        """
        self.loc = np.array(box)
        self.dim = dim
        
        self.xlow = self.loc[0,0]
        self.xhigh = self.loc[0,1]
        
        self.ylow = self.loc[1,0]
        self.yhigh = self.loc[1,1]

        if dim == 3:
            self.zlow = self.loc[2,0]
            self.zhigh = self.loc[2,1]
            self.center = np.array([(self.loc[0,0]+self.loc[0,1])/2, (self.loc[1,0]+self.loc[1,1])/2, (self.loc[2,0]+self.loc[2,1])/2])
        else:
            self.center = np.array([(self.loc[0,0]+self.loc[0,1])/2, (self.loc[1,0]+self.loc[1,1])/2])

    def __call__(self):
        """return the location of the cell

        Returns:
            [np.array]: the location of the cell
        """
        return self.loc
    
    def inside(self, p):
        """find whether particle is inside the cell

        Args:
            p (np.array): location of a particle

        Returns:
            [bool]: whether particle is inside the cell
        """
     
        if self.dim == 2:
            if (p[0] < self.xlow or p[0] > self.xhigh or p[1] < self.ylow or p[1] > self.yhigh):
                return False
            else:
                return True
        else:
            if (p[0] < self.xlow or p[0] > self.xhigh or p[1] < self.ylow or p[1] > self.yhigh or p[2] < self.zlow or p[2] > self.zhigh):
                return False
            else:
                return True
        
    def middle(self):
        """
        find the center of the cell

        Returns:
            [np.array]: the center of the cell
        """
        if self.dim == 3:
            return np.array([((self.xlow + self.xhigh))/2., (self.ylow + self.yhigh)/2., (self.zlow + self.zhigh)/2.])
        else:
            return np.array([((self.xlow + self.xhigh))/2., (self.ylow + self.yhigh)/2.])
    
    def bounds(self):
        """returns min/max values of the cell

        Returns:
            [np.array]: the min/max values of the cell
        """

        return np.array([self.loc.min(), self.loc.max()])


class node(object):
    def __init__(self, cell, particles, masses):
        """Initialize node object

        Args:
            cell (cell): the object with useful information of grid
            particles (np.array): positions of the particles
            masses (np.array): masses of the particles
        """
        super().__init__()
        self.dim = cell.dim
        self.cell = cell
        self.center = self.cell.middle() #center of node/cell
        self.com = self.center
        self.children = [] #list of children of this node
        
        self.n = 0 #number of particles 
        self.p = None #the *single* particle bound to this node. only set when we hit a leaf node
        self.pid = None
        
        self.particles = particles #positions of particles belonging 
        self.masses = masses #masses of particles belonging
        
        self.M = np.sum(self.masses) #total mass of all particles
        self.leaf = False 

    def insert(self, particle, idx):
        """ Recursivly find the leaf with 0 particles and insert the particle 
        to that node

        Args:
            particle (np.array): (dim, ) position of a particle to be added
            idx (int): idx to keep track of different leaves
        """
        if(not self.cell.inside(particle)): #check if particle is in the node/cell
            return 
        
        if(self.n == 0): #no particles in this node/cell
            self.p = particle #assign particle to this node/cell
            self.com = particle #com is just the particle position 
            self.leaf = True
        else:
            if(self.n == 1): #leaf node
                self.create_children(self.cell) #create children (aka subtree)
                for child in self.children:
                    child.insert(self.p, self.pid) #need to reassign the current particle bc the new particle also wants to be in this node/cell
                self.p = None #once it's reassigned, the particle belonging to this node/cell becomes none
            for child in self.children:
                child.insert(particle, idx) #iterate over the children and assign the input particle
            self.leaf = False
                
        self.pid = idx #pid of the particle assigned to the leaf
        self.update_com() 
        self.n += 1 
    
        return 

    def update_com(self):
        """
        updates center of mass. 
        """
        self.com = np.zeros(self.dim)
        self.com[0] = np.dot(self.particles[:,0], self.masses[:])/self.M
        self.com[1] = np.dot(self.particles[:,1], self.masses[:])/self.M
        if (self.dim == 3):
            self.com[2] = np.dot(self.particles[:,2], self.masses[:])/self.M
        
        return 
    
    def create_children(self, cell):
        """subdivides the current cell into subdivisions and creates those children nodes. 

        Args:
            cell (cell): the cell object for current node
        """

        xhalf = self.center[0]
        yhalf = self.center[1]
        if self.dim == 3:
            zhalf = self.center[2]
        
        #This will return a boolean array of shape (len(particles),dim) indicating which octant the particles belong in
        index = self.particles > self.cell.middle() 

        if self.dim == 2:
            c1_cell = cell(np.array([[cell.xlow, xhalf], [cell.ylow, yhalf]]))
            mask = np.all(index == np.bool_([0,0]), axis=1)
            c1 = node(c1_cell, self.particles[mask], self.masses[mask])
            
            c2_cell = cell(np.array([[xhalf, cell.xhigh], [cell.ylow, yhalf]]))
            mask = np.all(index == np.bool_([1,0]), axis=1)
            c2 = node(c2_cell, self.particles[mask], self.masses[mask])
            
            c3_cell = cell(np.array([[cell.xlow, xhalf], [yhalf, cell.yhigh]]))
            mask = np.all(index == np.bool_([0,1]), axis=1)
            c3 = node(c3_cell, self.particles[mask], self.masses[mask])
            
            c4_cell = cell(np.array([[xhalf, cell.xhigh], [yhalf, cell.yhigh]]))
            mask = np.all(index == np.bool_([1,1]), axis=1)
            c4 = node(c4_cell, self.particles[mask], self.masses[mask])
                
            self.children = [c1, c2, c3, c4] 

        if self.dim == 3:
            c1_cell = cell(np.array([[cell.xlow, xhalf], [cell.ylow, yhalf], [cell.zlow, zhalf]]))
            mask = np.all(index == np.bool_([0,0,0]), axis=1)
            c1 = node(c1_cell, self.particles[mask], self.masses[mask])
            
            c2_cell = cell(np.array([[xhalf, cell.xhigh], [cell.ylow, yhalf], [cell.zlow, zhalf]]))
            mask = np.all(index == np.bool_([1,0,0]), axis=1)
            c2 = node(c2_cell, self.particles[mask], self.masses[mask])
            
            c3_cell = cell(np.array([[cell.xlow, xhalf], [yhalf, cell.yhigh], [cell.zlow, zhalf]]))
            mask = np.all(index == np.bool_([0,1,0]), axis=1)
            c3 = node(c3_cell, self.particles[mask], self.masses[mask])
            
            c4_cell = cell(np.array([[xhalf, cell.xhigh], [yhalf, cell.yhigh], [cell.zlow, zhalf]]))
            mask = np.all(index == np.bool_([1,1,0]), axis=1)
            c4 = node(c4_cell, self.particles[mask], self.masses[mask])
            
            c5_cell = cell(np.array([[cell.xlow, xhalf], [cell.ylow, yhalf], [zhalf, cell.zhigh]]))
            mask = np.all(index == np.bool_([0,0,1]), axis=1)
            c5 = node(c5_cell, self.particles[mask], self.masses[mask])
            
            c6_cell = cell(np.array([[xhalf, cell.xhigh], [cell.ylow, yhalf], [zhalf, cell.zhigh]]))
            mask = np.all(index == np.bool_([1,0,1]), axis=1)
            c6 = node(c6_cell, self.particles[mask], self.masses[mask])
            
            c7_cell = cell(np.array([[cell.xlow, xhalf], [yhalf, cell.yhigh], [zhalf, cell.zhigh]]))
            mask = np.all(index == np.bool_([0,1,1]), axis=1)
            c7 = node(c7_cell, self.particles[mask], self.masses[mask])
            
            c8_cell = cell(np.array([[xhalf, cell.xhigh], [yhalf, cell.yhigh], [zhalf, cell.zhigh]]))
            mask = np.all(index == np.bool_([1,1,1]), axis=1)
            c8 = node(c8_cell, self.particles[mask], self.masses[mask])
                
            self.children = [c1, c2, c3, c4, c5, c6, c7, c8]



class tree(object):
    def __init__(self, particles, masses, cell):
        """initialize the tree structure

        Args:
            particles (np.array): positions of all particles (N, dim)
            masses (np.array): masses of all particles in simulation (N, 1)
            cell (cell): the cell object
        """
        self.particles = particles
        self.masses = masses
        self.cell = cell
        self.dim = cell.dim
        self.root = self.create_tree()
        self.leaves = []                        #  list of leaves 
        self.particle_dict = {}                 #  mapping particle index to the leaf node
        self.get_all_leaves(self.root)
        
        
    def create_tree(self):
        bl, bh = self.cell.bounds()
        if self.dim == 2:
            bb = cell([[bl, bh], [bl, bh]]) #does this make sense to do? or should i use the particle min/maxes
        else: 
            bb = cell([[bl, bh], [bl, bh], [bl, bh]])
        root = node(bb, self.particles, self.masses)
        
        for i in range(len(self.particles)): #parallelize tree construction!!!!
            root.insert(self.particles[i], i) 
        
        return root
    
    def get_all_leaves(self, n):
        """Get all the leaves of this node

        Args:
            n (node): a node object whoes leaves are to be added
        """
        if(n.leaf): #if n is a leaf
            self.leaves.append(n) #append to leaves list
            self.particle_dict[n.pid] = n #update partcile_dict
        else:
            for c in n.children: #otherwise loop over all its children
                self.get_all_leaves(c)
            
    def accel(self, theta, particle_id, G, eps=0.1):
        """Calculate acceleration for a given particle with some tolerance theta

        Args:
            theta (float): [description]
            particle_id ([type]): [description]
            G ([type]): [description]
            eps (float, optional): [description]. Defaults to 0.1.

        Returns:
            [type]: [description]
        """
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