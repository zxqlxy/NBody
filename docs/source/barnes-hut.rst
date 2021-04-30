.. _barnes-hut:

The Barnes-Hut tree algorithm
=============================

As we know, a naive algorithm that calculate the forces in a N-body system will
take :math:`O(N^2)` and this can scale too fast when the number of objects :math:`N`
becomes big. Thus, a better solution is proposed :cite:`barnesHierarchicalLogForcecalculation1986` 
where we only need to use :math:`O(Nlog(N))` to compute the force for the N-body system.

Overview
-----------------

For :math:`N` particles we have, we can use a tree structure to iteratively populate these :math:`N`
into the tree so that each inner nodes (which is not leaf) has no particle and each leaf no more than 
one particle. Of course, some of the leaves don't have particle.

In 2D scenario, once there are more than one particle in this "cell", we have to separte the 
this "cell" into 4 leaves and in 3D scenario, we have to separte the "cell" into 8 leaves.

In order to accelerate the computation, it is essential to know the center of mass of different "cell"
(can contain multiple particles) and each "cell" also has an area which is just a subdivision of
its parent "cell". 

This subdiving process won't go forever, it will stop until we the "cell"'s side length
is within our accepted error range. More specifically, if we let :math:`d` to be the distance between 
this particle and the center of mass of this treenode, :math:`l` to be the length of the side of the "cell",
we can define "opening angle" :math:`\theta = \frac{l}{d}`. If :math:`\theta` is smaller than some value, 
we can simply treat them as a single mass to compute the force.

Analysis
-----------------
The process of computing each leaf for each particle will generally take :math:`O(log(N))` because traverse 
down a tree is essentailly iterate over the tree's depth and it's in :math:`O(log(N))` if :math:`N` is the 
number of nodes in the tree. We have to do this for those :math:`N` particles and this will give :math:`O(Nlog(N))`.
The actual force evaluation will go for the same step as further chunks will be evaluated together and nearer 
chunks will be smaller (as it is in scale with the "cell" side length). So each force evaluation for particle
will also be :math:`O(log(N))`.

Unfortuately, this tree construction process need to be done for every iteration so it is the one thing we 
cannot get away with.


