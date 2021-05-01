.. _initialcondition:

Initial Conditions
===============

In collisionless N-body simulations, when two particles approaches each other, 
the potential will diverge, so the *point mass potential*

.. math::
        \phi(r) = -\frac{GM}{r}

is unphysical. Thus, a fix to this problem is to *gravitational softening*. We introduce 
*a* in the potential, and this is called Plummer model, *a* is called Plummer scale length. :cite:`10.1093/mnras/71.5.460`

Plummer Model
-----------------

Plummer potential is 

.. math::
        \phi(r) = -\frac{GM}{\sqrt{r^2 + a^2}}
   
In order to generate the initial conditions, we need to compute *Spatial Distribution* and 
*Velocity Distribution* of the Plummer model.

First compute the *Spatial Distribution*, suppose *k* is the portion of mass, then we have

.. math::
        M_0\frac{r^3}{(r^2+a^2)^\frac{3}{2}} = kM_0

We can solve to get 

.. math::
        r = \left(\frac{1}{k^\frac{2}{3}} - 1\right)^{-\frac{1}{2}}a

In 2-D problem, we have similar calculation

.. math::
        M_0\frac{r^2}{r^2+a^2} = kM_0 \\
        r = \left(\frac{k}{1-k}\right)^\frac{1}{2}a

:meth:`nbody.initialConditions.PlummerDist`

Then, we need to compute *Velocity Distribution*. We first compute escape velocity by setting 
kinetic energy to potential energy

.. math::
        \frac{1}{2}mv^2 = \frac{GMm}{\sqrt{r^2 + a^2}} \\

and we can get escape velocity

.. math::
        v_e = \sqrt{\frac{2GM}{\sqrt{r^2 + a^2}}}

Then, by the energy distribution function, we can compute probablity function  :math:`g(v)`

.. math::
        g(v)dv \propto  (-E(r,v))^{\frac{7}{2}}v^2dv

Then, we will have a distribution fo velocity. After set :math:`q = \frac{v}{v_e}`, we get

.. math::
        E(q) \propto q^2 - 1    \\
        g(q) = (1 - q^2)^{\frac{7}{2}}q^2

and we know that :math:`0\leq q \leq 1`.

In order to sample correct values, we will use rejection techniqu first came up by John von Neumann. 
It basically will shoot to a value where the probablity is smaller than the problility corresponding 
to specific :math:`q`. 

We can compute the maximum value of :math:`g(q)` is 0.092. So it's safe to set the maximum probablity 
to 0.1 as in :meth:`nbody.initialConditions.rejV_Plummer`. If it's 2-D scenario, simply set :math:`\frac{7}{2}`
to :math:`\frac{5}{2}` and we can compute :math:`g(q) < 0.0353`.


Reference
------------------

* https://en.wikipedia.org/wiki/Plummer_model
* http://www.artcompsci.org/kali/vol/plummer/ch04.html


.. bibliography:: ref_condition.bib
