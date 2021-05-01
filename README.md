# NBody

NBody simulator using the Barnes-Hut algorithm to achieve `NlogN` speedup
over traditional `N^2` algorithm.

## Run Code

```
python script.py --run-name [NAME] --num-part [NUM]
```

### Run MPI

Use this command to run with MPI to parallelize the computation
```
mpirun -n [num_cores] python script.py
```

## Results

One galaxy (1000 particles for 1000 timestep):

* [one_3d](analysis/one_3d.mp4) for 3D video.
* [one_2d](analysis/one_2d.mp4) for 2D video.

Two galaxies (500 particles each for 1000 timestep):

* [two_3d](analysis/two_3d.mp4) for 3D video.
* [two_2d](analysis/two_2d.mp4) for 2D video.


### Further
For a complete discussion of this NBody simulator please go to [gh-page](https://zxqlxy.github.io/NBody).

### TODO
- [x] Combine both 2D and 3D
- [x] Update the docstring
- [x] Added MPI to parallelize the computation
- [ ] Use cython and GPU for further speedup
- [ ] Explore more initial conditions
- [ ] Explore more algorithms in integrator
