# NBody

NBody simulator using Barnes-Hut algorithm to achieve $$NlogN$$ speedup
over traditional $N^2$ algorithm.

## Run Code

```
python script.py --run-name [NAME] --num-part [NUM]
```

### Run MPI

Use this command to run with MPI to parallelize the computation
```
mpirun -n [num_cores] python script.py
```

### Further
For a complete discussion of this NBody simulator please go to [gh-page](https://zxqlxy.github.io/NBody).

### TODO
- [x] Combine both 2D and 3D
- [x] Update the docstring
- [x] Added MPI to parallelize the computation
- [ ] Use cython and GPU for further speedup
- [ ] Explore more initial conditions
- [ ] Explore more algorithms in integrator
