# GPU RKF45 (Runge–Kutta–Fehlberg) ODE parallel solver written in pycuda

The Runge–Kutta–Fehlberg method (RKF45) is one of the widely used ODE solvers. GRKF45 is a parallel RKF45 solver with many different parameter sets.

The original RKF45 code was taken from [cpp code](http://people.sc.fsu.edu/~jburkardt/cpp_src/rkf45/rkf45.html) by John Burkardt (LGPL).

Currently, you need to define the ODEs in r4_f0, r4_f1, r4_f2, .... in GRKF45.


## codes

- grkf45_1d (1D ODE)
- grkf45_2d (2D ODE, the van der pol oscillator)

<img src="https://github.com/HajimeKawahara/grkf45/blob/master/documents/figs/vanderpol.png" Titie="explanation" Width=350px>

- grkf45_3d (3D ODE, the Lorenz attractor)

<img src="https://github.com/HajimeKawahara/grkf45/blob/master/documents/figs/Lorentz.png" Titie="explanation" Width=650px>




