Reconstruct a piecewise homogeneous object with DAMMER.

DAMMER runs based on a numpy representation of the phantom. However, it is straightforward to change the input to a numpy representation of the sinogram. 
The newer python file is an improved version of the code in the paper in terms of running time. However, outputs are the same. 

Dependencies:

numpy

Numba

astra (for projecting a sinogram from phantom images)

scipy

triangle

pylops

shapely

torch
