Reconstruct a piecewise homogeneous object with DAMMER.

If you want to use this code in your own research, please cite the following reference:

@article{Merckx2025,

  title={ DAMMER: Direct Adaptive Multi-resolution MEsh Reconstruction from X-ray Measurements},

  author={Merckx, Jannes and den Dekker, Arnold Jan and Sijbers, Jan and De Beenhouwer},
  
  journal={IEEE Transactions on Computational Imaging},
  
  volume={11},
  
  pages={926-941},
  
  year={2025},
  
  doi={10.1109/TCI.2025.3587408}
}


DAMMER runs based on a numpy representation of the phantom. However, it is straightforward to change the input to a numpy representation of the sinogram. 

Dependencies:

numpy

Numba

astra (for projecting a sinogram from phantom images)

scipy

triangle

pylops

shapely

torch
