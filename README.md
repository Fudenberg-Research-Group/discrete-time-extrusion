### Overview ###
This repository is associated with the manuscript "Interplay of diffusive and active motion governs cohesin extrusion and CTCF barrier function", and houses codes to simulate tension-dependent cohesin dynamics based on the interplay of diffusion, extrusion and regulator binding kinetics.

Cohesin kinetics are computed on a discrete 1D lattice, and can be run on either CPU or GPU via a native [CuPy](https://cupy.dev) implementation for large-scale simulations.

They can also be seamlessly integrated within 3D polymer simulations using [polychrom-hoomd](https://github.com/open2c/polychrom-hoomd), as described in [previous work](https://doi.org/10.1016/j.xgen.2025.101098).

<img width="6289" height="2937" alt="github-trim" src="https://github.com/user-attachments/assets/4b024e9a-31ab-42cf-a295-61cc6a47d087" />


An exemplary notebook illustrating code usage for tension-dependent simulations in double-tethered DNA assays is provided [here](https://github.com/Fudenberg-Research-Group/discrete-time-extrusion/blob/jump_diffusion/examples/jdm_simulations.ipynb). 

The movie below additionally displays predicted model kinetics at fixed, low (~0.1 pN) tensions below the cohesin stall force, showing how loop growth emerges from the competition of active, discrete DNA capture and passive, diffusion-driven slippage.

https://github.com/user-attachments/assets/37559df5-f343-4af5-b238-1f5a61155ffd
