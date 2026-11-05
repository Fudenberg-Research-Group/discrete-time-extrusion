### Implementation notes ###
The two design choices below have been made in the current code implementation.

#### 1. Tension is included at the `BaseExtruder` level ####
##### Pros #####
* Can be easily turned off by providing `tension=0` (or omitting tension) when invoking stepping functions
* Avoids proliferation of trivial derived classes `TensionDependentExtruder`, `TensionDependentMultistateExtruder` etc.

##### Cons #####
* Difficult to locate implementation (currently deferred to [`EngineFactory`](https://github.com/Fudenberg-Research-Group/discrete-time-extrusion/blob/jump_diffusion/discrete_time_extrusion/extruders/EngineFactory.py) and kernels therein)


#### 2. Keyword forwarding is extensively used throughout ####
##### Pros #####
* Abbreviates nested calls for functions with longer argument lists
* Avoids explicit mentions of arguments required only by derived methods in the base class

##### Cons #####
* Complicates tracking down definitions of default parameter values
* Occasionally results in the silent parsing of extraneous arguments
