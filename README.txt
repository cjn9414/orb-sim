Orbital Body Simulation
Date Modified: 3 December 2021
Authors:
	Carter Nesbitt <cjn9414>
	Paul Kelly <pjk2563>


Compile CPU version with the following command:
make cpu_barnes

Compile GPU version with the following command:
make gpu_barnes

Flags may be specified in the makefile within either the FLAGS makefile variable \
	(for CPU compilation) or the GPU_FLAGS makefile variable \
	(for GPU compilation). The available flags are specified below:

DIRECT: Performs direct orbital simulation as opposed to employing
	the Barnes-Hut clustering algorithm

LOG: Allows the automatic updating of a static logfile at each
	simulation iteration (out.txt)

ONLY_FINAL: If logging is enabled, only the final iteration data
	is logged when this flag is defined

Help message:
Options:
        -n <DIM>   Provide the number of dimensions for simulation
        -f <FNAME> Provide the name of file containing body data
        -d <DELTA> Provide the simulation timestep (seconds)
        -t <END>   Provide the simulation end time (seconds)
        -m <UNIT>  Provide the unit for mass: kg (k), sun mass (s), lbs (p)
        -l <UNIT>  Provide the unit for length: meters (m), parsecs (p), ft (f)
        -v <UNIT>  Provide the unit for velocity: m/s (m), km/s (k)

