#ifndef __BH_H__
#define __BH_H__

#include "common.h"

// Tree generation on GPU (incomplete)
__global__ void GenerateTreeKernel(Body *bodies, int body_count, Node *nodes, int ndim, double *thread_dim_lengths);

// Helper device function to count bodies in a specified domain
__device__ void CountBodiesInNodeBoundsGPU(Body *bodies, int body_count, Node *node, int *last_encountered, int ndim, int *bound_count);

// Helper devie function to determine if a body is in a specified domain
__device__ void BodyInRangeGPU(Body *body, double *min, double *max, int ndim, bool *in_range);

// Helper function for tree generation
__device__ void GenerateTreeHelper(Body *bodies, int body_count, int ndim, Node *parent, int sector);

// Kernel function to calculate forces and update body positions
__global__ void UpdateBodiesKernel(Body *bodies, int body_count, Tree *tree, int ndim, double dt);

// Helper function to compute forces
__device__ void ComputeForcesGPU(Body *body, Node *node, int ndim, double *force);

// Kernel function to perform a direct simulation step
__global__ void DirectSimulationStepKernel(Body *bodies, int body_count, int ndim, double dt);

// Main simulation function
bool RunSimulationGPU(Body *bodies, int body_count, int ndim, double *dim_ranges, FILE *fp, double tf, double dt, char mass_unit, char length_unit, char velocity_unit);

#endif
