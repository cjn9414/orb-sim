#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <ctime>

#include "bh.h"
#include "common.h"

// Preprocess block length
#ifndef BLOCK_LEN
    #define BLOCK_LEN 512
#endif

// Preprocess grid length
#ifndef GRID_LEN
    #define GRID_LEN 10
#endif

// Performing GPU tree generation (incomplete)
#ifdef GEN_TREE_GPU
    #define GPU_TREE_GPU_ENABLED 1
#else
    #define GPU_TREE_GPU_ENABLED 0
#endif

#define BODIES_PER_BLOCK(total, dim) (total / (dim.x * dim.y * dim.z))
#define NODES_PER_BLOCK(total, dim) (total / (dim.x * dim.y * dim.z))
#define MAX_FREE_BUFFER 10000
#define SECTION_LOCK 1
#define SECTION_UNLOCK 0
#define CUDA_ERROR_MEMORY_ALLOCATION 2

/*
 * Verifies that the provided cuda status is labelled successful.
 * Input status: Cuda status to evaluate
 * Output: True is successful, false otherwise
 */
bool verifyCuda(cudaError_t status) {
    // Perform closing operations if kernel execution failed
    if (status != cudaSuccess) {
        std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    return true;
}


// Helper function to generate tree on GPU
__device__ void GenerateTreeKernelHelper(Body *bodies, int body_count, int ndim, \
        Body *freeBodies, Node *freeNodes, int freeBodyCount, int freeNodeCount, \
        Node *parent, int *return_idx, int sector) {
  __shared__ int body_idx, node_idx;
  __shared__ int body_mutex, node_mutex;
  int i, bound_count;
  int last_encountered;
  double pos[DIM];
  double mass = 0.0;
  Node *node, *child;
  int child_idx;

  // Critical section! Create a new node at shared index
  while (true) {
    if (atomicExch(&node_mutex, SECTION_LOCK) == SECTION_UNLOCK) {
      // Are we over budget for nodes?
      // If so, stop tree construction,
      // this is a big time loss
      if (node_idx >= freeNodeCount) {
        atomicExch(&node_mutex, SECTION_UNLOCK);
        *return_idx = CUDA_ERROR_MEMORY_ALLOCATION;
        return;
      } else {
        // Create node instance
        node = new(freeNodes + node_idx) Node();
        *return_idx = node_idx;
        node_idx += 1;
        atomicExch(&node_mutex, SECTION_UNLOCK);
      }
    } 
  }

  // Divide parent's spatial bounds
  node->SetParent(parent);
  double *par_min = node->parent->min_range;
  double *par_max = node->parent->max_range;
  for (int i = 0; i < ndim; i++) {
      // Each bit of sector represents the half of the parent's domain that
      // is collected in each dimension (first half or second half)
      bool isDimEnd = (1 << i) & sector;
      
      // Get the half-side length of parent's domain -- this is now our side length
      double half_len = (par_max[i] - par_min[i]) / 2;

      // Set our bounding box
      node->min_range[i] = par_min[i] + (isDimEnd ? 0 : half_len);
      node->max_range[i] = par_max[i] - (isDimEnd ? half_len : 0);
  }

  CountBodiesInNodeBoundsGPU(bodies, body_count, node, &last_encountered, ndim, &bound_count);

  if (bound_count > 1) {

      for (i = 0; i < (1 << ndim); i++) {            
          // Recurse on function to generate descending nodes
          // This function call will place the resulting child node pointer
          // into the respective variable
          GenerateTreeKernelHelper(bodies, body_count, ndim, \
            freeBodies, freeNodes, freeBodyCount, freeNodeCount, node, &child_idx, i);

          if (child_idx != -1) {

              child = freeNodes + child_idx;

              node->AddChild(child);

              mass += child->cluster->mass;
              
              for (int j = 0; j < ndim; j++) {
                  pos[j] += child->cluster->pos[j] * child->cluster->mass;
              }
          } else if (body_idx >= freeBodyCount || node_idx >= freeNodeCount) {
            // Big problem, terminate kernel
            *return_idx = CUDA_ERROR_MEMORY_ALLOCATION;
            return;
          }
      }

      // Normalize center of mass vector by total mass of cluster
      for (int j = 0; j < ndim; j++) {
          pos[j] /= mass;
      }

      // Critical section! Create a new body at shared index
      while (true) {
        if (atomicExch(&body_mutex, SECTION_LOCK) == SECTION_UNLOCK) {
          // Are we over budget for bodies?
          // If so, stop tree construction,
          // this is a big time loss
          if (body_idx >= freeBodyCount) {
            atomicExch(&body_mutex, SECTION_UNLOCK);
            *return_idx = CUDA_ERROR_MEMORY_ALLOCATION;
            return;
          } else {
            // Create cluster body instance
            node->cluster = new(freeBodies + body_idx) Body(&(pos[0]), mass, ndim);              
            body_idx += 1;
            atomicExch(&body_mutex, SECTION_UNLOCK);
          }
        }    

        // Are we over budget for bodies?
        // If so, stop tree construction,
        // this is a big time loss
        if (body_idx >= freeBodyCount) {
          return;
        }     
      }
  } else if (bound_count == 1) {
      // No need to create new body instance, just grab body residing in our domain
      node->cluster = &(bodies[last_encountered]);
  } else {
      // No bodies in our domain, don't do anything, return null
      *return_idx = CUDA_ERROR_MEMORY_ALLOCATION;
  }
}

// GPU Kernel to generate simulation tree
__global__ void GenerateTreeKernel(Body *bodies, int body_count, \
        int ndim, double *thread_dim_lengths, Body *freeBodies, Node *freeNodes,\
        int freeBodyCount, int freeNodeCount, int *return_idx) {
    
    // Shared mutex variables
    __shared__ int body_idx, node_idx;
    __shared__ int body_mutex, node_mutex;
    
    int dims[DIM];
    int i, bound_count;
    int last_encountered;
    double pos[DIM];
    double mass = 0.0;
    Node *node, *child;
    int child_idx;

    // Mutex booleans
    bool nodeBlocked = true;
    bool bodyBlocked = true;

    Body *freeBodiesForBlock = freeBodies + \
        BODIES_PER_BLOCK(freeBodyCount, gridDim) * (blockIdx.z * (gridDim.y * gridDim.x) + \
        blockIdx.y * gridDim.x + blockIdx.x);
    Node *freeNodesForBlock = freeNodes + NODES_PER_BLOCK(freeNodeCount, gridDim) * (blockIdx.z * (gridDim.y * gridDim.x) + \
        blockIdx.y * gridDim.x + blockIdx.x);

    dims[0] = threadIdx.x + (blockIdx.x * blockDim.x);
    dims[1] = threadIdx.y + (blockIdx.y * blockDim.y);
    dims[2] = threadIdx.z + (blockIdx.z * blockDim.z);

    int thread_idx = dims[2] * (blockDim.y * gridDim.y) * (blockDim.x * gridDim.x) + \
                                        dims[1] * (blockDim.x * gridDim.x) + \
                                        dims[0];

    // Initialization of shared variables
    if (thread_idx == 0 ) {
        node_idx = 0;
        body_idx = 0;
        node_mutex = SECTION_UNLOCK;
        body_mutex = SECTION_UNLOCK;
    }
    __syncthreads();

    // Critical section! Create a new node at shared index
    while(nodeBlocked) {
        if(SECTION_UNLOCK == atomicCAS(&node_mutex, SECTION_UNLOCK, SECTION_LOCK)) {
            // Are we over budget for nodes?
            // If so, stop tree construction,
            // this is a big time loss
            if (node_idx >= freeNodeCount) {
                atomicExch(&node_mutex, SECTION_UNLOCK);
                *return_idx = CUDA_ERROR_MEMORY_ALLOCATION;
                break;
            } else {
                // Create node instance
                node = new(freeNodes + node_idx) Node();
                *return_idx = node_idx;
                node_idx += 1;
                atomicExch(&node_mutex, SECTION_UNLOCK);
                nodeBlocked = false;
            }
        }
    }

    __syncthreads();
    
    if (nodeBlocked) {
        // Something went wrong - overbudget
        return;
    }

    for (i = 0; i < ndim; i++) {
        node->min_range[i] = dims[i] * thread_dim_lengths[i];
        node->max_range[i] = node->min_range[i] + thread_dim_lengths[i];
    }

    CountBodiesInNodeBoundsGPU(bodies, body_count, node, &last_encountered, ndim, &bound_count);

    if (bound_count > 1) {

    for (i = 0; i < (1 << ndim); i++) {            
          // Recurse on function to generate descending nodes
          // This function call will place the resulting child node pointer
          // into the respective variable
          GenerateTreeKernelHelper(bodies, body_count, ndim, \
            freeBodiesForBlock, freeNodesForBlock, freeBodyCount, freeNodeCount, node, &child_idx, i);

          if (child_idx != -1) {
              child = freeNodes + child_idx;
              node->AddChild(child);
              mass += child->cluster->mass;

              for (int j = 0; j < ndim; j++) {
                  pos[j] += child->cluster->pos[j] * child->cluster->mass;
              }

          } else if (body_idx >= freeBodyCount || node_idx >= freeNodeCount) {
            // Big problem, terminate kernel
            *return_idx = CUDA_ERROR_MEMORY_ALLOCATION;
            return;
          }
      }

      // Normalize center of mass vector by total mass of cluster
      for (int j = 0; j < ndim; j++) {
          pos[j] /= mass;
      }

      // Critical section! Create a new body at shared index
      while(bodyBlocked) {
        if(SECTION_UNLOCK == atomicCAS(&node_mutex, SECTION_UNLOCK, SECTION_LOCK)) {
          // Are we over budget for bodies?
          // If so, stop tree construction,
          // this is a big time loss
          if (body_idx >= freeBodyCount) {
            atomicExch(&body_mutex, SECTION_UNLOCK);
            *return_idx = CUDA_ERROR_MEMORY_ALLOCATION;
            return;
          } else {
            // Create cluster body instance
            node->cluster = new(freeBodies + body_idx) Body(&(pos[0]), mass, ndim);              
            body_idx += 1;
            atomicExch(&body_mutex, SECTION_UNLOCK);
            bodyBlocked = false;
          }
        }    

        // Are we over budget for bodies?
        // If so, stop tree construction,
        // this is a big time loss
        if (body_idx >= freeBodyCount) {
          return;
        }     
      }
  } else if (bound_count == 1) {
      // No need to create new body instance, just grab body residing in our domain
      node->cluster = &(bodies[last_encountered]);
  } else {
      // No bodies in our domain, don't do anything, return null
      *return_idx = CUDA_ERROR_MEMORY_ALLOCATION;
  }
}

// Counts the number of bodies within a specified domain
__device__ void CountBodiesInNodeBoundsGPU(Body *bodies, int body_count, \
        Node *node, int *last_encountered, int ndim, int *bound_count) {
    int body_idx;
    bool in_range;
    Body *body;

    *bound_count = 0;
    
    for (body_idx = 0; body_idx < body_count; body_idx++) {
        body = bodies + body_idx;
        BodyInRangeGPU(body, node->min_range, node->max_range, ndim, &in_range);
        if (in_range) {
            *bound_count += 1;
            *last_encountered = body_idx;
        }
    }
}

// Checks if a body is in a specified domain
__device__ void BodyInRangeGPU(Body *body, double *min, double *max, int ndim, bool *in_range) {
    int dim;
    *in_range = true;
    for (dim = 0; dim < ndim; min++) {
       if (body->pos[dim] < min[dim] || body->pos[dim] >= max[dim]) {
            *in_range = false;
            break;
       }
    }
}


// Generate tree helper function (stub)
__device__ void GenerateTreeHelper(Body *bodies, int body_count, int ndim, \
        Node *parent, int sector) {
}


// Helper function to compute forces
__device__ void ComputeForcesGPU(Body *body, Node *curr_node, int ndim, double *force) {
    double dist = 0, dim_dist;
    bool direct = false;
    bool in_range;

    // Check if body is in node range
    BodyInRangeGPU(body, curr_node->min_range, curr_node->max_range, ndim, &in_range);

    if (in_range) {
        // Get constant side length
        double side_len = curr_node->max_range[0] - curr_node->min_range[0];

        // Derive distance magnitude
        for (int i = 0; i < ndim; i++) {
            dim_dist = (curr_node->cluster->pos[i] - body->pos[i]);
            dist += (dim_dist * dim_dist);
        }
        // Square root sum of squares
        dist = sqrt(dist);

        // Check if threshold is met for clustering!
        if (side_len / dist < BH_THETA) {
            body->AddForceFrom(curr_node->cluster, force, ndim);
        } else {
            direct = true;
        }
    } else {
        direct = true;
    }

    // Looks like we weren't able to cluster :(
    // Will either have to recurse further or do a direct force computation
    if (direct) {
        // Loop through each child and recurse on them
        for (int i = 0; i < curr_node->numChildren; i++) {
            ComputeForcesGPU(body, curr_node->children[i], ndim, force);
        }

        // If no children, must be a single-bodied node, direct force calculation
        if (curr_node->numChildren == 0 && curr_node->cluster != body) {
            body->AddForceFrom(curr_node->cluster, force, ndim);
        }
    }
}

// Kernel function to update bodies using tree
__global__ void UpdateBodiesKernel(Body *bodies, int body_count, \
        Tree *tree, int ndim, double dt) {
    int thread_idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (thread_idx > body_count) {
      return;
    }
    int i, incr = (blockDim.x * gridDim.x);
    double force[DIM];
    
    for (i = thread_idx; i < body_count; i += incr) {
        memset(force, 0, ndim * sizeof(double));

        ComputeForcesGPU(bodies + i, tree->root, ndim, force);

        (bodies + i)->UpdateBodyPosition(force, dt, ndim);
    }
}

// Kernel function to perform direct simulation step
__global__ void DirectSimulationStepKernel(Body *bodies, int body_count, int ndim, double dt) {

    // Get offset for thread body
    int thread_idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (thread_idx > body_count) {
      return;
    }

    // Force vector
    double force[DIM];
    for (int i = 0; i < DIM; i++) {
      force[i] = 0;
    }

    // Get pointer to body
    Body *curr = bodies + thread_idx;

    // Calculate force imposed on every other body
    for (int i = 0; i < body_count; i++) {
      if (i != thread_idx) {
        curr->AddForceFrom(bodies + i, force, ndim);        
      }
    }

    // Synchronization barrier
    __syncthreads();

    // Update body position
    curr->UpdateBodyPosition(force, dt, ndim);
}


// Maps object pointers from a CPU space to a specified GPU space
void MapObjectPointers(Body *bodies, Body *bodiesDev, int bodyIdx, Node *nodes, Node *nodesDev, int nodeIdx) {
    // Create a temporary CPU buffer and copy over node data
    Node *tmp_nodes = (Node *) malloc(nodeIdx * sizeof(Node));
    memcpy(tmp_nodes, nodes, nodeIdx * sizeof(Node));

    // Loop through each node
    for (int node = 0; node < nodeIdx; node++) {

        // Get pointers to nodes in buffers
		Node *curr_node = nodes + node;
        Node *curr_node_tmp = tmp_nodes + node;

        // Point the parent in temporary buffer to GPU space
		if (curr_node->parent != NULL) {
			curr_node_tmp->parent = nodesDev + (curr_node->parent->index);
		}

        // Point all children in temporary buffer to GPU space
		for (int child = 0; child < curr_node->numChildren; child++) {
			curr_node_tmp->children[child] = nodesDev + (curr_node->children[child]->index);
		}

        // If it exists, point cluster in temporary buffer to GPU space
		if (curr_node->cluster != NULL && curr_node->cluster->index != -1) {
            curr_node_tmp->cluster = bodiesDev + (curr_node->cluster->index);
        }
	}

    // Copy object data to device now, and free temporary buffer
    cudaMemcpy(nodesDev, tmp_nodes, nodeIdx * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMemcpy(bodiesDev, bodies, bodyIdx * sizeof(Body), cudaMemcpyHostToDevice);
    free(tmp_nodes);
}

// Complete tree construction (second kernel function, not implemented)
Tree *CompleteTreeConstruction(Body *bodies, int body_count, int ndim, Node *nodes, int freeNodeCount) {
  return NULL;
}

// Performs GPU simulation procedure
bool RunSimulationGPU(Body *bodies, int body_count, int ndim, double *dim_ranges, FILE *fp, double tf, double dt, char mass_unit, char length_unit, char velocity_unit) {

    cudaError_t status;
    int retval;
    int body_bytes = body_count * sizeof(Body);
    double time = 0;

    Body *bodies_dev;
    Node *root_node = (Node *) MEM_ERR_PTR;
    Tree *sim_tree, *sim_tree_dev;

    // For Barnes-Hut implementation
    Body *freeBodies, *freeBodiesDev;
    Node *freeNodes, *freeNodesDev;
    int freeBodyCount = 0, freeNodeCount = 0;
    int bodyIdx = 0, nodeIdx = 0, allocated = MIN_ALLOC_STATIC_ELEMENTS;

    // Allocate device memory for body data
    status = cudaMalloc((void **) &bodies_dev, body_bytes);
    if (!verifyCuda(status)) {
      std::cout << "Allocation failed. Terminating" << std::endl;
      return false;
    }

    // Copy body data from host to device
    status = cudaMemcpy((void *) bodies_dev, (void *) bodies, body_count * sizeof(Body), cudaMemcpyHostToDevice);
    if (!verifyCuda(status)) {
      std::cout << "Data copy failed. Terminating" << std::endl;
      return false;
    }

    // Log data if associated flags are defined as such
    if (LOG_DATA && !OUTPUT_FINAL_ONLY) {
        WriteBodyData(fp, bodies, body_count, ndim, time, mass_unit, length_unit, velocity_unit);
    }

    // Specify kernel launch dimensions
    dim3 dimBlock(BLOCK_LEN);
    dim3 dimGrid((int) ceil((float)body_count / (float)BLOCK_LEN));

    // Loop through each timestep until simulation time exceeds upper limit
    while (time < tf) {

        std::cout << "Beginning iteration at time = " << time << std::endl;

        // Perform direct simulation
        if (RUN_DIRECT) {

          // Call direct simulation kernel
          DirectSimulationStepKernel<<<dimGrid, dimBlock>>>(bodies_dev, body_count, ndim, dt);
          cudaDeviceSynchronize();

          // Check for kernel-associated errors
          if (!verifyCuda(cudaGetLastError())) {
            std::cout << "Body updates failed!" << std::endl;
            return false;
          }

        } else if (GPU_TREE_GPU_ENABLED) {
            // Performs GPU tree construction and simulation (this won't work)
            GetSpatialRange(bodies, body_count, ndim, dim_ranges);
            double thread_dim_lengths[ndim];

            // Set initial free node and free body buffer sizes (scale by dimGrid)
            int freeNodeCount = BODIES_PER_BLOCK(body_count, dimGrid);
            int freeBodyCount = BODIES_PER_BLOCK(body_count, dimGrid);

            // Get the spatial dimensions
            for (int i = 0; i < ndim; i++) {
                thread_dim_lengths[i] = (dim_ranges[(i << 1) + 1] - dim_ranges[(i << 1)]) / \
                                        (BLOCK_LEN * GRID_LEN);
            }

            // Static tree construction -- loops until tree construction is successful
            while (true) {

              // Allocated space for nodes and bodies
              cudaMalloc((void **) &freeNodes, sizeof(Node) * freeNodeCount * (dimGrid.x * dimGrid.y * dimGrid.z));
              cudaMalloc((void **) &freeBodies, sizeof(Body) * freeBodyCount * (dimGrid.x * dimGrid.y * dimGrid.z));

              // The first abs(dimBlock) nodes should be independent trees
              GenerateTreeKernel<<<dimGrid, dimBlock>>>(bodies_dev, body_count, \
                      ndim, thread_dim_lengths, freeBodies, freeNodes, freeBodyCount, \
                      freeNodeCount, &retval);
              cudaDeviceSynchronize();

              if (retval == CUDA_ERROR_MEMORY_ALLOCATION) {
                // Need to make free body and free node buffers larger
                
                // Free what we've previously allocated
                cudaFree(freeBodies);
                cudaFree(freeNodes);

                // Double the size of the next allocation
                freeBodyCount *= 2;
                freeNodeCount *= 2;

                // Let's not go too crazy here, set a buffer size constraint
                if (freeBodyCount >= MAX_FREE_BUFFER || \
                  freeNodeCount >= MAX_FREE_BUFFER) {
                    return false;
                  }
              } else {
                // Tree construction worked!
                break;
              }
            }
            sim_tree = CompleteTreeConstruction(bodies_dev, body_count, ndim, freeNodes, freeNodeCount);
            if (sim_tree == NULL) {
              std::cout << "ERROR: Tree construction failed" << std::endl;
              return false;
            }

        } else if (GEN_TREE_STATIC) {

          // Performs static tree construction
          GetSpatialRange(bodies, body_count, ndim, dim_ranges);
          freeBodies = (Body *) malloc(allocated * sizeof(Body));
          freeNodes = (Node *) malloc(allocated * sizeof(Node));
          freeBodyCount = allocated;
          freeNodeCount = allocated;

          // Loops until tree construction is complete
          while (root_node == (Node *) MEM_ERR_PTR) {

            root_node = ConstructSimulationTreeStatic(bodies, body_count, \
                ndim, dim_ranges, NULL, 0, freeBodies, freeNodes, freeBodyCount, \
                freeNodeCount, &(bodyIdx), &(nodeIdx));

            // Grow size of object buffers
            if (root_node == (Node *) MEM_ERR_PTR) {

              // Scale buffer sizes, reallocate, reset indices
              allocated *= ALLOC_FACTOR_STATIC;
              freeBodies = (Body *) realloc(freeBodies, allocated * sizeof(Body));
              freeNodes = (Node *) realloc(freeNodes, allocated * sizeof(Node));
              freeBodyCount = allocated;
              freeNodeCount = allocated;
              bodyIdx = 0;
              nodeIdx = 0;         

              // Check for maximum allocation size
              // Failsafe, not necessarily needed
              if (allocated > MAX_ALLOC_STATIC) {
                std::cout << "Allocated buffers have grown too large. Terminating" << std::endl;
                free(freeBodies);
                free(freeNodes);
                exit(EXIT_FAILURE);
              }
            }
          }

          // Allocate device-side body buffer
          status = cudaMalloc((void **) &freeBodiesDev, allocated * sizeof(Body));
          
          if (!verifyCuda(status)) {
            std::cout << "Allocation failed. Terminating" << std::endl;
            return false;
          }
          
          // Allocate device-side node buffer
          status = cudaMalloc((void **) &freeNodesDev, allocated * sizeof(Node));
          
          if (!verifyCuda(status)) {
            std::cout << "Allocation failed. Terminating" << std::endl;
            return false;
          }

          
          // Maps objects from host to device
          MapObjectPointers(freeBodies, freeBodiesDev, bodyIdx, freeNodes, freeNodesDev, nodeIdx);
          
      
          // Create tree object and copy to device memory space
          Node *root_node_dev = freeNodesDev;
          sim_tree = new Tree(root_node_dev);
          cudaMalloc((void **) &(sim_tree_dev), allocated * sizeof(Tree)); 
          cudaMemcpy(sim_tree_dev, sim_tree, sizeof(Tree), cudaMemcpyHostToDevice);
        
        } else {

          // Constructs tree
          root_node = ConstructSimulationTree(bodies_dev, body_count, ndim, dim_ranges, NULL, 0);
          sim_tree = new Tree(root_node);

        }

        // Only run kernel for Barnes-Hut simulations
        if (!RUN_DIRECT) {
          UpdateBodiesKernel<<<dimGrid, dimBlock>>>(bodies_dev, body_count, sim_tree_dev, ndim, dt);
          cudaDeviceSynchronize();
  
          if (!verifyCuda(cudaGetLastError())) {
            std::cout << "Body updates failed!" << std::endl;
            return false;
          }

          // Deconstruct tree if not using static allocation
          if (!GEN_TREE_STATIC) {
            sim_tree->DeconstructTree();
          }
        }


        // Incremeent time
        time += dt;

        // Log data if flags are specified as such
        if (LOG_DATA && !OUTPUT_FINAL_ONLY) {
            status = cudaMemcpy((void *) bodies, (void *) bodies_dev, body_bytes, cudaMemcpyDeviceToHost);
            if (!verifyCuda(status)) {
              std::cout << "Copy back from device failed. Terminating" << std::endl;
              return false;
            }
            WriteBodyData(fp, bodies, body_count, ndim, time, mass_unit, length_unit, velocity_unit);
        }
    }

    // Log final iteration
    if (LOG_DATA && OUTPUT_FINAL_ONLY) {
        status = cudaMemcpy((void *) bodies, (void *) bodies_dev, body_bytes, cudaMemcpyDeviceToHost);
        if (!verifyCuda(status)) {
          std::cout << "Copy back from device failed. Terminating" << std::endl;
          return false;
        }
        WriteBodyData(fp, bodies, body_count, ndim, time, mass_unit, length_unit, velocity_unit);
    }

    return true;
}

int main(int argc, char **argv) {
    int retval;
    int ndim, body_count;
    double dt, tf;
    double *dim_ranges;
    const char outputFile[] = "out.txt";
    char *fname = (char *) malloc(MAX_FNAME_LEN * sizeof(char));
    FILE *fp;
    char mass_unit, length_unit, velocity_unit;
    time_t tstart, tend;

    memset(fname, 0, MAX_FNAME_LEN * sizeof(char));

    // Get Inputs
    retval = ProcessInputs(argc, argv, &ndim, fname, &dt, &tf, &mass_unit, &length_unit, &velocity_unit);
    if (retval != 0) {
        PrintHelp();
        return retval;
    }

    // Allocate space to define space barriers at each iteration
    dim_ranges = (double *) malloc(2 * ndim * sizeof(double));
    if (dim_ranges == NULL) {
        exit(EXIT_FAILURE);
    }

    // Read from file and generate bodies
    Body *bodies = ReadBodyData(fname, &body_count, ndim, mass_unit, length_unit, velocity_unit); 
    if (bodies == NULL) {
      printf("ERROR: Body data could not be read from file. Terminating\n");
      return false;
    }

    std::cout << "Simulating " << body_count << " bodies for " << \
        tf << " seconds with a timestep of dt = " << dt << " seconds" << std::endl;
    
    if (LOG_DATA) {
        fp = fopen(outputFile, "w");
        if (fp == NULL) {
            return false;
        }
    } else {
        fp = NULL;
    }

    tstart = clock();
    RunSimulationGPU(bodies, body_count, ndim, dim_ranges, fp, tf, dt, mass_unit, length_unit, velocity_unit); 
    tend = clock();

    float duration_ms = 1000 * (float) (tend - tstart) / (float) CLOCKS_PER_SEC;

    std::cout << "Simulation complete" << std::endl;
    printf("Duration: %.4f (ms)\n", duration_ms);


    if (LOG_DATA) {
        fclose(fp);
    }

    free(dim_ranges);
    free(fname);
    free(bodies);

    std::cout << "Terminating successfully" << std::endl;
    return 0;
}
