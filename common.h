#ifndef __COMMON_H__
#define __COMMON_H__


#include <cmath>
#include <string.h>
using namespace std;


#ifdef __CUDACC__
  #define CUDA_CALLABLE_MEMBER __host__ __device__
  #include <cuda.h>
  #include <cuda_runtime_api.h>
  #include <device_launch_parameters.h>
#else
  #define CUDA_CALLABLE_MEMBER
#endif 


// Static allocation defines
#define MIN_ALLOC_STATIC_ELEMENTS 100
#define ALLOC_FACTOR_STATIC 10
#define MAX_ALLOC_STATIC 1000000

// Pre-processor power calculation
#define POW_2(n) (1 << n)

// Pre-processor dimension calculation
#ifndef DIM
    #define DIM 2
#endif

// Compiler flag -- direct simulation
#ifdef DIRECT
    #define RUN_DIRECT 1
#else
    #define RUN_DIRECT 0
#endif

// Compiler flag -- static tree generation
#ifdef STATIC_TREE
    #define GEN_TREE_STATIC 1
#else
    #define GEN_TREE_STATIC 0
#endif

// Compiler flag -- enable logging
#ifdef LOG
    #define LOG_DATA 1
#else
    #define LOG_DATA 0
#endif

// Compiler flag -- log only the final iteration
#ifdef ONLY_FINAL
    #define OUTPUT_FINAL_ONLY 1
#else
    #define OUTPUT_FINAL_ONLY 0
#endif

// Preprocess the number of children per node
#define NUM_CHILDREN (POW_2(DIM))

// Body list contains this many elements at a minimum
#define MIN_ALLOC_BODIES 10

// Body list increases by this factor on very realloc
#define BODY_REALLOC_FACTOR 2

// Buffer size (bytes) for filename
#define MAX_FNAME_LEN 256

// Buffer size (bytes) of get_opt option buffer
#define MAX_OPT_LEN 64

// Buffer size (bytes) of output file write buffer 
#define FILE_WRITE_BUFFER_LEN 256

// Distance to extend the bounds of spatial domain past furthest elements
#define SPATIAL_BUFFER (0.5f)

// Theta threshold for Barnes-Hut algorithm
#define BH_THETA 0.5f

// Memory error pointer return for static tree generation
#define MEM_ERR_PTR 0xFFFFFFFF

/* Stores data related to a body or a cluster (of bodies) */
class Body {
    public:
        // Set the position vector of a body
        CUDA_CALLABLE_MEMBER void SetBodyPosition(double *pos_i, int ndim);

        // Update the existing position of the body with a force vector
        CUDA_CALLABLE_MEMBER void UpdateBodyPosition(double *pos_f, double dt, int ndim);

        // Constructor for body instance
        CUDA_CALLABLE_MEMBER Body(double *pos_i, double *vel_i, double mass, int ndim, char mass_unit, char length_unit, char velocity_unit);
        
        // Constructor for cluster instance
        CUDA_CALLABLE_MEMBER Body(double *pos_i, double mass, int ndim, int index_);
        CUDA_CALLABLE_MEMBER Body(double *pos_i, double mass, int ndim);

        // Derives the force contribution from another body/cluster
        CUDA_CALLABLE_MEMBER void AddForceFrom(Body *other, double *force, int ndim);

        // Position vector
        double pos[DIM];

        // Velocity vector
        double vel[DIM];

        // Mass of body/cluster
        double mass;

        // Relative instance index
        int index;
};

/* Contains data related to a node in the spatial-tree */
class Node {
    public:
        // Pointer to parent node in tree (NULL if node is root)
        Node *parent;

        // List of pointers to all children nodes
        Node *children[NUM_CHILDREN];

        // Bounding box values associated with node
        // Vector size equal to number of dimensions
        double min_range[DIM];
        double max_range[DIM];

        // Relative instance index
        int index;

        // Number of children, ranges from zero to number of dimensions
        int numChildren;

        // Pointer to the body (or bodies this node represents)
        // If this node clusters many nodes, this body represents
        //     a "center of mass" body that is used for computation
        // Otherwise, this simply points to the single body
        //     inside the bounding box preserved by this node
        Body *cluster;

        // Constructor function
        CUDA_CALLABLE_MEMBER Node ();
        
        // Constructor function
        CUDA_CALLABLE_MEMBER Node (int index_);

        // Appends a child to this node's list of children
        CUDA_CALLABLE_MEMBER int AddChild(Node *child);

        // Removes a child from list at index
        CUDA_CALLABLE_MEMBER int RemoveChild(int childIndex);


        // Assigns the parent node to this node
        CUDA_CALLABLE_MEMBER void SetParent(Node *par);
};

// Contains information regarding tree structure
class Tree {
    public:
        // Pointer to root node in tree
        Node *root;

        // Constructor that assigns root node
        Tree (Node *root_);

        // Prints basic information of each node in each layer
        void PrintTree(int ndim); 

        // Frees all data among all nodes in tree
        Tree *DeconstructTree();

    protected:
        // Helper function to print tree information
        void PrintTreeHelper(Node *node, int depth, int ndim);

        // Helper function to deconstruct tree
        void DeconstructTreeHelper(Node *node);
};

// Tree construction function
CUDA_CALLABLE_MEMBER Node *ConstructSimulationTree(Body *bodies, int body_count, int ndim, double *dim_ranges, Node *parent, int sector);

// Static tree construction function
CUDA_CALLABLE_MEMBER Node *ConstructSimulationTreeStatic(Body *bodies, int body_count, int ndim, \
        double *dim_ranges, Node *parent, int sector, Body *freeBodies, Node *freeNodes, \
        int freeBodyCount, int freeNodeCount, int *nodeIdx, int *bodyIdx);
   
// Counts the number of bodies in a region, used for tree construction
CUDA_CALLABLE_MEMBER int CountBodiesInBound(Body *bodies, int body_count, int ndim, double *min, \
        double *max, int *last_encountered);

// Checks if a body in a bound
CUDA_CALLABLE_MEMBER bool BodyInRange(Body *body, double *min, double *max, int ndim);

// Write body data to an output stream
void WriteBodyData(FILE *fp, Body *bodies, int body_count, int ndim, double time, char mass_unit, char length_unit, char velocity_unit);

// Gets the domains bounds
int GetSpatialRange(Body *bodies, int body_count, int ndim, double *dim_ranges);

// Prints basic help message with arguments and description
int PrintHelp();

// Processes command-line arguments
int ProcessInputs(int argc, char **argv, int *ndim, char *fname, double *dt, double *tf, char *mass_unit, char *length_unit, char *velocity_unit);

// Reads body data as input to begin simulation
Body *ReadBodyData(char *fname, int *body_count, int ndim, char mass_unit, char length_unit, char velocity_unit);


#endif
