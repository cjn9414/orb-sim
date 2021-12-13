#include <cstdio>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <ctime>
#include "common.h"
using namespace std;

#define G (6.67430e-11)

#define PARSEC_TO_M (3.086e+16)
#define M_TO_PARSEC (3.24078e-17)

#define SUN_MASS_TO_KG (1.989e+30)
#define KG_TO_SUN_MASS (5.02785e-31)

#define LB_TO_KG (0.453592f)
#define KG_TO_LB (2.20462f)

#define FT_TO_M (0.3048f)
#define M_TO_FT (3.28084f)

#define KM_TO_M (1000.0f)
#define M_TO_KM (0.001f)

#define MASS_KG_CHAR 'k'
#define MASS_SUN_CHAR 's'
#define MASS_POUND_CHAR 'l'
#define LENGTH_METER_CHAR 'm'
#define LENGTH_PARSEC_CHAR 'p'
#define LENGTH_FEET_CHAR 'f'
#define VELOCITY_M_PER_S_CHAR 'm'
#define VELOCITY_KM_PER_S_CHAR 'k'

#define CONVERT_FROM (0)
#define CONVERT_TO (1)

#define MASS_FACTOR(M, convert_to) ( (M) == MASS_KG_CHAR ? 1 : ((M) == MASS_SUN_CHAR ? ( convert_to ? SUN_MASS_TO_KG : KG_TO_SUN_MASS ) : ( convert_to ? LB_TO_KG : KG_TO_LB ) ) )
#define LENGTH_FACTOR(L, convert_to) ( (L) == LENGTH_METER_CHAR ? 1 : ((L) == LENGTH_PARSEC_CHAR ? ( convert_to ? PARSEC_TO_M : M_TO_PARSEC ) : ( convert_to ? FT_TO_M : M_TO_FT ) ) )
#define VELOCITY_FACTOR(V, convert_to) ( (V) == VELOCITY_M_PER_S_CHAR ? 1 : (convert_to ? KM_TO_M : M_TO_KM) )

/*********************************************/
/* Node constructor and associated functions */
/*********************************************/

// Node constructor, resets children counter 
CUDA_CALLABLE_MEMBER Node::Node (int index_) {
    numChildren = 0;
    index = index_;
}

// Node constructor, resets children counter 
CUDA_CALLABLE_MEMBER Node::Node () {
    numChildren = 0;
    index = -1;
}

// Appends child to node's child list
CUDA_CALLABLE_MEMBER int Node::AddChild(Node *child) {
    // Increment children counter and append new child
    children[numChildren++] = child;

    // Set pointer to parent
    child->SetParent(this);
    return 0;
}

// Removes child from node's child list
CUDA_CALLABLE_MEMBER int Node::RemoveChild(int childIndex) {
    // Don't remove children if no child exists
    if (numChildren <= 0) return -1;

    // Overwrite child's location in list and shift 
    //     subsequent children down
    for (int i = childIndex; i < numChildren-1; i++) {
        children[i] = children[i+1];
    }

    // Nullify final index
    children[--numChildren] = NULL;
    
    return 0;
}

// Set parent pointer
CUDA_CALLABLE_MEMBER void Node::SetParent(Node *par) {
    parent = par;
}

/*********************************************/
/* Tree constructor and associated functions */
/*********************************************/

// Constructor, set root node
Tree::Tree (Node * root_) {
    root = root_;
}

// Prints out basic information of each node and children recursively
void Tree::PrintTree(int ndim) {
    cout << "Starting at root node..." << endl;

    // Call helper function on each child
    for (int i = 0; i < root->numChildren; i++) {
        PrintTreeHelper(root->children[i], 1, ndim);
    }
}

// Helper function to print tree information
void Tree::PrintTreeHelper(Node *node, int depth, int ndim) {
    int i;
    
    // Print tree depth
    cout << "At depth " << depth << endl;

    // Print center of mass
    cout << "\tCenter: (";
    for (i = 0; i < ndim; i++) {
        cout << node->cluster->pos[i] << ", ";
    }
    cout << endl;
    
    // Print mass
    cout << "\tMass: " << node->cluster->mass << endl;

    // Print spatial ranges of node
    cout << "\tMin: (";
    for (i = 0; i < ndim; i++) {
        cout << node->min_range[i] << ", ";
    }
    cout << endl;
    cout << "\tMax: (";
    for (i = 0; i < ndim; i++) {
        cout << node->max_range[i] << ", ";
    }
    cout << endl;

    // Recurse on each node's child and print same information
    for (int i = 0; i < node->numChildren; i++) {
        PrintTreeHelper(node->children[i], depth+1, ndim);
    }
}

// Performs tree deconstruction (frees all allocated data) 
Tree *Tree::DeconstructTree() {
    // Call helper function on each child of the root
    for (int i = 0; i < root->numChildren; i++) {
        DeconstructTreeHelper(root->children[i]);
    }
    // Delete root and finally the tree itself
    delete root;
    delete this;
    return NULL;
}

// Helper function to deconstruct tree
void Tree::DeconstructTreeHelper(Node *node) {
    // Recurse on each child node before deleting anything
    for (int i = 0; i < node->numChildren; i++) {
        DeconstructTreeHelper(node->children[i]);
    }
    // Now delete node (tree deleted bottom-up)
    delete node;
}

/*********************************************/
/* Body constructor and associated functions */
/*********************************************/

// Body constructor - Initializes position, velocity and mass
CUDA_CALLABLE_MEMBER Body::Body(double *pos_i, double *vel_i, double mass_, int ndim, char mass_unit, char length_unit, char velocity_unit) {
    int i;
    for (i = 0; i < ndim; i++) {
      pos[i] = LENGTH_FACTOR(length_unit, CONVERT_TO) * pos_i[i];
      vel[i] = VELOCITY_FACTOR(velocity_unit, CONVERT_TO) * vel_i[i];
    }
    mass = MASS_FACTOR(mass_unit, CONVERT_TO) * mass_;
}


// Body constructor for clusters - Initializes position and mass
CUDA_CALLABLE_MEMBER Body::Body(double *pos_i, double mass_, int ndim, int index_) {
    int i;
    for (i = 0; i < ndim; i++) {
      pos[i] = pos_i[i];
    }
    mass = mass_;
    index = index_;
}

// Body constructor for clusters - Initializes position and mass
CUDA_CALLABLE_MEMBER Body::Body(double *pos_i, double mass_, int ndim) {
    int i;
    for (i = 0; i < ndim; i++) {
      pos[i] = pos_i[i];
    }
    mass = mass_;
    index = -1;
}

// Sets only the position of a pre-initialized body instance
CUDA_CALLABLE_MEMBER void Body::SetBodyPosition(double *pos_i, int ndim) {
    int i;
    for (i = 0; i < ndim; i++) {
      pos[i] = pos_i[i];
    }
}

// Updates the position of a body object given a force vector and a timestep
CUDA_CALLABLE_MEMBER void Body::UpdateBodyPosition(double *forces, double dt, int ndim) {
    int i;
    for (i = 0; i < ndim; i++) {
        vel[i] += forces[i] * dt / mass;
        pos[i] += vel[i] * dt;
    }
}

// Adds the force contribution from another body
CUDA_CALLABLE_MEMBER void Body::AddForceFrom(Body *other, double *force, int ndim) {
    double dist[DIM];
    double abs_d = 0;
    int i;

    // Calculate the distance vector and distance magnitude
    for (i = 0; i < ndim; i++) {
        dist[i] = (other->pos[i] - this->pos[i]);
        abs_d += (dist[i] * dist[i]);
    }

    // Distance is square-root of sum-of-squares in each dimension
    abs_d = sqrt(abs_d);
    
    // Universal force equation applied to each dimension to derive force vector
    for (i = 0; i < ndim; i++) {
        force[i] += (G * (this->mass * other->mass) * dist[i] / (abs_d * abs_d * abs_d));
    }
}

// Constuct tree using constant size for each domain partition
// This function starts at the root node and recursively divides
//     the spatial domain until each bound occupies exactly one node
//     (any bounds with zero nodes are disregarded, not added to tree)
CUDA_CALLABLE_MEMBER Node *ConstructSimulationTreeStatic(Body *bodies, int body_count, int ndim, \
        double *dim_ranges, Node *parent, int sector, Body *freeBodies, Node *freeNodes, int freeBodyCount, int freeNodeCount, int *bodyIdx, int *nodeIdx) {
    
    int last_encountered, bound_count;

    // Construct new node (root node if not at any recursion depth)
    Node *node = new (freeNodes + *nodeIdx) Node(*nodeIdx);
    *nodeIdx += 1;

    if (*nodeIdx == freeNodeCount) {
        return (Node *) MEM_ERR_PTR;
    }

    // Check if at root, need to get the full bounding box if so
    if (parent == NULL) {
        for (int i = 0; i < ndim; i++) {
            node->min_range[i] = dim_ranges[i << 1];
            node->max_range[i] = dim_ranges[(i << 1) + 1];
        }
    } else {
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
    }
    
    // Get number of bodies in recently derived bounds
    bound_count = CountBodiesInBound(bodies, body_count, ndim, \
            node->min_range, node->max_range, &last_encountered);


    // Check if we need to recurse again
    if (bound_count > 1) {
        // Create new position vector and mass for cluster at this node
        double pos[DIM];
        double mass = 0.0;

        // Number of children will not exceed 2^(#Dimensions)

        // Loop through each divide of current spatial domain
        for (int i = 0; i < (1 << ndim); i++) {

            // Looks like we're going down the rabbit hole again
            Node *child = ConstructSimulationTreeStatic(bodies, body_count, ndim, dim_ranges, node, i, freeBodies, freeNodes, freeBodyCount, freeNodeCount, bodyIdx, nodeIdx); 

            if (child == (Node *) MEM_ERR_PTR) {
                return (Node *) MEM_ERR_PTR;
            }

            // Check for null (meaning that no body exists within this child's bounds)
            if (child != NULL) {
                // Add child to list
                node->AddChild(child);

                // Append child's mass to sum
                mass += child->cluster->mass;
                
                // Loop through each dimension and update center of mass vector with child contribution
                for (int j = 0; j < ndim; j++) {
                    pos[j] += child->cluster->pos[j] * child->cluster->mass;
                }
            }
        }

        // Normalize center of mass vector by total mass of cluster
        for (int j = 0; j < ndim; j++) {
            pos[j] /= mass;
        }

        // Create cluster body instance
        node->cluster = new (freeBodies + *bodyIdx) Body(pos, mass, ndim, *bodyIdx);
        *bodyIdx += 1;

        if (*bodyIdx == freeBodyCount) {
            return (Node *) MEM_ERR_PTR;
        }

    } else if (bound_count == 1) {
        // No need to create new body instance, just grab body residing in our domain
        node->cluster = &(bodies[last_encountered]);
    } else {
        // No bodies in our domain, don't do anything, return null
        node = NULL;
    }
    return node;
}

// Constuct tree using constant size for each domain partition
// This function starts at the root node and recursively divides
//     the spatial domain until each bound occupies exactly one node
//     (any bounds with zero nodes are disregarded, not added to tree)
CUDA_CALLABLE_MEMBER Node *ConstructSimulationTree(Body *bodies, int body_count, int ndim, \
        double *dim_ranges, Node *parent, int sector) {
    
    int last_encountered, bound_count;

    // Construct new node (root node if not at any recursion depth)
    Node *node = new Node();

    // Check if at root, need to get the full bounding box if so
    if (parent == NULL) {
        for (int i = 0; i < ndim; i++) {
            node->min_range[i] = dim_ranges[i << 1];
            node->max_range[i] = dim_ranges[(i << 1) + 1];
        }
    } else {
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
    }
    
    // Get number of bodies in recently derived bounds
    bound_count = CountBodiesInBound(bodies, body_count, ndim, \
            node->min_range, node->max_range, &last_encountered);


    // Check if we need to recurse again
    if (bound_count > 1) {
        // Create new position vector and mass for cluster at this node
        double pos[DIM];
        double mass = 0.0;

        // Number of children will not exceed 2^(#Dimensions)

        // Loop through each divide of current spatial domain
        for (int i = 0; i < (1 << ndim); i++) {

            // Looks like we're going down the rabbit hole again
            Node *child = ConstructSimulationTree(bodies, body_count, ndim, dim_ranges, node, i); 

            // Check for null (meaning that no body exists within this child's bounds)
            if (child != NULL) {
                // Add child to list
                node->AddChild(child);

                // Append child's mass to sum
                mass += child->cluster->mass;
                
                // Loop through each dimension and update center of mass vector with child contribution
                for (int j = 0; j < ndim; j++) {
                    pos[j] += child->cluster->pos[j] * child->cluster->mass;
                }
            }
        }

        // Normalize center of mass vector by total mass of cluster
        for (int j = 0; j < ndim; j++) {
            pos[j] /= mass;
        }

        // Create cluster body instance
        node->cluster = new Body(pos, mass, ndim);

    } else if (bound_count == 1) {
        // No need to create new body instance, just grab body residing in our domain
        node->cluster = &(bodies[last_encountered]);
    } else {
        // No bodies in our domain, don't do anything, return null
        node = NULL;
    }
    return node;
}

// Count the number of bodies in a given bound
CUDA_CALLABLE_MEMBER int CountBodiesInBound(Body *bodies, int body_count, int ndim, double *min, \
        double *max, int *last_encountered) { 
    int body_idx;
    int bound_count = 0;
    Body *body;

    // Loop through each body
    for (body_idx = 0; body_idx < body_count; body_idx++) {
        body = bodies + body_idx;
        // Check if body is in range, increment if so
        if (BodyInRange(body, min, max, ndim)) {
            bound_count++;
            // Mark body as most recently encountered
            *last_encountered = body_idx;
        }
    }
    return bound_count;
}

// Check if given body is in the given range
CUDA_CALLABLE_MEMBER bool BodyInRange(Body *body, double *min, double *max, int ndim) {
    // Loop over each dimension
    for (int dim = 0; dim < ndim; dim++) {
        // Simple bound check, return immediately if out of bounds
        if (body->pos[dim] < min[dim] || body->pos[dim] >= max[dim]) {
            return false;
        }
    }
    return true;
}


// Writes out mass, position and velocity data for each body at given time
void WriteBodyData(FILE *fp, Body *bodies, int body_count, int ndim, double time, char mass_unit, char length_unit, char velocity_unit) {
    int i, j;
    char buffer[FILE_WRITE_BUFFER_LEN];
    int offset;
    offset = sprintf(buffer, "Body data at time = %.4f\n", time);
    fwrite(buffer, sizeof(char), offset, fp); 
    
    for (i = 0; i < body_count; i++) {
        Body body = bodies[i];
        offset = sprintf(buffer, "%.4f ", MASS_FACTOR(mass_unit, CONVERT_FROM) * body.mass);
        for (j = 0; j < ndim; j++) {
            offset += sprintf(buffer + offset, "%.4f ", LENGTH_FACTOR(length_unit, CONVERT_FROM) * body.pos[j]);
        }
        for (j = 0; j < ndim; j++) {
            offset += sprintf(buffer + offset, "%.4f ", VELOCITY_FACTOR(velocity_unit, CONVERT_FROM) * body.vel[j]);
        }
        buffer[offset] = '\n';
        fwrite(buffer, sizeof(char), offset+1, fp); 
    }
}

// Derive the domain for current iteration
int GetSpatialRange(Body *bodies, int body_count, int ndim, double *dim_ranges) {
    int body_idx, dim;
    double pos;
    Body *body = NULL;

    // Loop through each body
    for (body_idx = 0; body_idx < body_count; body_idx++) {
        body = bodies + body_idx;
        // Loop through each dimension
        for (dim = 0; dim < ndim; dim++) {
            // Simple max value checks
            pos = body->pos[dim];
            if (pos < dim_ranges[dim << 1]) {
                dim_ranges[dim << 1] = pos;
            } else if (pos > dim_ranges[(dim << 1) + 1]) {
                dim_ranges[(dim << 1) + 1] = pos;
            }
        }
    }

    // Extend the bounds of the derived domain by constant value
    for (dim = 0; dim < ndim; dim++) {
        dim_ranges[dim << 1] -= SPATIAL_BUFFER;
        dim_ranges[(dim << 1) + 1] += SPATIAL_BUFFER;
    }
    return 0;
}

// Prints simple help message with arguments and description
int PrintHelp() {
    std::cout << "Description: Performs n-body simulation \
        using Barnes-Hut clustering algorithm." << std::endl << std::endl;

    std::cout << "Options:" << std::endl;
    std::cout << "\t-n <DIM>   Provide the number of dimensions for simulation              " << std::endl;
    std::cout << "\t-f <FNAME> Provide the name of file containing body data                " << std::endl;
    std::cout << "\t-d <DELTA> Provide the simulation timestep (seconds)                    " << std::endl;
    std::cout << "\t-t <END>   Provide the simulation end time (seconds)                    " << std::endl;
    std::cout << "\t-m <UNIT>  Provide the unit for mass: kg (k), sun mass (s), lbs (p)     " << std::endl;
    std::cout << "\t-l <UNIT>  Provide the unit for length: meters (m), parsecs (p), ft (f) " << std::endl;
    std::cout << "\t-v <UNIT>  Provide the unit for velocity: m/s (m), km/s (k)             " << std::endl;
    return 0;
}

// Reads input body data from provided file
Body *ReadBodyData(char *fname, int *body_count, int ndim, char mass_unit, char length_unit, char velocity_unit) {
    char line[256];
    char *token;
    int arg_count, line_count = 0;
    double mass;
    double pos[ndim], vel[ndim];
    int vector_index;
    FILE *fd;
    int allocated_bodies = MIN_ALLOC_BODIES;

    *body_count = 0;
    
    // Initial body list allocation
    Body *bodies = (Body *) malloc(allocated_bodies * sizeof(Body));
    if (bodies == NULL) {
        return NULL;
    }

    // Try to obtain file handle
    fd = fopen(fname, "r");
    if (fd == NULL) {
        return NULL;
    }

    // Loop through each line of data file
    while (fgets(line, sizeof(line), fd)) { 
        // Use strtok to parse each line
        token = strtok(line, " ");
        arg_count = 0;

        // Reset vectors
        vector_index = 0;
        
        // Loop through split line to obtain each parameter
        while (token != NULL) {
            // Assign incoming values according to index
            if (arg_count == 0) {
                mass = (double) atof(token);
            } else if (arg_count <= ndim) {
                pos[vector_index++] = (double) atof(token);
                if (vector_index == ndim) {
                    vector_index = 0;
                }
            } else if (arg_count <= (ndim << 1)) {
                vel[vector_index++] = (double) atof(token);
            } else {
                std::cout << "WARN: Additional argument on line " << line_count \
                    << " argument " << arg_count << ". Ignoring." << std::endl;
            }

            // Get next delimited string of line
            token = strtok(NULL, " ");
            arg_count++;
        }

        // Check for expected argument count
        if (arg_count == (1 + (ndim << 1))) {
            bodies[*body_count] = Body(pos, vel, mass, ndim, mass_unit, length_unit, velocity_unit);
            *body_count += 1;
        } else if (arg_count > 1) {
            // Not a newline and not enough arguments provided on line
            std::cout << "ERROR: Too few parameters on line " << line_count << \
                ". Number of parameters: " << arg_count << std::endl;
        }

        // Check if we've reached the end of the current list
        if (*body_count >= allocated_bodies) {
            // We need to reallocate to grow the list
            allocated_bodies *= BODY_REALLOC_FACTOR;
            bodies = (Body *) realloc(bodies, allocated_bodies * sizeof(Body));

            // Make sure reallocation was successful
            if (bodies == NULL) {
                return NULL;
            }
        } 
        line_count++;
    }
    fclose(fd);
    return bodies;

}

// Parse input arguments
int ProcessInputs(int argc, char **argv, int *ndim, char *fname, double *dt, double *tf, char *mass_unit, char *length_unit, char *velocity_unit) {
    const char options[MAX_OPT_LEN] = "n:f:d:t:m:l:v:\0";
    bool fname_flag = false;
    int opt;

    // Assign default argument values in case they were not provided
    *ndim = 2;
    *dt = 1.0f;
    *tf = 10000.0f;
    *mass_unit = MASS_KG_CHAR;
    *length_unit = LENGTH_METER_CHAR;
    *velocity_unit = VELOCITY_M_PER_S_CHAR;

    // Use getopt and loop through each argument
    while ((opt = getopt(argc, argv, options)) != -1) {
        switch (opt) {
            case 'n':
                // Get number of dimensions
                *ndim = (int) atoi(optarg);

                // Limit to two or three for now
                if (*ndim < 2 || *ndim > 3) {
                    std::cout << "ERROR: Dimensionality limited to\
                        two or three dimensions." << std::endl;
                    return EXIT_FAILURE;
                }
                break;
            case 'f':
                // Get filename and set filename flag
                memcpy(fname, optarg, strnlen(optarg, MAX_FNAME_LEN));
                fname_flag = true;
                break;
            case 'd':
                // Get delta on each iteration, make sure it's positive
                *dt = (double) atof(optarg);
                if (*dt <= 0) {
                    std::cout << "ERROR: Timestep must be greater than zero." << std::endl;
                    return EXIT_FAILURE;
                }
                break;
            case 't':
                // Get time upper-bound, make sure it's positive
                *tf = (double) atof(optarg);
                if (*tf <= 0) {
                    std::cout << "ERROR: Final time must be greater than zero." << std::endl;
                    return EXIT_FAILURE;
                }
                break; 
            case 'm':
                // Get the mass unit
                *mass_unit = (char) optarg[0];
                if (*mass_unit != MASS_KG_CHAR && *mass_unit != MASS_SUN_CHAR && *mass_unit != MASS_POUND_CHAR) {
                    std::cout << "ERROR: Unknown argument for mass unit option: " << *mass_unit << std::endl;
                    return EXIT_FAILURE;
                }
                break;
            case 'l':
                // Get the length unit
                *length_unit = (char) optarg[0];
                if (*length_unit != LENGTH_METER_CHAR && *length_unit != LENGTH_PARSEC_CHAR && *length_unit != LENGTH_FEET_CHAR) {
                    std::cout << "ERROR: Unknown argument for length unit option:" << *length_unit << std::endl;
                    return EXIT_FAILURE;
                }
                break;
            case 'v':
                // Get the length unit
                *velocity_unit = (char) optarg[0];
                if (*velocity_unit != VELOCITY_M_PER_S_CHAR && *velocity_unit != VELOCITY_KM_PER_S_CHAR) {
                    std::cout << "ERROR: Unknown argument for velocity unit option:" << *velocity_unit << std::endl;
                    return EXIT_FAILURE;
                }
                break;
            case '?':
                // Unexpected argument or argument not provided with known option
                std::cout << "ERROR: Unknown option: '-" << optopt << "'" << std::endl;
                return EXIT_FAILURE;
            default:
                // Not sure what happened here, let's just throw a generic error
                std::cout << "ERROR: Unexpected result " << opt << \
                     " at line " << __LINE__ << std::endl;
                return EXIT_FAILURE;
        }
    }

    // Make sure we got the filename argument
    if (!fname_flag) {
        std::cout << "ERROR: Body data filename not provided. Terminating." << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}

