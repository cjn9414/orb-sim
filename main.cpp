#include <cstdio>
#include <iostream>
#include "common.h"
#include "main.h"
#include <ctime>

// Runs direct simulation, O(n*n) operation but simple
int DirectSimulationStep(Body *bodies, int body_count, int ndim, double dt) {
    double forces[ndim];

    // Outer-loop through each body
    for (int i = 0; i < body_count; i++) {
        // Reset force vector
        memset(forces, 0, ndim * sizeof(double));

        // Inner-loop through each body
        for (int j = 0; j < body_count; j++) {
            // Add force so long as bodies are different
            if (i != j) {
                bodies[i].AddForceFrom(bodies + j, forces, ndim);
            }
        }

        // After final force vector is derived, update body position and velocity
        bodies[i].UpdateBodyPosition(forces, dt, ndim);
    }
    return 0;
}

// Main part of Barnes-Hut - Uses tree to cluster bodies for force contributions
int ComputeForces(Body *body, Node *curr_node, int ndim, double *force) {
    
    double dist = 0, dim_dist;
    bool direct = false;

    // Check if body is in node range
    if (!BodyInRange(body, curr_node->min_range, curr_node->max_range, ndim)) {
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
            ComputeForces(body, curr_node->children[i], ndim, force);
        }

        // If no children, must be a single-bodied node, direct force calculation
        if (curr_node->numChildren == 0 && curr_node->cluster != body) {
            body->AddForceFrom(curr_node->cluster, force, ndim);
        }
    }
    return 0;
}

// Main function for Barnes-Hut iteration. Computes forces and updates positions
int UpdateBodies(Body *bodies, int body_count, Tree *tree, int ndim, double dt) {
    // Initialize force vector
    double force[ndim];

    // Loop through each body and derive force vector
    for (int i = 0; i < body_count; i++) {
        // Reset force vector
        memset(force, 0, ndim * sizeof(double));

        // Recursively derive forces on body
        ComputeForces(bodies + i, tree->root, ndim, force);

        // Updates position of body with new force vector
        (bodies + i)->UpdateBodyPosition(force, dt, ndim);
    }
    return 0;
}

// Main function - Parses arguments, reads and writes to associated files, performs simulation
int main(int argc, char **argv) {
    int retval;
    int ndim, body_count;
    double dt, tf, time = 0;
    double *dim_ranges;
    const char outputFile[] = "out.txt";
    char *fname = (char *) malloc(MAX_FNAME_LEN * sizeof(char));
    char mass_unit, length_unit, velocity_unit;
    time_t tstart, tend;

    // Only for static tree generation
    Node *freeNodes;
    Body *freeBodies;
    int freeBodyCount = 0, freeNodeCount = 0;
    int bodyIdx = 0, nodeIdx = 0, allocated = MIN_ALLOC_STATIC_ELEMENTS;

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

    std::cout << "Simulating " << body_count << " bodies for " << \
        tf << " seconds with a timestep of dt = " << dt << " seconds" << std::endl;

    // Open output file for writing
    FILE *fp = fopen(outputFile, "w");
    if (fp == NULL) {
        exit(EXIT_FAILURE);
    }

    if (LOG_DATA && !OUTPUT_FINAL_ONLY) {
        WriteBodyData(fp, bodies, body_count, ndim, time, mass_unit, length_unit, velocity_unit);
    }

    Node *sim_tree_root = (Node *) MEM_ERR_PTR;

    tstart = clock();

    // Run for number of iterations
    while (time < tf) {
        std::cout << "Beginning iteration at time = " << time << std::endl;

        // Log data conditionally
        if (LOG_DATA && !OUTPUT_FINAL_ONLY) {
            WriteBodyData(fp, bodies, body_count, ndim, time, mass_unit, length_unit, velocity_unit);
        }

        // Check if we're running a direct simulation or using Barnes-Hut
        if (RUN_DIRECT) {
            // Run direct simulation step, ~O(n^2)
            retval = DirectSimulationStep(bodies, body_count, ndim, dt);
        } else {
            // First get the bounds of the domain
            retval = GetSpatialRange(bodies, body_count, ndim, dim_ranges);

            // Are we statically generating the tree?
            if (GEN_TREE_STATIC) {
                // Perform initial allocation of static objects
                freeBodies = (Body *) malloc(allocated * sizeof(Body));
                freeNodes = (Node *) malloc(allocated * sizeof(Node));
                freeBodyCount = allocated;
                freeNodeCount = allocated;

                // Loop until tree is created successfully
                while (sim_tree_root == (Node *) MEM_ERR_PTR) {
                    // Try to construct the tree
                    sim_tree_root = ConstructSimulationTreeStatic(bodies, body_count, \
                            ndim, dim_ranges, NULL, 0, freeBodies, freeNodes, freeBodyCount, \
                            freeNodeCount, &(bodyIdx), &(nodeIdx));

                    // Did the tree construction throw an error?
                    if (sim_tree_root == (Node *) MEM_ERR_PTR) {
                        // We need to grow the number of allocated objects
                        allocated *= ALLOC_FACTOR_STATIC;
                        freeBodies = (Body *) realloc(freeBodies, allocated * sizeof(Body));
                        freeNodes = (Node *) realloc(freeNodes, allocated * sizeof(Node));
                        freeBodyCount = allocated;
                        freeNodeCount = allocated;

                        // Reset the body indexes, tree construction starts from beginning
                        bodyIdx = 0;
                        nodeIdx = 0;

                        // Make sure we don't go overboard with static allocation
                        // This is mostly a failsafe and isn't technically necessary
                        if (allocated > MAX_ALLOC_STATIC) {
                            std::cout << "Allocated buffers have grown too large. Terminating" << std::endl;
                            free(freeBodies);
                            free(freeNodes);
                            exit(EXIT_FAILURE);
                        }
                    }
                }
            } else {
                // Dynamic tree construction
                sim_tree_root = ConstructSimulationTree(bodies, body_count, ndim, dim_ranges, NULL, 0);
            }

            // Create the tree object from the root
            Tree *sim_tree = new Tree(sim_tree_root);

            // Perform Barnes-Hut algorithm
            UpdateBodies(bodies, body_count, sim_tree, ndim, dt);

            // Need to deconstruct the tree unless generated statically
            if (!GEN_TREE_STATIC) {
                sim_tree->DeconstructTree();
            }
        }

        // Update the time
        time += dt;
    }

    // Get the end time for simulation
    tend = clock();

    // Calculate simulation duration
    float duration_ms = 1000 * (float) (tend - tstart) / (float) CLOCKS_PER_SEC;

    std::cout << "Simulation complete" << std::endl;
    printf("Duration: %.4f (ms)\n", duration_ms);
    
    // Log output data if associated flags are specified as such
    if (LOG_DATA && OUTPUT_FINAL_ONLY) {
        WriteBodyData(fp, bodies, body_count, ndim, time, mass_unit, length_unit, velocity_unit);
    }

    // Close output file
    fclose(fp);
    free(dim_ranges);
    free(fname);
    free(bodies);

    // Need to free statically allocated pointers
    if (GEN_TREE_STATIC) {
        free(freeBodies);
        free(freeNodes);
    }

    std::cout << "Terminating successfully" << std::endl;
    return 0;
}
