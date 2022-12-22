#ifndef PHASE1_H
#define PHASE1_H

#define FULL_MASK 0xffffffff

#include "helper.cuh"
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include<algorithm>
#include <cuda.h>
#include <cuda_runtime.h>


const int THREADS = 128;
const int numDBucketsI = 8;
const int bucketsI[8] = {0, 4, 8, 16, 32, 84, 319, INT_MAX};

const dim3 dimsI[7]{ 
    {4, 32}, {8, 16}, {16, 8}, {32, 4}, {32, 4}, {128, 1}, {128, 1}
};


bool ModularityOptimisationPhase(float , host_graph &, device_graph & );

float Calc_Q(int , float , device_graph );

void printOrgToComm(host_graph & , device_graph & );


#endif /* PHASE1_H */

