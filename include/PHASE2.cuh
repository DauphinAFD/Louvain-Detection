#ifndef PHASE2_H
#define PHASE2_H

#define FULL_MASK 0xffffffff


#include "helper.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/distance.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include<algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

//const int THREADS = 128;
const int numCBuckets = 4;
const int bucketsII[4] = {0, 127, 479, INT_MAX};
const dim3 dimsII[3] { 
    {32, 4}, {128, 1}, {128, 1}
};

void aggregateCommunityPhase(host_graph &, device_graph &, new_graph &);

#endif /* PHASE2_H */
