#ifndef HELPER_H
#define HELPER_H

# define CHECK( call )\
{\
    const cudaError_t error = call;\
    if( error != cudaSuccess) \
    {\
        cout << " Error " << __FILE__ << " : " << __LINE__ << endl;\
        cout << " Code : " << error << ", reason : " << cudaGetErrorString(error);\
        exit(1);\
    }\
}


#include <cuda.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include<algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <sstream>
#include <fstream>


__constant__ __device__ float M;


struct host_graph{
    int V,E,old_V;
    float M = 0.0;
    int *vtxComm, *edges, *edgesIdx, *oldToNewComm;
    float *weights, *commWeights;
};

struct device_graph{
    int *V, *E, *old_V;
    int *vtxComm, *edges, *edgesIdx, *oldToNewComm;
    float *weights, *commWeights, *vtx_Kvalue, *weightSum_InComm;
    int *newVtxComm, *CommSize, *partition;
};

struct new_graph{
    int *commDeg, *newId, *orderCom, *newEdges, *edgePos, *edgeTocurPos, *VtxStart;
    float *newWeights;
};

host_graph Read(std::string );
int findPrime(int );
int LargestDegree(host_graph& );
void prepareDevice(host_graph &, device_graph &, new_graph &);
void deleteDeviceVars(host_graph &, device_graph &, new_graph &);

#endif /* HELPER_H */