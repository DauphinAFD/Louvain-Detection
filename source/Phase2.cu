#include "../include/PHASE2.cuh"
using namespace std;
const int THREADS = 128;

struct CheckBucketPhase2{
    int minDeg, maxDeg;
	int *CommDeg;

	CheckBucketPhase2(int minimum, int maximum, int *comunityDegree) {
		minDeg = minimum;
		maxDeg = maximum;
		CommDeg = comunityDegree;
	}
	__host__ __device__ bool operator()(const int &v) const {
		int Deg = CommDeg[v];
		return Deg > minDeg && Deg <= maxDeg;
	}
};

__device__ int HashII(int val, int idx, int primeNo){
	int h1 = val % primeNo;
	int h2 = 1 + (val % (primeNo - 1));
	return (h1 + idx * h2) % primeNo;
}

__device__ int HashArraysII(int comm, int primeNo, int hashOffset, float weight, int *hashComm, float *hashWeight){
    int it = 0;
    while(true) {
        int curPos = hashOffset + HashII(comm, it++, primeNo);
        if(hashComm[curPos] == comm) {
            atomicAdd(&hashWeight[curPos], weight);
            return -1;
        } 
        else if(hashComm[curPos] == -1){
            if(atomicCAS(&hashComm[curPos], -1, comm) == -1){
                atomicAdd(&hashWeight[curPos], weight);
                return curPos;
            } 
            else if(hashComm[curPos] == comm){
                atomicAdd(&hashWeight[curPos], weight);
                return -1;
            }
        }
    }
}

__device__ void mergeCommunity( int V, int primeNo, int *comSet, int *hashComm, float *hashWeight, int *prefixSum, new_graph NewGraph, device_graph Dgraph ){

    int	*edgeTocurPos = NewGraph.edgeTocurPos;
    float *newWeights = NewGraph.newWeights;
    int	*orderCom = NewGraph.orderCom;
    int	*newEdges = NewGraph.newEdges;
    int	*VtxStart = NewGraph.VtxStart;
    int *edgePos = NewGraph.edgePos;
    int *commDeg = NewGraph.commDeg;

    int CommOwned = 0;
    int CommPerBlock = blockDim.y;
    int No_of_Neigh = blockDim.x;
    int hashOffset = threadIdx.y * primeNo;
    int CommIdx = blockIdx.x * CommPerBlock + threadIdx.y;

    int comm = comSet[CommIdx];
    if( (CommIdx < V) && (Dgraph.CommSize[comm] > 0) ){

        // printf("Community and Community Size %d : %d\n",comm,Dgraph.CommSize[comm]);
        for( int i = threadIdx.x; i < primeNo; i += No_of_Neigh ){
            hashComm[hashOffset + i] = -1;
            hashWeight[hashOffset + i] = 0;
        }
        if (No_of_Neigh > 32){
            prefixSum[threadIdx.x] = 0;
            __syncthreads();
        }

        for(int VtxIdx = 0; VtxIdx < Dgraph.CommSize[comm]; VtxIdx++){
            int vtx = orderCom[VtxStart[comm] + VtxIdx];
            int StartIdx = Dgraph.edgesIdx[vtx];
            int vtxDeg = Dgraph.edgesIdx[vtx + 1] - StartIdx;
            
            for (int NeighIdx = threadIdx.x; NeighIdx < vtxDeg; NeighIdx += No_of_Neigh){
                int Idx = StartIdx + NeighIdx;
                int Neigh = Dgraph.edges[Idx];
                float weight = Dgraph.weights[Idx];
                int NeighComm = Dgraph.vtxComm[Neigh];
                int curPos = HashArraysII(NeighComm, primeNo, hashOffset, weight, hashComm, hashWeight);
                
                if (curPos > -1) {
                    // printf("%d : %d : %d : %d : %f\n",Idx,NeighComm, primeNo,hashOffset,weight);
                    // printf("%d : %d : %f\n",curPos, hashComm[curPos],hashWeight[curPos]);
                    edgeTocurPos[Idx] = curPos;
                    CommOwned++;
                }
            }
        }

        int CommPrefixSum = CommOwned;
        if(No_of_Neigh <= 32){
            for (int offset = 1; offset <= No_of_Neigh/2; offset *= 2) {
                int otherSum = __shfl_up_sync(FULL_MASK, CommPrefixSum, offset);
                if (threadIdx.x >= offset) {
                    CommPrefixSum += otherSum;
                }
            }
            CommPrefixSum -= CommOwned;
        } 
        else{
            for (int offset = 1; offset <= No_of_Neigh / 2; offset *= 2) {
                
                prefixSum[threadIdx.x] = CommPrefixSum;
                __syncthreads();
                if (threadIdx.x >= offset)
                    CommPrefixSum += prefixSum[threadIdx.x - offset];
            }
            CommPrefixSum -= CommOwned;
        }

        int newEdgesIdx = edgePos[comm] + CommPrefixSum;
        if (threadIdx.x == No_of_Neigh - 1) {
            commDeg[comm] = CommPrefixSum + CommOwned;
            atomicAdd(Dgraph.E, commDeg[comm]);
        }
        for (int vtxIdx = 0; vtxIdx < Dgraph.CommSize[comm]; vtxIdx++) {
            int vtx = orderCom[VtxStart[comm] + vtxIdx];
            int start = Dgraph.edgesIdx[vtx];
            int vtxDeg = Dgraph.edgesIdx[vtx + 1] - start;

            for (int NeighIdx = threadIdx.x; NeighIdx < vtxDeg; NeighIdx += No_of_Neigh){
                int index = start + NeighIdx;
                int curPos = edgeTocurPos[index];
                if (curPos > -1) {
                    newEdges[newEdgesIdx] = hashComm[curPos];
                    newWeights[newEdgesIdx] = hashWeight[curPos];
                    newEdgesIdx++;
                }
            }
        }

        // if((threadIdx.x == 0)) //&& (threadIdx.y == 0))
        // printf("executed merge Comm Inner Loop : %d\n", CommOwned);

    }
}

__global__ void Set_CommSize_CommDeg(int V, device_graph Dgraph, new_graph NewGraph){
	int vtx = blockIdx.x * THREADS + threadIdx.x;
    int *commDeg = NewGraph.commDeg;
    int	*newId = NewGraph.newId;
    int *CommSize = Dgraph.CommSize;
    int *vtxComm =  Dgraph.vtxComm;
    int *edgesIdx = Dgraph.edgesIdx;
	if(vtx < V){
		int comm = vtxComm[vtx];
		atomicAdd(&CommSize[comm], 1);
		int vtxDeg = edgesIdx[vtx + 1] - edgesIdx[vtx];
		atomicAdd(&commDeg[comm], vtxDeg);
		newId[comm] = 1;
        // if((threadIdx.x == 0) && (threadIdx.y == 0))
        // printf("executed fillarray \n");
	}
}

__global__ void Order_Vtx_in_OrgCom(int V, int *vtxComm, new_graph NewGraph){
	int vtx = blockIdx.x * THREADS + threadIdx.x;
    int *orderCom = NewGraph.orderCom;
    int *VtxStart = NewGraph.VtxStart;
	if(vtx < V){
		int comm = vtxComm[vtx];
		int idx = atomicAdd(&VtxStart[comm], 1);
		orderCom[idx] = vtx;
        // if((threadIdx.x == 0) && (threadIdx.y == 0))
        // printf("executed Prepare orderCom \n");
	}
}

__global__ void mergeCommunity_Shared(int V, int primeNo, int *comSet, new_graph NewGraph, device_graph Dgraph){

    int CommPerBlock = blockDim.y;
    int CommIdx = blockIdx.x * CommPerBlock + threadIdx.y;
    if(CommIdx < V){
        extern __shared__ int s[];
        int *hashComm = s;
        float *hashWeight = (float *) &hashComm[CommPerBlock * primeNo];
        int *prefixSum = (int *) &hashWeight[CommPerBlock * primeNo];
        mergeCommunity(V, primeNo, comSet, hashComm, hashWeight, prefixSum, NewGraph, Dgraph );
        
        // if((threadIdx.x == 0) && (threadIdx.y == 0))
        // printf("executed merge Comm \n");
    }
}

__global__ void mergeCommunity_Global(int V, int primeNo, int *comSet, int *hashComm, float *hashWeight, new_graph NewGraph, device_graph Dgraph){

    int CommPerBlock = blockDim.y;
    int CommIdx = blockIdx.x * CommPerBlock + threadIdx.y;
    if(CommIdx < V){
        extern __shared__ int s[];
        int *prefixSum = s;
        hashComm = &hashComm[blockIdx.x * primeNo];
        hashWeight = &hashWeight[blockIdx.x * primeNo];
        mergeCommunity(V, primeNo, comSet, hashComm, hashWeight, prefixSum, NewGraph, Dgraph);
        // if((threadIdx.x == 0) && (threadIdx.y == 0))
        // printf("executed merge Comm Global \n");
    }
}

__global__ void compressEdges(int V, device_graph Dgraph, new_graph NewGraph){
    int *commDeg = NewGraph.commDeg;
    int	*newEdges = NewGraph.newEdges;
    float *newWeights = NewGraph.newWeights;
    int	*newId = NewGraph.newId;
    int *edgePos = NewGraph.edgePos;
    int	*VtxStart = NewGraph.VtxStart;

    int CommPerBlock = blockDim.y;
    int No_of_Neigh = blockDim.x;
    int CommIdx = blockIdx.x * CommPerBlock + threadIdx.y;
    if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0){
        Dgraph.edgesIdx[*Dgraph.V] = *Dgraph.E;
    }
    if(CommIdx < V && Dgraph.CommSize[CommIdx] > 0){
        int start = edgePos[CommIdx];
        int CommNewId = newId[CommIdx];
        if(threadIdx.x == 0){
            Dgraph.vtxComm[CommNewId] = CommNewId;
            Dgraph.newVtxComm[CommNewId] = CommNewId;
            Dgraph.edgesIdx[CommNewId] = VtxStart[CommIdx];
        }
        for(int Idx = threadIdx.x; Idx < commDeg[CommIdx]; Idx += No_of_Neigh){
            int newIdx = Idx + start;
            int oldIdx = VtxStart[CommIdx] + Idx;
            Dgraph.edges[oldIdx] = newId[newEdges[newIdx]];
            Dgraph.weights[oldIdx] = newWeights[newIdx];
            atomicAdd(&Dgraph.commWeights[CommNewId], newWeights[newIdx]);
        }
    }
}

__global__ void Update_Old_To_NewComm(device_graph Dgraph, int *newId){
    int vtx = blockIdx.x * THREADS + threadIdx.x;
    if(vtx < *Dgraph.old_V){
        int comm = Dgraph.oldToNewComm[vtx];
        Dgraph.oldToNewComm[vtx] = newId[comm];
    }
}


void aggregateCommunityPhase(host_graph &Hgraph, device_graph &Dgraph, new_graph &NewGraph){
    int BLOCKS, primeNo, VtxInBckt, LastBucketNo;
    int *vtxEnd, *LastVtxEnd;
    dim3 DIMENSIONS;
    int V = Hgraph.V;
    int E = Hgraph.E;

    thrust::sequence(thrust::device, Dgraph.partition, Dgraph.partition + V, 0);

    thrust::fill(thrust::device, NewGraph.newId, NewGraph.newId + V, 0);
	thrust::fill(thrust::device, Dgraph.CommSize, Dgraph.CommSize + V, 0);
	thrust::fill(thrust::device, NewGraph.commDeg, NewGraph.commDeg + V, 0);
    //cout<<"Entered aggregate Community Successfully \n";
    BLOCKS = (V + THREADS - 1) / THREADS;
    
    Set_CommSize_CommDeg <<< BLOCKS, THREADS >>>(V, Dgraph, NewGraph);

    int newV = thrust::reduce(thrust::device, NewGraph.newId, NewGraph.newId + V);
    // cout << "New V : " << newV << endl;
    thrust::exclusive_scan(thrust::device, NewGraph.newId, NewGraph.newId + V , NewGraph.newId);
	thrust::exclusive_scan(thrust::device, NewGraph.commDeg, NewGraph.commDeg + V, NewGraph.edgePos);
    thrust::fill(thrust::device,  NewGraph.edgeTocurPos,  NewGraph.edgeTocurPos + E, -1);
    thrust::exclusive_scan(thrust::device, Dgraph.CommSize, Dgraph.CommSize + V, NewGraph.VtxStart);
    thrust::fill(thrust::device, Dgraph.E, Dgraph.E + 1, 0);

    Order_Vtx_in_OrgCom <<< BLOCKS, THREADS >>>(V, Dgraph.vtxComm, NewGraph);

    thrust::exclusive_scan(thrust::device, Dgraph.CommSize, Dgraph.CommSize + V, NewGraph.VtxStart);
    
    cout << "Phase II \n";
    for (int BucketNo = 0; BucketNo < numCBuckets - 2; BucketNo++){
        DIMENSIONS = dimsII[BucketNo];
        primeNo = findPrime(bucketsII[BucketNo + 1]*1.5);
        vtxEnd = thrust::partition(thrust::device, Dgraph.partition, Dgraph.partition + V, CheckBucketPhase2(bucketsII[BucketNo], bucketsII[BucketNo + 1], NewGraph.commDeg));
        VtxInBckt = thrust::distance(Dgraph.partition, vtxEnd);
        
        if(VtxInBckt > 0) {
            //cout << "Vtx In Bucket is : " << VtxInBckt << endl;
            int sharedMemSize = DIMENSIONS.y * primeNo * (sizeof(float) + sizeof(int)) + DIMENSIONS.x * sizeof(int);
            BLOCKS = (VtxInBckt + DIMENSIONS.y - 1) / DIMENSIONS.y;
            mergeCommunity_Shared <<< BLOCKS, DIMENSIONS, sharedMemSize >>>(VtxInBckt, primeNo, Dgraph.partition, NewGraph, Dgraph);
            CHECK( cudaDeviceSynchronize() );
        }
    }

    LastBucketNo = numCBuckets - 2;
    DIMENSIONS = dimsII[LastBucketNo];
    int commDegree = newV;
    primeNo = findPrime(commDegree * 1.5);
    LastVtxEnd = thrust::partition(thrust::device, Dgraph.partition, Dgraph.partition + V, CheckBucketPhase2(bucketsII[LastBucketNo], bucketsII[LastBucketNo + 1], NewGraph.commDeg));
    VtxInBckt = thrust::distance(Dgraph.partition, LastVtxEnd);
    // cout << "Vtx in Last Bucket : " << VtxInBckt << endl;
    if( VtxInBckt > 0 ){

        int *hashComm;
		float *hashWeight;
		cudaMalloc((void**)&hashComm, primeNo * VtxInBckt * sizeof(int));
		cudaMalloc((void**)&hashWeight, primeNo * VtxInBckt * sizeof(float));
		int sharedMemSize = THREADS * sizeof(int);
		BLOCKS = (VtxInBckt + DIMENSIONS.y - 1) / DIMENSIONS.y;
		mergeCommunity_Global <<< BLOCKS, DIMENSIONS, sharedMemSize >>> (VtxInBckt, primeNo, Dgraph.partition, hashComm, hashWeight, NewGraph, Dgraph);
		
        CHECK( cudaFree(hashComm) );
		CHECK( cudaFree(hashWeight) );

    }

    cudaMemcpy(&Hgraph.E, Dgraph.E, sizeof(int), cudaMemcpyDeviceToHost);
	Hgraph.V = newV;
	cudaMemcpy(Dgraph.V, &newV, sizeof(int), cudaMemcpyHostToDevice);
	thrust::fill(thrust::device, Dgraph.CommSize, Dgraph.CommSize + Hgraph.V, 1);
	BLOCKS = (V * 32 + THREADS - 1) / THREADS;
	DIMENSIONS = {32, THREADS / 32};

    thrust::fill(thrust::device, Dgraph.commWeights, Dgraph.commWeights + Hgraph.V, 0.0);
	
	thrust::exclusive_scan(thrust::device, NewGraph.commDeg, NewGraph.commDeg + V, NewGraph.VtxStart);

    compressEdges <<< BLOCKS, DIMENSIONS >>>(V, Dgraph, NewGraph);
    Update_Old_To_NewComm <<< (Hgraph.old_V + THREADS - 1) / THREADS, THREADS >>>(Dgraph, NewGraph.newId);
	

    cout << "Finished Phase II Aggresssion Phase \n \n";



}