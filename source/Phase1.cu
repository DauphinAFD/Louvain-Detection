#include "../include/PHASE1.cuh"
using namespace std;

struct square {
    __device__ float operator()(const float &x) const {
        return x * x;
    }
};

struct CheckBucketPhase1{
    int minDeg, maxDeg;
	int *edgeIdx;

	CheckBucketPhase1(int minimum, int maximum, int *edgesIndex) {
		minDeg = minimum;
		maxDeg = maximum;
		edgeIdx = edgesIndex;
	}

	__host__ __device__ bool operator()(const int &vtx) const {
		int Deg = edgeIdx[vtx + 1] - edgeIdx[vtx];
		return (Deg > minDeg && Deg <= maxDeg);
	}
};

__device__ int Hash(int val, int idx, int primeNo) {
	int h1 = val % primeNo;
	int h2 = 1 + (val % (primeNo - 1));
    //printf("passed Returning Hash");
	return (h1 + idx * h2) % primeNo;
}

__device__ float Calc_delQ(int vtx, int comm, int currComm, float M, float *commWeights, float *vtx_Kvalue, float weightSum_wrt_OtherComm) {
    float CommWght = commWeights[comm];
    float currCommWght = commWeights[currComm] - vtx_Kvalue[vtx];
    //printf("Inside DelQ %d : %d : %f : %f : %f \n",vtx,comm,CommWght,currCommWght,M);
    float gain = weightSum_wrt_OtherComm / M  + vtx_Kvalue[vtx] * (currCommWght - CommWght) / (2 * M * M) ;
    //printf("Finishedd calculating Gain delQ \n");
    return gain;
}

__device__ int HashArrays(int comm, int primeNo, int hashOffset, float weight, float *hashWeight, int *hashComm) {
    int it = 0, curPos;
    
     do{
    //         // printf("Inside DO While ");
            curPos = hashOffset + Hash(comm, it++, primeNo);
            if (hashComm[curPos] == comm)
                atomicAdd(&hashWeight[curPos], weight);
            else if (hashComm[curPos] == -1){
                if (atomicCAS(&hashComm[curPos], -1, comm) == -1) // made a change
                    atomicAdd(&hashWeight[curPos], weight);
                else if (hashComm[curPos] == comm)
                    atomicAdd(&hashWeight[curPos], weight);
            } 
    }while (hashComm[curPos] != comm);
    //printf("Passed HashArray\n");
    // curPos = 1; //Delete it later
    return curPos ;
}

__device__ void ComputeMove(int V, int primeNo,float M, int *vSet, int *hashComm, float *hashWeight, float *weightSum_wrt_SameComm, int *bestCommies, float *DEL_Q, device_graph Dgraph){

    int vtxPerBlock = blockDim.y;
    int vtxIdx = blockIdx.x * vtxPerBlock + threadIdx.y;

    if(vtxIdx < V){
        //printf("M value :%f ",M);
        float *commWeights = Dgraph.commWeights;
		float *vtx_Kvalue = Dgraph.vtx_Kvalue;
		int *newVtxComm = Dgraph.newVtxComm;
		float *weights = Dgraph.weights;
        int *CommSize = Dgraph.CommSize;
        int *edgesIdx = Dgraph.edgesIdx;
        int *vtxComm = Dgraph.vtxComm;
		int *edges = Dgraph.edges;

        int hashOffset = threadIdx.y * primeNo;
		int No_of_Neigh = blockDim.x;

        if (threadIdx.x == 0){
		    weightSum_wrt_SameComm[threadIdx.y] = 0;
            
        }
		for (int i = threadIdx.x; i < primeNo; i += No_of_Neigh){
			hashWeight[hashOffset + i] = 0;
			hashComm[hashOffset + i] = -1;
		}
        //printf("%d : %d\n",primeNo,hashOffset);
        //if (No_of_Neigh > 32)
        if (No_of_Neigh > 32)
		    __syncthreads();

        float del_q = 0.0;
        int curPos;

        int vtx = vSet[vtxIdx];
        int currComm = vtxComm[vtx];
        int end = edgesIdx[vtx+1];
        int start = edgesIdx[vtx];
        int bestComm = currComm;

        for(int Idx = start + threadIdx.x; Idx < end; Idx += No_of_Neigh){
           
            int Neigh = edges[Idx];
            int comm = vtxComm[Neigh];
            float weight = weights[Idx];
            //printf("%d : %d\n",primeNo,hashOffset);
            //printf("%d : %d : %d : %d : %d\n",primeNo,hashOffset,comm,Neigh,weight);
            
            if(Neigh != vtx){
                curPos = HashArrays(comm, primeNo, hashOffset, weight, hashWeight, hashComm );
                if (comm == currComm){
                    atomicAdd(&weightSum_wrt_SameComm[threadIdx.y], weight);
                }
            }
            // printf("%d : %d : %d\n",vtx,comm,currComm);
            if ( (comm < currComm || CommSize[comm] > 1 || CommSize[currComm] > 1 ) && comm != currComm ){
                
                //printf(" Doing gain %d : %d : %d : %f\n",vtx,comm,currComm,hashWeight[curPos]);
				float gain = Calc_delQ(vtx, comm, currComm, M, commWeights, vtx_Kvalue, hashWeight[curPos]);
                //printf(" Doing Hash %d : %d : %d : %f\n",vtx,comm,currComm,gain);
				if ( gain > del_q || (gain == del_q && comm < bestComm) ){
					del_q = gain;
					bestComm = comm;
				}
			}
        }

        if (No_of_Neigh <= 32){

			for(int offset = No_of_Neigh / 2; offset > 0; offset /= 2){
                int otherComm = __shfl_down_sync(FULL_MASK, bestComm, offset);
				float otherGain = __shfl_down_sync(FULL_MASK, del_q, offset);
				if (otherGain > del_q || (otherGain == del_q && otherComm < bestComm)){
					del_q = otherGain;
					bestComm = otherComm;
				}
			}
		} 
        else{
            DEL_Q[threadIdx.x] = del_q;
            bestCommies[threadIdx.x] = bestComm;
			for (int offset = No_of_Neigh / 2; offset > 0; offset /= 2){
			//	__syncthreads();
				if (threadIdx.x < offset) {
                    int otherComm = bestCommies[threadIdx.x + offset];
					float otherGain = DEL_Q[threadIdx.x + offset];
					if (otherGain > DEL_Q[threadIdx.x] ||
					   (otherGain == DEL_Q[threadIdx.x] && otherComm < bestCommies[threadIdx.x])) {
						DEL_Q[threadIdx.x] = otherGain;
						bestCommies[threadIdx.x] = otherComm;
					}
				}
			}
            del_q = DEL_Q[threadIdx.x];
            bestComm = bestCommies[threadIdx.x];
		}

        
        if( (threadIdx.x == 0) && del_q - weightSum_wrt_SameComm[threadIdx.y] / M > 0){
			newVtxComm[vtx] = bestComm;
        }
        else
			newVtxComm[vtx] = currComm;

        // if((threadIdx.x == 0) && (threadIdx.y == 0)){
        //     printf("Completed Compute Move Hash Arraay \n");
        // }

        
    }

}

__global__ void Calc_Kvalue(device_graph Dgraph) {
	int vtxPerBlock = blockDim.y;
	int No_of_Neigh = blockDim.x;
    int V = *Dgraph.V;
	float edgeSum = 0.0;
	int vtx = blockIdx.x * vtxPerBlock + threadIdx.y;
	if (vtx < V) {

		int start = Dgraph.edgesIdx[vtx];
        int end = Dgraph.edgesIdx[vtx + 1];

		for (int idx = start + threadIdx.x; idx < end; idx += No_of_Neigh)
			edgeSum += Dgraph.weights[idx];
        
		for (int offset = No_of_Neigh / 2; offset > 0; offset /= 2) {
			edgeSum += __shfl_down_sync(FULL_MASK, edgeSum, offset);
		}
        
		if (threadIdx.x == 0) {
			Dgraph.vtx_Kvalue[vtx] = edgeSum;
            //printf("%d : %f \n",vtx,edgeSum);
		}
	}
}

__global__ void Calc_EdgesSumInComm(device_graph Dgraph){
    int vtxPerBlock = blockDim.y;
	int V = *Dgraph.V;
    int No_of_Neigh = blockDim.x;
    float edgeSum = 0.0;
	int vtx = blockIdx.x * vtxPerBlock + threadIdx.y;
    int comm = Dgraph.vtxComm[vtx];
    if(vtx < V){
        int start = Dgraph.edgesIdx[vtx];
        int end = Dgraph.edgesIdx[vtx + 1];

		for (int idx = start + threadIdx.x; idx < end; idx += No_of_Neigh){
            int Neigh = Dgraph.edges[idx];
            if(Dgraph.vtxComm[Neigh] == comm){
                edgeSum += Dgraph.weights[idx]; // Madee a change in idx -> Neigh
            }
        }
        __syncthreads();
        for (int offset = No_of_Neigh / 2; offset > 0; offset /= 2) {
			edgeSum += __shfl_down_sync(FULL_MASK, edgeSum, offset);
		}
        if (threadIdx.x == 0) {
			Dgraph.weightSum_InComm[vtx] = edgeSum;
            
            //printf("%d : %d : %d : %d : %f \n",vtx,comm,start,end,edgeSum);
		}
    }
}

__global__ void computeMove_Shared(int VtxInBckt, int primeNo,float M, int *vSet, device_graph Dgraph){
    int vtxPerBlock = blockDim.y;
    int vtx = blockIdx.x * vtxPerBlock + threadIdx.y;
   
    if (vtx < VtxInBckt){
        extern __shared__ int s[];
        //printf("check\n");
        //printf("%d : %d\n",vtx,VtxInBckt);
        int *hashComm = s;
        float *hashWeight = (float *)&hashComm[vtxPerBlock*primeNo];
        float *weightSum_wrt_SameComm = (float *)&hashWeight[vtxPerBlock*primeNo];
        int *bestCommies = (int *)&weightSum_wrt_SameComm[vtxPerBlock];
        float *DEL_Q = (float *)&bestCommies[THREADS];
        //__syncthreads();
        ComputeMove(VtxInBckt, primeNo, M, vSet, hashComm, hashWeight, weightSum_wrt_SameComm, bestCommies, DEL_Q, Dgraph);
        // if((threadIdx.x == 0) && (threadIdx.y == 0)){
        //     printf("Completed Compute Move \n");
        // }
    }
}

__global__ void computeMove_Global(int VtxInBckt, int primeNo,int M, int *vSet, int *hashComm, float *hashWeight, device_graph Dgraph){
    int vtxPerBlock = blockDim.y;
    int vtx = blockIdx.x * vtxPerBlock + threadIdx.y;
    if (vtx < VtxInBckt){
        extern __shared__ int s[];
        
        float *weightSum_wrt_SameComm = (float *)s;
        int *bestCommies = (int *)&weightSum_wrt_SameComm[vtxPerBlock];
        float *DEL_Q = (float *)&bestCommies[THREADS];
        hashComm = hashComm + blockIdx.x * primeNo;
        hashWeight = hashWeight + blockIdx.x * primeNo;
        __syncthreads();
        ComputeMove(VtxInBckt, primeNo, M, vSet, hashComm, hashWeight, weightSum_wrt_SameComm, bestCommies, DEL_Q, Dgraph);
    }
}

__global__ void updateOldToNewComm(device_graph Dgraph) {
	int vtx = blockIdx.x * THREADS + threadIdx.x;
    int V = *Dgraph.old_V;
	if (vtx < V){
		int comm = Dgraph.oldToNewComm[vtx];
		Dgraph.oldToNewComm[vtx] = Dgraph.vtxComm[comm];
	}
}

__global__ void updateVtxComm(int VtxInBckt, int *vSet, device_graph Dgraph) {
	int idx = blockIdx.x * THREADS + threadIdx.x;
	if (idx < VtxInBckt) {
		int vtx = vSet[idx];
		int oldComm = Dgraph.vtxComm[vtx];
		int newComm = Dgraph.newVtxComm[vtx];
        //printf("%d : %d : %d\n",vtx,Dgraph.newVtxComm[vtx],Dgraph.vtxComm[vtx]);
		if (oldComm != newComm) {
			Dgraph.vtxComm[vtx] = newComm;
			atomicSub(&Dgraph.CommSize[oldComm], 1);
			atomicAdd(&Dgraph.CommSize[newComm], 1);
		}
        // if((threadIdx.x == 0) && (threadIdx.y == 0)){
        //     printf("Updated Vtx Comm Successfully \n");
        // }
        // printf("%d : %d : %d\n",vtx,Dgraph.newVtxComm[vtx],Dgraph.vtxComm[vtx]);
        
	}
}

__global__ void Calc_CommWeight(device_graph Dgraph) {
	int vtx = blockIdx.x * THREADS + threadIdx.x;
	if (vtx < *Dgraph.V) {
		int community = Dgraph.vtxComm[vtx];
		atomicAdd(&Dgraph.commWeights[community], Dgraph.vtx_Kvalue[vtx]);
        // if((threadIdx.x == 0) && (threadIdx.y == 0)){
        //     printf("Updated Comm Weight Successfully \n");
        // }
	}
}



float Calc_Q(int V, float M, device_graph Dgraph){

    int BLOCKS = (V*32 + THREADS -1) / THREADS; 
    dim3 DIMENSIONS {32, THREADS / 32};
    // cout << "Calc Q Blocks : " << BLOCKS << " : " << DIMENSIONS.x << " : " << DIMENSIONS.y << endl;
    // cout << "Weight sum in comm "<< endl; 
    Calc_EdgesSumInComm <<< BLOCKS, DIMENSIONS >>>(Dgraph);

    float SUMcommWeightSum = thrust::transform_reduce(thrust::device, Dgraph.commWeights, Dgraph.commWeights + V, square(), 0.0, thrust::plus<float>());
    float SUMweightSum_wrt_Comm = thrust::reduce(thrust::device, Dgraph.weightSum_InComm, Dgraph.weightSum_InComm + V);
    float result = ( SUMweightSum_wrt_Comm / (2 * M) ) - ( SUMcommWeightSum  / (4 * M * M) );
    return result;

}



bool ModularityOptimisationPhase(float thresh, host_graph &Hgraph, device_graph &Dgraph){
    int primeNo, VtxInBckt, VtxInLastBckt;
    int V = Hgraph.V;
    int BLOCKS;
    int *vtxEnd, *LastVtxEnd;

    float Modular_Gain = thresh;
    bool found_change = false;

    BLOCKS = (V*32 + THREADS -1) / THREADS; 
    dim3 DIMENSIONS {32, THREADS / 32};
    Calc_Kvalue<<< BLOCKS, DIMENSIONS >>>(Dgraph);
    CHECK( cudaDeviceSynchronize() );
    
    int *partition = (int *)malloc(V*sizeof(int));

    int *hashComm;
    float *hashWeight;
    int LastBucketNo = numDBucketsI - 2;

    thrust::sequence(thrust::device, Dgraph.partition, Dgraph.partition + V, 0);
    cudaMemcpy(partition, Dgraph.partition , V*(sizeof(int)), cudaMemcpyDeviceToHost);
    LastVtxEnd = thrust::partition(thrust::host, partition, partition + V, CheckBucketPhase1(bucketsI[LastBucketNo], bucketsI[LastBucketNo + 1], Hgraph.edgesIdx));
    VtxInLastBckt = thrust::distance(partition, LastVtxEnd);
    
    
    if(VtxInLastBckt > 0){

        DIMENSIONS = dimsI[LastBucketNo];
        BLOCKS = (VtxInLastBckt + DIMENSIONS.y - 1) / DIMENSIONS.y;
        primeNo = findPrime(LargestDegree(Hgraph) * 1.5);
        //cout << "Last Prime : "<< primeNo << " max Deg : " << LargestDegree(Hgraph) * 1.5 << endl;

        cudaMalloc((void**)&hashWeight, primeNo * BLOCKS * sizeof(float));
        cudaMalloc((void**)&hashComm, primeNo * BLOCKS * sizeof(int));
    }
    


    while(Modular_Gain >= thresh){
        float Modular_start = Calc_Q(V, Hgraph.M, Dgraph);
        //cout << "Modular_start : " << Modular_start << endl;
        for(int BucketNo = 0; BucketNo < LastBucketNo; BucketNo++){
            cudaMemcpy(partition, Dgraph.partition , V*(sizeof(int)), cudaMemcpyDeviceToHost);
            vtxEnd = thrust::partition(thrust::host, partition, partition + V, CheckBucketPhase1(bucketsI[BucketNo], bucketsI[BucketNo + 1], Hgraph.edgesIdx));
            VtxInBckt = thrust::distance(partition, vtxEnd);
            cudaMemcpy( Dgraph.partition, partition, V*(sizeof(int)), cudaMemcpyHostToDevice);


            if(VtxInBckt > 0){
                //cout << "Bucket No : " << BucketNo << " Vtx In Bucket : " << VtxInBckt << endl;
                DIMENSIONS = dimsI[BucketNo];
                BLOCKS = (VtxInBckt + DIMENSIONS.y -1) / DIMENSIONS.y;
                primeNo = findPrime(bucketsI[BucketNo + 1]*1.5);
                //cout << "Prime No : " << primeNo << endl;
                int sharedMemSize = DIMENSIONS.y * primeNo * ( sizeof(int) + sizeof(float) ) + DIMENSIONS.y*sizeof(float);
                //if (DIMENSIONS.x > 32)
                sharedMemSize += THREADS * (sizeof(int) + sizeof(float));
                computeMove_Shared <<< BLOCKS, DIMENSIONS, sharedMemSize >>> (VtxInBckt, primeNo, Hgraph.M, Dgraph.partition, Dgraph); 
                CHECK( cudaDeviceSynchronize() );
                //cout << "Came to end of Shared memory" << endl;
                BLOCKS = (V + THREADS - 1) / THREADS;
                updateVtxComm <<< BLOCKS, THREADS >>> (VtxInBckt, Dgraph.partition, Dgraph);
                
                thrust::fill(thrust::device, Dgraph.commWeights, Dgraph.commWeights + V, 0.0);
                Calc_CommWeight <<< BLOCKS, THREADS >>> (Dgraph);
 
            }  
                
        }
        thrust::sequence(thrust::device, Dgraph.partition, Dgraph.partition + V, 0);
        cudaMemcpy(partition, Dgraph.partition , V*(sizeof(int)), cudaMemcpyDeviceToHost);
        LastVtxEnd = thrust::partition(thrust::host, partition, partition + V, CheckBucketPhase1(bucketsI[LastBucketNo], bucketsI[LastBucketNo + 1], Hgraph.edgesIdx));
        VtxInLastBckt = thrust::distance(partition, LastVtxEnd);
        cudaMemcpy( Dgraph.partition, partition, V*(sizeof(int)), cudaMemcpyHostToDevice);
        
        

        if(VtxInLastBckt > 0){
            
            DIMENSIONS = dimsI[LastBucketNo];
            BLOCKS = (VtxInLastBckt + DIMENSIONS.y - 1) / DIMENSIONS.y;

            int sharedMemSize = THREADS * ( sizeof(int) + sizeof(float) ) + DIMENSIONS.y * sizeof(float);
            computeMove_Global<<<BLOCKS, DIMENSIONS, sharedMemSize >>>( VtxInBckt, primeNo, Hgraph.M, Dgraph.partition, hashComm, hashWeight, Dgraph );

            BLOCKS = (V + THREADS - 1) / THREADS;
            updateVtxComm <<< BLOCKS, THREADS >>> (VtxInLastBckt, Dgraph.partition, Dgraph);
            thrust::fill(thrust::device, Dgraph.commWeights, Dgraph.commWeights + V, 0.0);
            Calc_CommWeight <<< BLOCKS, THREADS >>> (Dgraph);
            
        }


        //cout << "Modular_start : " << Modular_start << endl;
        float Modular_end = Calc_Q(V, Hgraph.M, Dgraph);
        Modular_Gain = Modular_end - Modular_start;
		found_change = (Modular_Gain > 0) | found_change;
        cout << "Modular start : " << Modular_start << " Modular end : " << Modular_end << " Modular Gain : " << Modular_Gain << endl;

    }

    if(VtxInLastBckt){
        CHECK( cudaFree(hashComm) );
        CHECK( cudaFree(hashWeight) );
    }
    BLOCKS = (Hgraph.old_V + THREADS - 1) / THREADS;
    updateOldToNewComm <<< BLOCKS, THREADS >>> (Dgraph);
    cout << "\nFinished end of OPTIMATION PHASE " << endl;
    cout << "The value of found change : " << found_change << endl;
    return found_change;

}

void printOrgToComm(host_graph &Hgraph, device_graph &Dgraph){
    int V = Hgraph.V;
    vector<vector<int>> CommVec(V);
	cudaMemcpy(Hgraph.oldToNewComm, Dgraph.oldToNewComm, Hgraph.old_V*sizeof(int), cudaMemcpyDeviceToHost);

	for (int vector = 0; vector < Hgraph.old_V; vector++) {
		int comm = Hgraph.oldToNewComm[vector];
		CommVec[comm].push_back(vector);
	}
	printf("No of Vertices : %d\n", Hgraph.V);
	for (int comm = 0; comm < Hgraph.V; comm++) {
		printf(" community [%d] : ", comm + 1);
		for (int i = 0; i < CommVec[comm].size(); i++)
			printf(" %d", CommVec[comm][i] + 1);
		printf("\n");
	}

}