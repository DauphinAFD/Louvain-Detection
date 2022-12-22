#include "../include/HELPER.cuh"

using namespace std;

bool isPrime(int n) {
	for (int i = 2; i < sqrt(n) + 1; i++)
		if (n % i == 0)
			return false;
	return true;
}

int findPrime(int n) {
    n++;
	while(!isPrime(n)) {
		n++;
	} 
	return n;
}

int LargestDegree(host_graph& Hgraph) {
    int maxDeg = 0;

    for (int i = 0; i < Hgraph.V; i++){
        
        maxDeg = std::max(maxDeg, Hgraph.edgesIdx[i+1] - Hgraph.edgesIdx[i]);
        
    }
    
    return maxDeg;
}

host_graph Read(string filename){

    int V, E;
    int v, u, idx = 0;
    string line;
    float weight;
    stringstream ss;
    host_graph Hgraph;
    fstream louvain(filename);
    if(louvain.is_open()){
        
        getline(louvain,line);
        ss.clear();
        ss.str("");
        ss.str(line);
        ss >> V >> E;
        // cout << "Original :" << V << E << endl;
        
        Hgraph.old_V = V;
        Hgraph.V = V;
        Hgraph.E = E;

        size_t Cwghtsize = V*sizeof(float);
        size_t idxsize = (V+1)*sizeof(int);
        size_t graphsize = V*sizeof(int);
        Hgraph.commWeights = (float *)malloc(Cwghtsize);
        Hgraph.oldToNewComm = (int *)malloc(graphsize);
        Hgraph.vtxComm = (int *)malloc(graphsize);
        Hgraph.edgesIdx = (int *)malloc(idxsize);

        vector<vector<pair<int , float>>> neigh(V);
        thrust::fill(thrust::host, Hgraph.commWeights, Hgraph.commWeights + V, 0);
        while(getline(louvain,line)) {
            ss.clear();
            ss.str("");
            ss.str(line);
        
            ss >> v >> u >> weight;
            //cout << v << " " << u << " " << weight << endl;
            Hgraph.commWeights[v] += weight;
            neigh[v].push_back(make_pair(u, weight));
            Hgraph.M += weight;
            if(v != u){
                E++;
                Hgraph.commWeights[u] += weight;
                neigh[u].push_back(make_pair(v, weight));
                Hgraph.M += weight;
            }
        }

        Hgraph.M /= 2.0;
        // cout << Hgraph.M << endl;
        Hgraph.E = E;
        size_t wghtsize = E*sizeof(float);
        size_t edgesize = E*sizeof(int);
        Hgraph.weights = (float *)malloc(wghtsize);
        Hgraph.edges = (int *)malloc(edgesize);
       

        for(int i = 0; i < V; i++){
           
            Hgraph.edgesIdx[i] = idx;
            for(auto x : neigh[i]){
                Hgraph.edges[idx] = x.first;
                Hgraph.weights[idx] = x.second;
                
                idx++;
            }
        }
        Hgraph.edgesIdx[V] = Hgraph.E;

        louvain.close();
        
    }
    
    return Hgraph;


}

void prepareDevice(host_graph &Hgraph, device_graph & Dgraph, new_graph &NewGraph){
    int V = Hgraph.V;
    int E = Hgraph.E;

    //cout << "Entering Function"<<endl;

    size_t Cwghtsize = V*sizeof(float);
    size_t idxsize = (V+1)*sizeof(int);
    size_t wghtsize = E*sizeof(float);
    size_t graphsize = V*sizeof(int);
    size_t edgesize = E*sizeof(int);

    CHECK(cudaMalloc((void**)&Dgraph.commWeights, Cwghtsize));
    CHECK(cudaMalloc((void**)&Dgraph.newVtxComm, graphsize));
    CHECK(cudaMalloc((void**)&Dgraph.vtx_Kvalue, Cwghtsize));
    CHECK(cudaMalloc((void**)&Dgraph.partition, graphsize));
    CHECK(cudaMalloc((void**)&Dgraph.oldToNewComm, graphsize));
    CHECK(cudaMalloc((void**)&Dgraph.weightSum_InComm, graphsize));
    CHECK(cudaMalloc((void**)&Dgraph.CommSize, graphsize));
    CHECK(cudaMalloc((void**)&Dgraph.vtxComm, graphsize));
    CHECK(cudaMalloc((void**)&Dgraph.old_V,sizeof(int)));
    CHECK(cudaMalloc((void**)&Dgraph.weights,wghtsize));
    CHECK(cudaMalloc((void**)&Dgraph.edgesIdx,idxsize));
    CHECK(cudaMalloc((void**)&Dgraph.edges, edgesize));
    CHECK(cudaMalloc((void**)&Dgraph.V,sizeof(int)));
    CHECK(cudaMalloc((void**)&Dgraph.E,sizeof(int)));

	thrust::sequence(thrust::device, Dgraph.newVtxComm, Dgraph.newVtxComm + V, 0);
	thrust::sequence(thrust::device, Dgraph.oldToNewComm, Dgraph.oldToNewComm + V, 0);
	thrust::sequence(thrust::device, Dgraph.vtxComm, Dgraph.vtxComm + V, 0);
    thrust::fill(thrust::device, Dgraph.CommSize, Dgraph.CommSize + V, 1);

    CHECK(cudaMemcpy(Dgraph.commWeights, Hgraph.commWeights, Cwghtsize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(Dgraph.weights, Hgraph.weights, wghtsize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(Dgraph.edgesIdx, Hgraph.edgesIdx, idxsize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(Dgraph.old_V, &Hgraph.old_V, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(Dgraph.edges, Hgraph.edges, edgesize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(Dgraph.V, &Hgraph.V, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(Dgraph.E, &Hgraph.E, sizeof(int), cudaMemcpyHostToDevice));

	CHECK(cudaMalloc((void**)&NewGraph.edgeTocurPos, edgesize));
	CHECK(cudaMalloc((void**)&NewGraph.newWeights, wghtsize));
	CHECK(cudaMalloc((void**)&NewGraph.orderCom, graphsize));
	CHECK(cudaMalloc((void**)&NewGraph.VtxStart, graphsize));
	CHECK(cudaMalloc((void**)&NewGraph.newEdges, edgesize));
    CHECK(cudaMalloc((void**)&NewGraph.commDeg, graphsize));
	CHECK(cudaMalloc((void**)&NewGraph.edgePos, graphsize));
	CHECK(cudaMalloc((void**)&NewGraph.newId, graphsize));

    //cout << "finish Preparing"<<endl;

}

void deleteDeviceVars(host_graph &Hgraph, device_graph & Dgraph, new_graph &NewGraph){

    //cout << "Started deleting" << endl;

    CHECK(cudaFree(Dgraph.weightSum_InComm));
	CHECK(cudaFree(Dgraph.oldToNewComm));
	CHECK(cudaFree(Dgraph.commWeights));
	CHECK(cudaFree(Dgraph.vtx_Kvalue));
	CHECK(cudaFree(Dgraph.newVtxComm));
	CHECK(cudaFree(Dgraph.partition));
	CHECK(cudaFree(Dgraph.CommSize));
	CHECK(cudaFree(Dgraph.edgesIdx));
    CHECK(cudaFree(Dgraph.vtxComm));
	CHECK(cudaFree(Dgraph.weights));
	CHECK(cudaFree(Dgraph.edges));
	CHECK(cudaFree(Dgraph.old_V));
	CHECK(cudaFree(Dgraph.E));
	CHECK(cudaFree(Dgraph.V));
    
	CHECK(cudaFree(NewGraph.edgeTocurPos));
	CHECK(cudaFree(NewGraph.newWeights));
	CHECK(cudaFree(NewGraph.orderCom));
	CHECK(cudaFree(NewGraph.VtxStart));
	CHECK(cudaFree(NewGraph.newEdges));
	CHECK(cudaFree(NewGraph.edgePos));
	CHECK(cudaFree(NewGraph.commDeg));
	CHECK(cudaFree(NewGraph.newId));
    
    free(Hgraph.oldToNewComm);
    free(Hgraph.commWeights);
    free(Hgraph.edgesIdx);
    free(Hgraph.vtxComm);
    free(Hgraph.weights);
    free(Hgraph.edges);

    //cout << "finish deleting" << endl;
    
}
