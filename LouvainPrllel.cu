#include "./include/PHASE1.cuh"
#include "./include/PHASE2.cuh"
#include "./include/HELPER.cuh"

using namespace std;

int main(){

    string filename = "./text/louvain.txt";
   
    host_graph Hgraph = Read(filename);
    cout << "Completed Reading the file\n "<<endl ;
    // cout << Hgraph.V << " " << Hgraph.E << endl;

    
    device_graph Dgraph;
    new_graph NewGraph;
    int sym = Hgraph.M;
    CHECK(cudaMemcpyToSymbol(M, &sym, sizeof(float)));
    CHECK(cudaMemcpyFromSymbol(&sym, M, sizeof(float)));
    
    
    prepareDevice(Hgraph, Dgraph, NewGraph);
    cout << endl;
    //cout << "Executed Successfully" << endl;

    float thresh = 0.2;

    // bool val = ModularityOptimisationPhase(thresh, Hgraph, Dgraph);
    
    // aggregateCommunityPhase(Hgraph, Dgraph, NewGraph);
    while(true){
		if (!ModularityOptimisationPhase(thresh, Hgraph, Dgraph)){
			break;
        }
        CHECK( cudaDeviceSynchronize() );
        aggregateCommunityPhase(Hgraph, Dgraph, NewGraph);
	}

   

    printOrgToComm(Hgraph,Dgraph);

    deleteDeviceVars(Hgraph, Dgraph, NewGraph);

    
    cout << "Finished successfully";
    return 0;

}