#include "./include/LOUVAIN.h"
#include<fstream>
#include<sstream>
#include<string>
using namespace std;
int main(){
    string line;
    stringstream ss;
    int No_of_Vertex,from_Vertex,to_Vertex;
    fstream file("./text/louvaincpu.txt");
    
    file>>No_of_Vertex;
    Network N(No_of_Vertex);
    if(file.is_open()){
        while(getline(file,line)){
            file >> from_Vertex >> to_Vertex;
            N.addEdge(from_Vertex, to_Vertex, 1);
        }
        file.close();
    }
    cout << endl << "\t\t ---- Louvain Community Detection Algorithm Serial ----\n\n " << endl; 
    
    N.louvain();
   
    return 0;
}