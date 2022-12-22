#include "../include/LOUVAIN.h"
using namespace std;

Network::Network(){
    this->Network_size = 0;
    Graph = vector<vector<int>>(Network_size,vector<int>(Network_size));
    community = vector<set<int>>(Network_size);
}

Network::Network(int Network_size){
    this->Network_size = Network_size;
    Graph = vector<vector<int>>(Network_size,vector<int>(Network_size));
    community = vector<set<int>>(Network_size);
}

void Network::addEdge(int from,int to, int weight){
    Graph[from][to] = weight;
}

void Network::set_community(){
    for(int i = 0; i < community.size(); i++){
        community[i].insert(i);
    }
}

int Network::find_normlizefact(){
    int m = 0;
    for(int i = 0; i < Network_size; i++){
        for (auto x : Graph[i])
            m += x;
    }
    return m;
}

vector<int> Network::find_inside_edge_sum(){
    int edge_sum;
    vector<int> inside_edge_sum;
    for(int i = 0; i < community.size(); i++){
        edge_sum = 0;
        for (auto x : community[i]){
            for(auto y : community[i]){
                edge_sum += Graph[x][y];
            }
        }
        inside_edge_sum.push_back(edge_sum);
    }
    return inside_edge_sum;
}

vector<int> Network::find_total_edge_sum(){
    int edge_sum;
    vector<int> total_edge_sum;
    for(int i = 0; i < community.size(); i++){
        edge_sum = 0;
        for (auto x : community[i]){
            for(auto y : Graph[x]){
                edge_sum += y;
            }
        }
        total_edge_sum.push_back(edge_sum);
    }
    return total_edge_sum;
}


int Network::find_wu(int u){
    int w_u = 0;
    for(auto weight : Graph[u]){
        w_u += weight;
    }
    return w_u;
}

vector<int> Network::find_wu_c(int u){
    vector<int> w_uc(community.size(),0);
    for(int i = 0; i < community.size(); i++ ){
        for(auto v : community[i]){
            w_uc[i] += Graph[u][v];
        }
    }
    return w_uc;
}

vector<float> Network::find_Community_delQ(int u, int m,int w_u, vector<int> w_uc, vector<int> total_edge_sum){
    vector<float> del_Qc(Network_size,0);
    for(int i = 0; i < community.size(); i++ ){ 
        del_Qc[i] = float(w_uc[i])/(m) - float(total_edge_sum[i]*w_u*2)/(m*m);
    }
    return del_Qc;
}

int Network::find_community(int u){
    for(int i = 0; i < community.size(); i++ ){
        for(auto x : community[i]){
            if(u == x){
                return i;
            }
        }
    }
    return -1;
}

bool Network::check_similar_community(vector<set<int>> copy_community){
    if(community.size() != copy_community.size()){
        return false;
    }
    for(int i = 0; i < community.size(); i++){
        
        if(copy_community[i] == community[i]){
            continue;
        }
        else{
            return false;
        }
        
    }
    return true;
}

void Network::print_community(){

    cout<<endl;
    for(int i = 0; i < community.size(); i++){
        cout << "community [" << i << "] : ";
        for (auto x : community[i])
            cout << x << " ";
        cout << endl;    
    } 
    cout << endl;

}

void Network::print_all(){

    cout<<endl;
    for(int i = 0; i < Network_size; i++){
        cout << "Network [" << i << "] : ";
        for (auto x : Graph[i])
            cout << x << " ";
        cout << endl;
    }
    cout << endl;

    for(int i = 0; i < community.size(); i++){
        cout << "community [" << i << "] : ";
        for (auto x : community[i])
            cout << x << " ";
        cout << endl;    
    } 
    cout << endl;


}


void Network::louvain(){
    vector<set<int>> copy_community;
    bool similar_community;
    int count = 1;
    set_community();
    
    do{
        cout << endl << "\t\t ---- For Iteration No : "<< count << " ----\n\n "; 
        
        print_all();
        float delQ;
        vector<float> del_Qc;
        int pref_community,org_community,community_change;
        vector<set<int>> dup_community;

        int m = find_normlizefact();
        
        copy_community = community;

        vector<int> total_edge_sum = find_total_edge_sum();     // for all community
        vector<int> inside_edge_sum = find_inside_edge_sum();   // for all community 
            //Phase 1
        do{
            community_change = 0;
            
            for(int u = 0; u < Network_size; u++){

                int w_u = find_wu(u);
                vector<int> w_uc = find_wu_c(u);
                org_community = find_community(u);

                del_Qc = find_Community_delQ(u,m,w_u,w_uc,total_edge_sum);
                pref_community = std::max_element(del_Qc.begin(),del_Qc.end()) - del_Qc.begin();
                delQ = *max_element(del_Qc.begin(), del_Qc.end());
            
                if(delQ > 0){
                    total_edge_sum[pref_community] += w_u;
                    total_edge_sum[org_community] -= w_u;
                    inside_edge_sum[pref_community] += w_uc[pref_community];
                    inside_edge_sum[org_community] -= w_uc[org_community];
                    
                    //cout << "Original Community : " << org_community << " Prefered Community : " << pref_community << endl;
                        // update community information
                    community[org_community].erase(u);
                    community[pref_community].insert(u);

                        //check for a community change
                    if(pref_community != org_community && !community_change){
                        community_change = 1;
                    }
                }
            }
        }while(community_change);
            
            // Calculate Community set and Modularity
        int Q = 0;
        for(int i = 0; i < total_edge_sum.size(); i++ ){
            Q += total_edge_sum[i] + inside_edge_sum[i];
        }
            //erase if community has no element
        for(int  i = 0; i < community.size(); i++ ){
            if(community[i].size()){
                dup_community.push_back(community[i]);
            }
        }
        community = dup_community;

            //print_community
        cout << " \tAfter Phase 1 :"<<endl;
        print_community();
        
            
            // Phase 2 Implementation
        Network_size = community.size();
        vector<vector<int>>edge_info(Network_size,vector<int>(Network_size));
            //edge list creation
        for(int i = 0; i < community.size(); i++){
            for(int j = 0; j < community.size(); j++){
                if(i != j){
                    for(auto x : community[i]){
                        for(auto y : community[j]){
                            edge_info[i][j] += Graph[x][y]; 
                        }
                    }
                }
                else{
                    edge_info[i][j] = 0;
                }
            }
        }

        Graph = edge_info;
        community = vector<set<int>>(Network_size);
        set_community();
        cout << " \tAfter Phase 2 :"<<endl;
        print_community();
        cout << "The Q value is : " << Q << endl;
        similar_community = check_similar_community(copy_community);
        count++;
    }while(!similar_community);
    

} 
 

