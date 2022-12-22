#ifndef LOUVAIN_H
#define LOUVAIN_H

#include <algorithm>
#include<iostream>
#include<vector>
#include <set>

class Network{
    private :
        int Network_size;
        std::vector<std::vector<int>>Graph;
        std::vector<std::set<int>> community;
        bool check_similar_community(std::vector<std::set<int>>);
        int find_normlizefact();
        int find_community(int);
        int find_wu(int);
        void print_community();
        std::vector<int> find_wu_c(int);
        std::vector<int> find_inside_edge_sum();
        std::vector<int> find_total_edge_sum();
        std::vector<float> find_Community_delQ(int, int, int, std::vector<int>, std::vector<int>);
    public :
        Network();
        Network(int);
        void addEdge(int, int, int);
        void set_community();
        void print_all();
        void louvain();
};

#endif /*LOUVAIN_H*/