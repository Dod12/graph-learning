//
// Created by Daniel on 2022-06-06.
//
#pragma once
#include <armadillo>
#include <utility>

#define ushort unsigned short

namespace Graphs {

    class Graph {
    public:
        explicit Graph(const std::vector<std::vector<ushort>>& adjacency_list); // Construct from adjacency list
        explicit Graph(const arma::sp_dmat& A); // Construct from adjacency matrix

        // void add_node(std::vector<short>& edge_list); // Add a node from edges
        std::vector<std::pair<size_t, size_t>> edges();
        std::vector<double> weights();
        std::vector<double> successor_weights(double gamma = 0.8);

        arma::sp_dmat adjacency() { return this->_adjacency; };
        size_t nodes() { return this->_nodes; };
        arma::sp_dmat laplacian();
        arma::sp_umat degrees();
        arma::sp_dmat transition_matrix();
        arma::dmat successor_representation(double gamma = 0.8);

    protected:
        arma::sp_dmat _adjacency;
        size_t _nodes;

        int step_to(int start_node, int end_node);
    };

    class Grid : public Graph {
    public:
        explicit Grid(const int& n_edges, const int& n_dims = 2);

    private:
        static arma::sp_dmat _create_grid(const int &n_edges, const int &n_dims);
    };

    /*
    class Ball : public Graph {
       explicit Ball(const int& radius, const int& n_dims = 2);
    }
    */

} // Graphs
