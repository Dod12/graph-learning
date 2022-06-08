//
// Created by Daniel on 2022-06-06.
//

#include "graph.h"

namespace Graphs {

    Graph::Graph(const std::vector<std::vector<ushort>> &adjacency_list) {
        _nodes = adjacency_list.size();
        _adjacency = arma::zeros<arma::sp_dmat>(_nodes, _nodes);
        for (int i = 0; i < _nodes; ++i) {
            for (auto node: adjacency_list[i]) {
                this->_adjacency(node, i) = 1;
            }
        }
    }

    Graph::Graph(const arma::sp_dmat& adjacency) {
        _adjacency = adjacency;
        _nodes = adjacency.n_cols;
    }

    arma::sp_umat Graph::degrees() {
        auto degrees = arma::zeros<arma::sp_umat>(this->_nodes, this->_nodes);
        for (int i = 0; i < _nodes; ++i) {
            degrees(i,i) = _adjacency.col(i).n_nonzero;
        }
        return degrees;
    }

    std::vector<std::pair<size_t, size_t>> Graph::edges() {

        std::vector<std::pair<size_t, size_t>> edges;

        for (auto it = _adjacency.begin(); it != _adjacency.end(); ++it) {
            if (it.col() >= it.row() || true) { // Since the graph is undirected, we only add edges once.
                edges.emplace_back(it.col(), it.row());
            }
        }
        return edges;
    }

    std::vector<double> Graph::weights() {

        std::vector<double> weights;

        for (auto edge : edges()) {
            weights.push_back(_adjacency(edge.second, edge.first));
        }
        return weights;
    }

    // Create transition matrix from graph. This normalizes the adjacency matrix so that the transition probability for
    // each node sums to 1.
    arma::sp_dmat Graph::transition_matrix() {
        arma::sp_dmat T = _adjacency;

        for (int i = 0; i < _nodes; ++i) {
            T.col(i) /= arma::sum(T.col(i));
        }
        return T;
    }

    arma::sp_dmat Graph::laplacian() {
        return degrees() - _adjacency;
    }

    arma::dmat Graph::successor_representation(double gamma) {
        arma::dmat T = (arma::dmat) transition_matrix();
        return arma::inv(arma::eye(arma::size(T)) - gamma * T);
    }

    std::vector<double> Graph::successor_weights(double gamma) {
        std::vector<double> out;
        auto S = successor_representation(gamma);

        for (auto edge : edges()) {
            out.push_back(S(edge.second, edge.first));
        }
        return out;
    }

    // Return the index of the first node on the shortest path between the start and end nodes.
    int Graph::step_to(int start_node, int end_node) {

        int n_iter = 0;
        auto T = transition_matrix();
        while (T(start_node, end_node) == 0 and n_iter < 1000) {
            T *= T;
        }
        return 0;
    }

    // Grid

    Grid::Grid(const int &n_edges, const int &n_dims) : Graph(_create_grid(n_edges, n_dims)) {};

    arma::sp_dmat Grid::_create_grid(const int &n_edges, const int &n_dims) {
        int matrix_size = (int) std::pow(n_edges, n_dims);
        auto A = arma::zeros<arma::sp_dmat>(matrix_size, matrix_size);
        for (int i = 0; i < n_edges; ++i) {
            for (int j = 0; j < n_edges; ++j) {
                auto k = i*n_edges + j;

                if (j > 0) {
                    A(k-1, k) = 1;
                    A(k, k-1) = 1;
                }
                if (i > 0) {
                    A(k-n_edges, k) = 1;
                    A(k, k-n_edges) = 1;
                }
            }
        }
        return A;
    }

} // Graphs