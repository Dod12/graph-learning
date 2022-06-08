#include <matplot/matplot.h>
#include "graphs/graph.h"
#include <iostream>
#include <armadillo>
#include <algorithm>
#include <execution>

void plot_graph(const std::vector<std::pair<size_t, size_t>>& edges, const std::vector<double>& weights, const std::vector<std::vector<int>>& clusters) {

    auto graph = matplot::graph(edges);
    graph -> show_labels(true);
    //graph -> layout_algorithm(matplot::network::layout::force);
    //graph -> edge_labels(weights);
    //graph -> line_widths(weights);

    std::vector<std::string> labels(graph->n_vertices());
    std::vector<std::string> color_list = {"red", "green", "blue", "yellow", "magenta", "cyan"};
    std::vector<std::string> colors(graph->n_vertices());
    size_t n_nodes = graph->n_vertices();

    #pragma omp parallel for default(none) shared(std::cerr, n_nodes, clusters, labels, colors, color_list)
    for (int n = 0; n < n_nodes; ++n) {
        bool found_cluster = false;
        for (int i = 0; i < clusters.size(); ++i) {
            if (std::find(clusters[i].begin(), clusters[i].end(), n) != clusters[i].end()) {
                labels[n] = "{" + std::to_string(i)  + "}";
                //colors[n] = color_list[i];
                found_cluster = true;
                break;
            }
        }
        if (!found_cluster) {
            std::cerr << "No cluster found for vertex " << n << "." << std::endl;
        }
    }

    graph->node_labels(labels);

    std::vector<double> x, y;

    for (int i = 0; i < n_nodes; ++i) {
        x.push_back((int) i % (int) std::sqrt(n_nodes));
        y.push_back((int) i / (int) std::sqrt(n_nodes));
    }

    for (int i = 0; i < x.size(); i++) {
        std::cout << "Node at " << x[i] << ", " << y[i] << std::endl;
    }

    graph -> x_data(x);
    graph -> y_data(y);
    matplot::show();
}

int main() {

    auto grid = Graphs::Grid(25, 2);
    
    arma::sp_dmat adj = (arma::sp_dmat) arma::dmat({{0,2,2,0,0,0,0,0,0},
                                                            {2,0,2,0,0,0,0,0,0},
                                                            {2,2,0,1,0,0,0,0,0},
                                                            {0,0,1,0,3,3,0,0,0},
                                                            {0,0,0,3,0,3,0,0,0},
                                                            {0,0,0,3,3,0,1,0,0},
                                                            {0,0,0,0,0,1,0,1,1},
                                                            {0,0,0,0,0,0,1,0,1},
                                                            {0,0,0,0,0,0,1,1,0}});
    
    auto graph = Graphs::Graph(adj);

    auto S = grid.successor_representation(0.80);

    std::vector<int> n_clusters = {10, 30, 40};
    int n_nodes = 100000;

    for (auto n_c : n_clusters) {
        // Setup random clusters
        std::vector<std::vector<int>> cluster_idxs;
        for (int i = 0; i < n_c; ++i) {
            cluster_idxs.push_back(arma::conv_to<std::vector<int>>::from(arma::randi<arma::uvec>(1, arma::distr_param(0, (int) grid.nodes() - 1))));
        };

        for (int i = 0; i < n_nodes; ++i) {
            // Randomly pick a random node
            int node = arma::randi<int>(arma::distr_param(0,(int) grid.nodes() - 1));
            int max_cluster = 0;
            for (int j = 0; j < cluster_idxs.size(); ++j) {

                auto this_cluster_successor = std::transform_reduce(cluster_idxs[j].begin(), cluster_idxs[j].end(), 0.,
                                                 [](const auto& x, const auto& y) {return x + y;},
                                                 [S, node](const auto& x) {return S(x, node);}) / (double) cluster_idxs[j].size();

                auto max_cluster_successor = std::transform_reduce(cluster_idxs[max_cluster].begin(), cluster_idxs[max_cluster].end(), 0.,
                                                                    [](const auto& x, const auto& y) {return x + y;},
                                                                    [S, node](const auto& x) {return S(x, node);}) / (double) cluster_idxs[max_cluster].size();

                if (this_cluster_successor > max_cluster_successor) max_cluster = j;
                else if (this_cluster_successor < max_cluster_successor) max_cluster = (arma::randi<int>() % 2) ? j : max_cluster;

/*                for (auto cluster_node : cluster_idxs[j]) {

                    double max_cluster_distance = 0;
                    for (auto index : cluster_idxs[max_cluster]) {
                        max_cluster_distance = (S(index, node) > max_cluster_distance) ? S(index, node) : max_cluster_distance;
                    }

                    auto cluster_node_distance = S(cluster_node, node);

                    if (cluster_node == node) {
                        // Node is already in the cluster. We should test if it would be placed in the same cluster again.
                        continue;
                    }
                    else if (cluster_node_distance > max_cluster_distance) {
                        // We found a cluster node that is closer than the current cluster, let's update max_cluster
                        max_cluster = j;
                        break;
                    } else if (cluster_node_distance == max_cluster_distance) {
                        // If the two are equal, the node is added to a cluster at random.
                        max_cluster = (arma::randi<int>() % 2) ? j : max_cluster;
                    }
                }*/
            }
            cluster_idxs[max_cluster].push_back(node);
        }
        plot_graph(grid.edges(), grid.weights(), cluster_idxs);
    }

    //plot_graph(graph.edges(), graph.weights());


    return 0;
}
