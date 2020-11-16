#ifndef CPP_TESTS_HPP
#define CPP_TESTS_HPP

#include <torch/extension.h>
#include "factor_graph.hpp"

FactorGraph get_five_node_graph(bool set_scores = false);  // Fixture

std::vector<float> test_unary_node();

std::vector<float> test_binary_node();

std::vector<float> test_graph_adjacency_list();

std::vector<float> test_factor_graph_compute_scores();

std::vector<int> test_init_with_oracle(std::vector<int> ground_truth,
    float p, bool no_oracle);

std::vector<torch::Tensor> test_unary_node_conditional_scores();

std::vector<torch::Tensor> test_binary_node_conditional_scores();

torch::Tensor test_gibbs_sample_probabilities(std::vector<int> label_vals,
    std::vector<int> block, float temp);

std::vector<std::vector<int>> test_gibbs_sample_forced();

std::vector<float> test_neg_hamming_loss();

#endif
