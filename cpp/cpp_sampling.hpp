#ifndef CPP_SAMPLING_HPP
#define CPP_SAMPLING_HPP

#include <torch/extension.h>

#include "factor_graph.hpp"

torch::Tensor compute_conditional_scores(FactorNode* factor_node, 
    std::vector<int> label_ids);

torch::Tensor gibbs_sample_step(FactorGraph& factor_graph, 
    std::vector<int>& label_ids, float temp = 1.0);

std::vector<int> gibbs_sampling_inference(std::vector<int> label_space_sizes,
    std::map<std::string, torch::Tensor> factor_scores,
    std::map<std::string, std::vector<std::vector<int>>> factor_dependencies,
    std::vector<std::vector<int>> sample_pool,
    int num_samples, float init_temp, float anneal_rate, float min_temp,
    bool is_debug, std::vector<int> ground_truth, std::string dump_root
);

std::vector<std::vector<int>> gibbs_sampling_inference_multithreading(
    std::vector<std::vector<int>> label_space_sizes,
    std::vector<std::map<std::string, torch::Tensor>> factor_scores,
    std::vector<std::map<std::string, 
                         std::vector<std::vector<int>>>> factor_dependencies,
    std::vector<std::vector<std::vector<int>>> sample_pool,
    int num_samples, float init_temp, float anneal_rate, float min_temp,
    std::string dump_root
);

std::vector<torch::Tensor> sample_rank_loss(std::vector<int> label_space_sizes,
    std::map<std::string, std::vector<std::vector<int>>> factor_dependencies,
    std::vector<std::string> factor_types,
    std::vector<torch::Tensor> factor_score_values,
    std::vector<std::vector<int>> sample_pool, std::vector<int> ground_truth,
    int num_samples, bool gold_loss_enabled, bool pair_loss_enabled,
    bool init_with_oracle
);

#endif
