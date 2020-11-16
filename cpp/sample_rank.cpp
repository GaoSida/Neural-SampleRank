#include "factor_graph.hpp"
#include "cpp_sampling.hpp"
#include "margin_metric.hpp"

#include <algorithm>
#include <iostream>


void update_sample_rank_gradients(FactorGraph& graph,
    std::vector<int>& other_sample, bool other_sample_better,
    std::vector<int> diff_label_indices,
    std::map<std::string, float*>& factor_gradients) {
  std::unordered_set<FactorNode*> diff_factors;
  for (int label_idx : diff_label_indices) {
    for (FactorNode* factor : graph.label_nodes[label_idx]->factor_nodes) {
      diff_factors.insert(factor);
    }
  }

  for (FactorNode* factor : diff_factors) {
    float* gradients = factor_gradients[factor->factor_type];
    int current_idx = 0;
    int other_idx = 0;
    for (int i = 0; i < factor->label_nodes.size(); i++) {
      current_idx += factor->strides[i] * 
                     factor->label_nodes[i]->current_value;
      other_idx += factor->strides[i] * 
                   other_sample[factor->label_nodes[i]->node_id];
    }
    int row_size = factor->score_table.size();
    if (other_sample_better) {
      gradients[factor->node_idx * row_size + other_idx] -= 1.0;
      gradients[factor->node_idx * row_size + current_idx] += 1.0;
    }
    else {
      gradients[factor->node_idx * row_size + other_idx] += 1.0;
      gradients[factor->node_idx * row_size + current_idx] -= 1.0;
    }
  }
}


/* Compute the SampleRank loss and its gradients w.r.t. the factor values
Returns: The first tensor is the loss value. 
         The subsequent Tensors are the gradients (same order as inputs)
*/
std::vector<torch::Tensor> sample_rank_loss(std::vector<int> label_space_sizes,
    std::map<std::string, std::vector<std::vector<int>>> factor_dependencies,
    std::vector<std::string> factor_types,
    std::vector<torch::Tensor> factor_score_values,
    std::vector<std::vector<int>> sample_pool, std::vector<int> ground_truth,
    int num_samples, bool gold_loss_enabled, bool pair_loss_enabled,
    bool init_with_oracle
) {
  FactorGraph graph = FactorGraph(label_space_sizes, factor_dependencies);
  std::map<std::string, torch::Tensor> factor_scores;
  std::map<std::string, torch::Tensor> factor_gradients;
  std::map<std::string, float*> factor_gradients_pointer;
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  for (int i = 0; i < factor_types.size(); i++) {
    factor_scores.insert({factor_types[i], factor_score_values[i]});
    torch::Tensor table = torch::zeros_like(factor_score_values[i], options);
    factor_gradients.insert({factor_types[i], table});
    factor_gradients_pointer.insert({factor_types[i], table.data_ptr<float>()});
  }
  graph.set_score_table(factor_scores);

  MarginMetric* metric = new NegHammingDistance(ground_truth);
  float gold_score = graph.compute_score_on_labels(ground_truth);
  std::unordered_set<int> ground_truth_diff_set;
  float gold_metric = metric->compute_metric(ground_truth,
    ground_truth_diff_set);

  float prev_score;
  if (init_with_oracle) {
    prev_score = graph.random_init_labels(&ground_truth, 
      graph.uniform_0_1(graph.rng));
  }
  else {
    prev_score = graph.random_init_labels();
  }
  std::vector<int> prev_sample = graph.get_label_values();
  float prev_metric = metric->compute_metric(prev_sample, 
    ground_truth_diff_set);

  float accumulated_loss = 0;
  int num_loss_terms = 0;
  float current_metric, loss_term;
  for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    std::random_shuffle(sample_pool.begin(), sample_pool.end());
    for (auto& block : sample_pool) {
      gibbs_sample_step(graph, block);
      current_metric = metric->incremental_compute_metric(graph.label_nodes,
        prev_metric, prev_sample, block, ground_truth_diff_set);
      
      if (gold_loss_enabled && current_metric < gold_metric) {
        num_loss_terms += 1;
        loss_term = (gold_metric - current_metric) - 
          (gold_score - graph.current_score);
        if (loss_term > 0) {
          accumulated_loss += loss_term;
          update_sample_rank_gradients(graph, ground_truth, true,
            std::vector<int>(ground_truth_diff_set.begin(), 
            ground_truth_diff_set.end()), factor_gradients_pointer);
        }
      }

      if (pair_loss_enabled) {
        if (current_metric > prev_metric) {
          num_loss_terms += 1;
          loss_term = (current_metric - prev_metric) - 
            (graph.current_score - prev_score);
          if (loss_term > 0) {
            accumulated_loss += loss_term;
            update_sample_rank_gradients(graph, prev_sample, false,
              block, factor_gradients_pointer);
          }
        }
        else if (current_metric < prev_metric) {
          num_loss_terms += 1;
          loss_term = (prev_metric - current_metric) - 
            (prev_score - graph.current_score);
          if (loss_term > 0) {
            accumulated_loss += loss_term;
            update_sample_rank_gradients(graph, prev_sample, true,
              block, factor_gradients_pointer);
          }
        }
      }

      prev_score = graph.current_score;
      prev_metric = current_metric;
      for (int i : block) {
        prev_sample[i] = graph.label_nodes[i]->current_value;
      }
    }
  }

  torch::Tensor loss_tensor = torch::zeros({1}, options);
  if (num_loss_terms > 0) {
    loss_tensor[0] = accumulated_loss / float(num_loss_terms);
  }  // else, there is no loss terms, so loss is 0
  std::vector<torch::Tensor> results;
  results.push_back(loss_tensor);
  for (int i = 0; i < factor_types.size(); i++) {
    results.push_back(factor_gradients[factor_types[i]]);
  }

  delete metric;
  return results;
}
