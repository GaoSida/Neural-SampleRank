// Mirror the implementation of nsr.graph.gibbs_sampling

#include "factor_graph.hpp"
#include "cpp_sampling.hpp"

#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <thread>
#include <future>


torch::Tensor compute_conditional_scores(FactorNode* factor_node, 
    std::vector<int> label_ids) {
  std::unordered_map<int, int> label_node_map;
  for (int i = 0; i < factor_node->label_nodes.size(); i++) {
    label_node_map.insert({factor_node->label_nodes[i]->node_id, i});
  }

  // About the intersection of the block and the factor
  int block_score_table_size = 1;
  std::vector<int> block_label_space_sizes;
  std::vector<int> block_label_strides;
  std::vector<int64_t> block_score_table_shape;
  for (int i = 0; i < label_ids.size(); i++) {
    int label_node_id = label_ids[i];
    if (label_node_map.find(label_node_id) != label_node_map.end()) {
      int node_idx_in_factor = label_node_map[label_node_id];
      LabelNode* node = factor_node->label_nodes[node_idx_in_factor];
      block_score_table_size *= node->label_space_size;
      block_label_space_sizes.push_back(node->label_space_size);
      block_label_strides.push_back(factor_node->strides[node_idx_in_factor]);
      block_score_table_shape.push_back(node->label_space_size);
    }
    else {
      block_score_table_shape.push_back(1);
    }
  }

  int base_index = 0;  // Accumulated from labels nodes out of the block
  std::unordered_set<int> label_id_set(label_ids.begin(), label_ids.end());
  for (int i = 0; i < factor_node->label_nodes.size(); i++) {
    LabelNode* node = factor_node->label_nodes[i];
    if (label_id_set.find(node->node_id) == label_id_set.end()) {
      base_index += node->current_value * factor_node->strides[i];
    }
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor block_score_table = torch::zeros(block_score_table_size,
    options);
  float* table = block_score_table.data_ptr<float>();
  for (int i = 0; i < block_score_table_size; i++) {
    int flat_index = base_index;
    int block_index = i;
    for (int j = block_label_space_sizes.size() - 1; j >= 0; j--) {
      flat_index += (block_index % block_label_space_sizes[j]) 
        * block_label_strides[j];
      block_index /= block_label_space_sizes[j];
    }
    table[i] = factor_node->score_table[flat_index];
  }
  torch::IntArrayRef target_shape(block_score_table_shape);
  return block_score_table.view(target_shape);
}


torch::Tensor gibbs_sample_step(FactorGraph& factor_graph, 
    std::vector<int>& label_ids, float temp) {
  std::unordered_set<FactorNode*> factor_nodes_set;
  std::vector<int64_t> table_shape;
  for (int idx : label_ids) {
    for (FactorNode* node : factor_graph.label_nodes[idx]->factor_nodes) {
      factor_nodes_set.insert(node);
    }
    table_shape.push_back(factor_graph.label_nodes[idx]->label_space_size);
  }
  
  std::vector<FactorNode*> factor_nodes_list(factor_nodes_set.begin(),
    factor_nodes_set.end());
  torch::Tensor conditional_scores = torch::zeros(
    torch::IntArrayRef(table_shape));
  for (FactorNode* node : factor_nodes_list) {
    conditional_scores += compute_conditional_scores(node, label_ids);
  }

  conditional_scores /= temp;
  torch::Tensor flat_score = conditional_scores.flatten();
  // Minus the max to avoid overflow when normalizing
  flat_score = torch::softmax(flat_score - torch::max(flat_score), 0);
  
  float u = factor_graph.uniform_0_1(factor_graph.rng);
  int num_label_combinations = flat_score.size(0);
  float* probabilities = flat_score.data_ptr<float>();
  int next_sample = -1;
  for (int i = 0; i < num_label_combinations; i++) {
    u -= probabilities[i];
    if (u <= 0) {
      next_sample = i;
      break;
    }
  }

  // Parse the next_sample index into the index of each label
  for (int i = label_ids.size() - 1; i >= 0; i--) {
    LabelNode* node = factor_graph.label_nodes[label_ids[i]];
    node->current_value = next_sample % node->label_space_size;
    next_sample = next_sample / node->label_space_size;
  }
  // Update the scores
  for (FactorNode* factor : factor_nodes_set) {
    factor_graph.current_score -= factor->current_score;
    factor->update_score();
    factor_graph.current_score += factor->current_score;
  }

  return conditional_scores;
}


float graph_accuracy(std::vector<int>& ground_truth, FactorGraph& graph) {
  int num_labels = 0;
  int num_corrects = 0;
  for (int i = 0; i < ground_truth.size(); i++) {
    if (ground_truth[i] != 1) {  // Not padding
      num_labels += 1;
      num_corrects += (ground_truth[i] == 
                       graph.label_nodes[i]->current_value);
    }
  }
  return 1.0 * num_corrects / num_labels;
}


std::string generate_random_string(int length=10) {
  const std::string VALID_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::string random_string;
  std::generate_n(std::back_inserter(random_string), length, [&]() {
      return VALID_CHARS[rand() % VALID_CHARS.length()];
  });
  return random_string;
}


/* See nsr.graph.gibbs_sampling.gibbs_sampling_inference for documentation.
Instead of directly passing in the FactorGraph object, pass in the arguments
to build a Cpp FactorGraph object.
*/
std::vector<int> gibbs_sampling_inference(std::vector<int> label_space_sizes,
    std::map<std::string, torch::Tensor> factor_scores,
    std::map<std::string, std::vector<std::vector<int>>> factor_dependencies,
    std::vector<std::vector<int>> sample_pool,
    int num_samples, float init_temp, float anneal_rate, float min_temp,
    bool is_debug, std::vector<int> ground_truth, std::string dump_root
) {
  FactorGraph graph = FactorGraph(label_space_sizes, factor_dependencies);
  graph.set_score_table(factor_scores);

  graph.random_init_labels();
  float max_score = graph.current_score;
  std::vector<int> best_label = graph.get_label_values();
  
  if (is_debug) {
    float ground_truth_score = graph.compute_score_on_labels(ground_truth);
    std::cout << "Ground truth score is: " << ground_truth_score << std::endl;
  }

  float temperature = init_temp;
  std::vector<std::vector<int>> sample_history;
  for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    std::random_shuffle(sample_pool.begin(), sample_pool.end());
    for (auto& block : sample_pool) {
      gibbs_sample_step(graph, block, temperature);
      if (graph.current_score > max_score) {
        max_score = graph.current_score;
        best_label = graph.get_label_values();
        if (is_debug) {
          std::cout << "New best score: " << max_score << ", accuracy: " <<
            graph_accuracy(ground_truth, graph) << std::endl;
        }
      }
    }
    temperature = std::max(min_temp, temperature * anneal_rate);
    if (is_debug) {
      std::cout << "Inference sample " << sample_idx + 1 << ": accuracy " <<
        graph_accuracy(ground_truth, graph) << 
        "; score " << graph.current_score << std::endl;
    }
    if (dump_root.length() > 0) {
      sample_history.push_back(graph.get_label_values());
    }
  }

  if (dump_root.length() > 0) {
    std::ofstream fout;
    fout.open(dump_root + "/samples-" + generate_random_string() + ".txt");
    fout << factor_dependencies.at("unary").size() << std::endl;
    for (auto& sample : sample_history) {
      for (int label : sample) {
        fout << label << ";";
      }
      fout << std::endl;
    }
    fout.close();
  }

  return best_label;
}


std::vector<std::vector<int>> gibbs_sampling_inference_multithreading(
    std::vector<std::vector<int>> label_space_sizes,
    std::vector<std::map<std::string, torch::Tensor>> factor_scores,
    std::vector<std::map<std::string, 
                         std::vector<std::vector<int>>>> factor_dependencies,
    std::vector<std::vector<std::vector<int>>> sample_pool,
    int num_samples, float init_temp, float anneal_rate, float min_temp,
    std::string dump_root
) {
  std::vector<std::future<std::vector<int>>> futures;
  for (int i = 0; i < label_space_sizes.size(); i++) {
    futures.push_back(std::async(gibbs_sampling_inference, label_space_sizes[i],
      factor_scores[i], factor_dependencies[i], sample_pool[i],
      num_samples, init_temp, anneal_rate, min_temp,
      false, std::vector<int>(), dump_root));
  }
  std::vector<std::vector<int>> predictions;
  for (auto& fut : futures) {
    predictions.push_back(fut.get());
  }
  return predictions;
}
