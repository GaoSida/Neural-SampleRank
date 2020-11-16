// Mirror the Python implementation in nsr.graph.factor_graph

#ifndef FACTOR_GRAPH_HPP
#define FACTOR_GRAPH_HPP

#include <torch/extension.h>

#include <vector>
#include <map>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>

class FactorNode;

class LabelNode {
public:
  int node_id;
  int label_space_size;
  int current_value;
  std::vector<FactorNode*> factor_nodes;

  LabelNode(int node_id, int label_space_size) {
    this->node_id = node_id;
    this->label_space_size = label_space_size;
    current_value = 0;
    factor_nodes = std::vector<FactorNode*>();
  }
};

/* It's unclear which torch or ATen API can be used for indexing into a Tensor
whose dimension is decided at runtime. (We can do this in Python, but difficult
to achieve this in C++). Therefore we access a naked float pointer here.
*/
class FactorNode {
public:
  std::string factor_type;
  int node_idx;  // Index within the same type
  std::vector<LabelNode*> label_nodes;
  std::vector<int> strides;  // For indexing into the score table
  std::vector<float> score_table;
  float current_score;

  FactorNode(std::vector<LabelNode*> label_nodes, std::string factor_type = "",
      int node_idx = -1) {
    this->factor_type = factor_type;
    this->node_idx = node_idx;

    this->label_nodes = label_nodes;
    std::vector<int> label_space_sizes;
    for (auto& node : label_nodes) {
      label_space_sizes.push_back(node->label_space_size);
    }
    strides.resize(label_space_sizes.size());
    int current_stride = 1;
    for (int i = label_space_sizes.size() - 1; i >= 0; i--) {
      strides[i] = current_stride;
      current_stride *= label_space_sizes[i];
    }
    score_table.resize(current_stride);  // current_stride now is the full size
  }

  void set_score_table(torch::Tensor flat_table) {
    // The Tensor object might not out-live the FactorNode, hence a copy
    float* table_array = flat_table.data_ptr<float>();
    std::copy(table_array, table_array + score_table.size(), 
      score_table.begin());
  }

  float update_score() {
    int index = 0;
    for (int i = 0; i < label_nodes.size(); i++) {
      index += label_nodes[i]->current_value * strides[i];
    }
    current_score = score_table[index];
    return current_score; 
  }

  float& get_score(const std::vector<int>& indices) {
    int index = 0;
    for (int i = 0; i < indices.size(); i++) {
      index += indices[i] * strides[i];
    }
    return score_table[index];
  }
};


class FactorGraph {
public:
  std::vector<LabelNode*> label_nodes;
  std::map<std::string, std::vector<FactorNode*>> factor_nodes;
  float current_score;

  // For random initialization with an oracle
  std::mt19937 rng;
  std::uniform_real_distribution<float> uniform_0_1;

  FactorGraph(std::vector<int>& label_space_sizes,
      std::map<std::string, std::vector<std::vector<int>>>& factor_dependencies
  ) {
    // Create the label nodes
    label_nodes.resize(label_space_sizes.size());
    for (int i = 0; i < label_space_sizes.size(); i++) {
      label_nodes[i] = new LabelNode(i, label_space_sizes[i]);
    }

    // Create the factor nodes
    for (auto& kv : factor_dependencies) {
      std::vector<FactorNode*> factors;
      for (int node_idx = 0; node_idx < kv.second.size(); node_idx++) {
        std::vector<int>& nodes = kv.second[node_idx];
        std::vector<LabelNode*> dependencies;
        for (int& i : nodes) {
          dependencies.push_back(label_nodes[i]);
        }
        FactorNode* new_factor_node = new FactorNode(dependencies, kv.first,
          node_idx);
        factors.push_back(new_factor_node);
        for (int& i : nodes) {
          label_nodes[i]->factor_nodes.push_back(new_factor_node);
        }
      }
      factor_nodes.insert({kv.first, factors});
    }

    // Prepare the random distribution
    std::random_device rd;
    rng = std::mt19937(rd());
    uniform_0_1 = std::uniform_real_distribution<float>(0.0, 1.0);
  }

  ~FactorGraph() {
    for (LabelNode* node : label_nodes) {
      delete node;
    }
    for (auto& kv : factor_nodes) {
      for (FactorNode* node : kv.second) {
        delete node;
      }
    }
  }

  float set_label_values(std::vector<int>& label_vals) {
    for (int i = 0; i < label_nodes.size(); i++) {
      label_nodes[i]->current_value = label_vals[i];
    }

    current_score = 0;
    for (auto& kv : factor_nodes) {
      for (FactorNode* node : kv.second) {
        current_score += node->update_score();
      }
    }

    return current_score;
  }

  std::vector<int> get_label_values() {
    std::vector<int> label_vals;
    for (LabelNode* node : label_nodes) {
      label_vals.push_back(node->current_value);
    }
    return label_vals;
  }

  void set_score_table(std::map<std::string, torch::Tensor>& factor_scores) {
    for (auto& kv : factor_nodes) {
      torch::Tensor& scores = factor_scores[kv.first];
      for (int i = 0; i < kv.second.size(); i++) {
        kv.second[i]->set_score_table(scores[i]);
      }
    }
  }

  float random_init_labels(std::vector<int>* ground_truth = nullptr,
      float p = -1.0) {
    std::vector<int> label_vals;
    for (LabelNode* node : label_nodes) {
      if (ground_truth == nullptr || uniform_0_1(rng) > p) {
        std::uniform_int_distribution<int> uniform_labels(0, 
          node->label_space_size - 1);
        label_vals.push_back(uniform_labels(rng));
      }
      else {
        label_vals.push_back((*ground_truth)[node->node_id]);
      }
    }
    return set_label_values(label_vals);
  }

  float compute_score_on_labels(std::vector<int>& label_vals) {
    float score_sum = 0.0;
    for (auto& kv : factor_nodes) {
      for (FactorNode* factor : kv.second) {
        std::vector<int> factor_label_vals;
        for (LabelNode* node : factor->label_nodes) {
          factor_label_vals.push_back(label_vals[node->node_id]);
        }
        score_sum += factor->get_score(factor_label_vals);
      }
    }
    return score_sum;
  }
};

#endif
