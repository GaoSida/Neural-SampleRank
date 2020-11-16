#include "factor_graph.hpp"
#include "cpp_tests.hpp"
#include "cpp_sampling.hpp"

#include <algorithm>


std::vector<float> test_unary_node() {
  std::vector<float> test_results;

  LabelNode* label = new LabelNode(3, 9);
  FactorNode* factor = new FactorNode({label});
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor table = torch::arange(9, options);
  factor->set_score_table(table);
  
  factor->label_nodes[0]->current_value = 0;
  test_results.push_back(factor->update_score());  // Expect: 0.0

  factor->label_nodes[0]->current_value = 1;
  test_results.push_back(factor->update_score());  // Expect: 1.0

  factor->label_nodes[0]->current_value = 3;
  test_results.push_back(factor->update_score());  // Expect: 3.0

  std::vector<int> indices({7});
  test_results.push_back(factor->get_score(indices));  // Expect: 7.0
  factor->get_score(indices) = 101.7;
  test_results.push_back(factor->get_score(indices));  // Expect: 101.7

  delete label;
  delete factor;
  return test_results;
}


std::vector<float> test_binary_node() {
  std::vector<float> test_results;

  LabelNode* label_0 = new LabelNode(3, 9);
  LabelNode* label_1 = new LabelNode(24, 7);
  FactorNode* factor = new FactorNode({label_0, label_1});
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor table = torch::arange(63, options);
  factor->set_score_table(table);

  factor->label_nodes[0]->current_value = 5;
  factor->label_nodes[1]->current_value = 2;
  test_results.push_back(factor->update_score());  // Expect 37.0
  
  std::vector<int> indices({5, 2});
  test_results.push_back(factor->get_score(indices));  // Expect 37.0
  indices = {3, 4};
  test_results.push_back(factor->get_score(indices));  // Expect 25.0
  indices = {8, 6};
  test_results.push_back(factor->get_score(indices));  // Expect 62.0

  delete label_0;
  delete label_1;
  delete factor;
  return test_results;
}


// Test fixture for the test cases
FactorGraph get_five_node_graph(bool set_scores) {
  std::vector<int> label_space_sizes({3, 3, 3, 3, 3});
  std::map<std::string, std::vector<std::vector<int>>> dependencies = {
    {"unary", {{0}, {1}, {2}, {3}, {4}}},
    {"binary", {{1, 4}, {2, 3}, {3, 4}}},
    {"tertiary", {{0, 1, 2}}}
  };
  FactorGraph graph = FactorGraph(label_space_sizes, dependencies);
  
  if (set_scores) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor unary_table = torch::arange(3, options);
    torch::Tensor binary_table = torch::arange(9, options);
    torch::Tensor tertiary_table = torch::arange(27, options);
    for (FactorNode* node : graph.factor_nodes["unary"]) {
      node->set_score_table(unary_table);
    }
    for (FactorNode* node : graph.factor_nodes["binary"]) {
      node->set_score_table(binary_table);
    }
    for (FactorNode* node : graph.factor_nodes["tertiary"]) {
      node->set_score_table(tertiary_table);
    }
  }

  return graph;
}


std::vector<float> test_graph_adjacency_list() {
  std::vector<float> results;
  FactorGraph graph = get_five_node_graph();

  results.push_back(graph.label_nodes.size());  // Expect: 5.0
  for (int i = 0; i < 5; i++) {
    results.push_back(graph.label_nodes[i]->node_id);
    results.push_back(graph.label_nodes[i]->label_space_size);
    results.push_back(graph.label_nodes[i]->factor_nodes.size());
    // Expect: [2, 3, 3, 3, 3]
  }

  results.push_back(graph.factor_nodes.size());  // Expect: 3
  auto& unary_nodes = graph.factor_nodes["unary"];
  results.push_back(unary_nodes.size());  // Expect 5
  for (int i = 0; i < 5; i++) {
    FactorNode* node = unary_nodes[i];
    results.push_back(node->label_nodes.size());
    results.push_back(node->label_nodes[0]->node_id);
    auto& l_factors = graph.label_nodes[i]->factor_nodes;
    results.push_back(std::find(l_factors.begin(), l_factors.end(), node) 
      != l_factors.end());
  }

  auto& binary_nodes = graph.factor_nodes["binary"];
  results.push_back(binary_nodes.size());  // Expect 3
  for (FactorNode* node : binary_nodes) {
    for (LabelNode* l_node : node->label_nodes) {
      results.push_back(l_node->node_id);
      auto& l_factors = l_node->factor_nodes;
      results.push_back(std::find(l_factors.begin(), l_factors.end(), node) 
        != l_factors.end());
    }
  }

  auto& tertiary_nodes = graph.factor_nodes["tertiary"];  // Expect 1
  results.push_back(tertiary_nodes.size());
  for (LabelNode* l_node : tertiary_nodes[0]->label_nodes) {
    results.push_back(l_node->node_id);
    auto& l_factors = l_node->factor_nodes;
    results.push_back(std::find(l_factors.begin(), l_factors.end(), 
      tertiary_nodes[0]) != l_factors.end());
  }

  return results;
}


std::vector<float> test_factor_graph_compute_scores() {
  std::vector<float> results;
  FactorGraph graph = get_five_node_graph(true);
  
  std::vector<int> label_vals({1, 0, 2, 1, 2});
  results.push_back(graph.set_label_values(label_vals));
  results.push_back(graph.current_score);

  label_vals = {0, 0, 0, 0, 0};
  results.push_back(graph.compute_score_on_labels(label_vals));
  label_vals = {1, 0, 2, 1, 2};
  results.push_back(graph.compute_score_on_labels(label_vals));
  label_vals = {2, 1, 0, 0, 1};
  results.push_back(graph.compute_score_on_labels(label_vals));
  results.push_back(graph.current_score);

  return results;
}


// For this test case we wrap a unit of functionalities and assert in Python
std::vector<int> test_init_with_oracle(std::vector<int> ground_truth,
    float p, bool no_oracle) {
  // Create a dummy graph with Label nodes only
  std::vector<int> label_space_sizes;
  label_space_sizes.resize(100, 10);
  std::map<std::string, std::vector<std::vector<int>>> factor_dependencies;
  FactorGraph graph = FactorGraph(label_space_sizes, factor_dependencies);
  
  if (no_oracle) {
    graph.random_init_labels();
  }
  else {
    graph.random_init_labels(&ground_truth, p);
  }

  return graph.get_label_values();
}
