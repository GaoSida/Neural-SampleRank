#include "cpp_tests.hpp"
#include "cpp_sampling.hpp"
#include "factor_graph.hpp"


std::vector<torch::Tensor> test_unary_node_conditional_scores() {
  std::vector<torch::Tensor> test_results;

  LabelNode* label = new LabelNode(3, 9);
  FactorNode* factor = new FactorNode({label});
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor table = torch::arange(9, options);
  factor->set_score_table(table);

  test_results.push_back(compute_conditional_scores(factor, {3}));
  test_results.push_back(compute_conditional_scores(factor, {1, 3, 5}));

  label->current_value = 3;
  factor->update_score();
  test_results.push_back(compute_conditional_scores(factor, {5, 7}));

  delete label;
  delete factor;
  return test_results;
}


std::vector<torch::Tensor> test_binary_node_conditional_scores() {
  std::vector<torch::Tensor> test_results;

  LabelNode* label_0 = new LabelNode(3, 9);
  LabelNode* label_1 = new LabelNode(24, 7);
  FactorNode* factor = new FactorNode({label_0, label_1});
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor table = torch::arange(63, options);
  factor->set_score_table(table);
  label_0->current_value = 5;
  label_1->current_value = 2;
  factor->update_score();

  test_results.push_back(compute_conditional_scores(factor, {3}));
  test_results.push_back(compute_conditional_scores(factor, {24}));
  test_results.push_back(compute_conditional_scores(factor, {3, 24}));
  test_results.push_back(compute_conditional_scores(factor, {24, 3}));
  test_results.push_back(compute_conditional_scores(factor, {5, 3, 19, 24}));
  test_results.push_back(compute_conditional_scores(factor, {5, 7, 19}));

  delete label_0;
  delete label_1;
  delete factor;
  return test_results;
}


torch::Tensor test_gibbs_sample_probabilities(std::vector<int> label_vals,
    std::vector<int> block, float temp) {
  FactorGraph graph = get_five_node_graph(true);
  graph.set_label_values(label_vals);
  return gibbs_sample_step(graph, block, temp);
}


std::vector<std::vector<int>> test_gibbs_sample_forced() {
  std::vector<std::vector<int>> results;
  FactorGraph graph = get_five_node_graph(true);
  std::vector<int> label_vals({1, 0, 2, 1, 2});
  graph.set_label_values(label_vals);

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor table_3 = torch::zeros({3, 3, 3}, options);
  table_3[2][1][0] = 200.0;
  graph.factor_nodes["tertiary"][0]->set_score_table(table_3.flatten());
  std::vector<int> block({0, 2, 1});
  gibbs_sample_step(graph, block);
  results.push_back(graph.get_label_values());

  torch::Tensor table_2 = torch::zeros({3, 3}, options);
  table_2[2][1] = 200.0;
  graph.factor_nodes["binary"][2]->set_score_table(table_2.flatten());
  block = std::vector<int>({3, 4});
  gibbs_sample_step(graph, block);
  results.push_back(graph.get_label_values());

  table_2[2][2] = 500.0;
  graph.factor_nodes["binary"][2]->set_score_table(table_2.flatten());
  block = std::vector<int>({4});
  gibbs_sample_step(graph, block);
  results.push_back(graph.get_label_values());

  torch::Tensor table_1 = torch::zeros({3}, options);
  table_1[1] = 300.0;
  graph.factor_nodes["unary"][2]->set_score_table(table_1);
  block = std::vector<int>({2});
  gibbs_sample_step(graph, block);
  results.push_back(graph.get_label_values());

  block = std::vector<int>({3, 2});
  gibbs_sample_step(graph, block);
  results.push_back(graph.get_label_values());

  return results;
}
