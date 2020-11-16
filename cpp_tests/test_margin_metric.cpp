#include "cpp_tests.hpp"
#include "margin_metric.hpp"

#include <algorithm>


void set_label_node_values(std::vector<LabelNode*>& label_nodes,
    std::vector<int>& label_vals) {
  for (int i = 0; i < label_nodes.size(); i++) {
    label_nodes[i]->current_value = label_vals[i];
  }
}


std::vector<float> test_neg_hamming_loss() {
  std::vector<int> ground_truth({2, 0, 4, 3, 2, 3, 0, 2, 1, 1});
  MarginMetric* metric = new NegHammingDistance(ground_truth);
  std::vector<LabelNode*> current_sample;
  for (int i = 0; i < 10; i++) {
    current_sample.push_back(new LabelNode(i, 5));
  }
  std::vector<int> label_vals(10, 0);
  std::vector<float> results;

  set_label_node_values(current_sample, label_vals);
  std::unordered_set<int> ground_truth_diff_set;
  results.push_back(metric->compute_metric(label_vals, ground_truth_diff_set));
  std::copy(ground_truth_diff_set.begin(), ground_truth_diff_set.end(),
    std::back_inserter(results));

  current_sample[0]->current_value = 2;
  std::vector<int> block({0});
  results.push_back(metric->incremental_compute_metric(
    current_sample, -6, label_vals, block, ground_truth_diff_set));
  std::copy(ground_truth_diff_set.begin(), ground_truth_diff_set.end(),
    std::back_inserter(results));
  label_vals[0] = 2;
    
  current_sample[3]->current_value = 3;
  block = std::vector<int>({3});  
  results.push_back(metric->incremental_compute_metric(
    current_sample, -5, label_vals, block, ground_truth_diff_set));
  std::copy(ground_truth_diff_set.begin(), ground_truth_diff_set.end(),
    std::back_inserter(results));
  label_vals[3] = 3;
    
  current_sample[5]->current_value = 3;
  current_sample[7]->current_value = 2;
  block = std::vector<int>({5, 7});  
  results.push_back(metric->incremental_compute_metric(
    current_sample, -4, label_vals, block, ground_truth_diff_set));
  std::copy(ground_truth_diff_set.begin(), ground_truth_diff_set.end(),
    std::back_inserter(results));
  label_vals[5] = 3;
  label_vals[7] = 2;
    
  current_sample[5]->current_value = 4;
  current_sample[7]->current_value = 2;
  block = std::vector<int>({5, 7});
  results.push_back(metric->incremental_compute_metric(
    current_sample, -2, label_vals, block, ground_truth_diff_set));
  std::copy(ground_truth_diff_set.begin(), ground_truth_diff_set.end(),
    std::back_inserter(results));
  label_vals[5] = 4;
  label_vals[7] = 2;
    
  current_sample[0]->current_value = 3;
  block = std::vector<int>({0});
  results.push_back(metric->incremental_compute_metric(
    current_sample, -3, label_vals, block, ground_truth_diff_set));
  std::copy(ground_truth_diff_set.begin(), ground_truth_diff_set.end(),
    std::back_inserter(results));
  label_vals[0] = 3;

  delete metric;
  for (int i = 0; i < 10; i++) {
    delete current_sample[i];
  }
  return results;
}
