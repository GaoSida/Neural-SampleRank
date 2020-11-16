// Mirror nsr.graph.margin_metric

#include "factor_graph.hpp"

#include <vector>
#include <unordered_set>


class MarginMetric {
public:
  std::vector<int> ground_truth;

  MarginMetric(std::vector<int>& ground_truth) {
    this->ground_truth = ground_truth;
  }

  virtual float compute_metric(std::vector<int>& label_sample,
    std::unordered_set<int>& ground_truth_diff_set) = 0;

  virtual float incremental_compute_metric(std::vector<LabelNode*>& sample,
    float prev_metric, std::vector<int>& prev_sample, 
    std::vector<int>& diff_indicies, 
    std::unordered_set<int>& ground_truth_diff_set) = 0;
};


class NegHammingDistance : public MarginMetric {
public:
  NegHammingDistance(std::vector<int>& ground_truth) : 
    MarginMetric(ground_truth) {}

  float compute_metric(std::vector<int>& label_sample, 
      std::unordered_set<int>& ground_truth_diff_set) {
    float metric = 0;
    ground_truth_diff_set.clear();
    for (int i = 0; i < label_sample.size(); i++) {
      // Padded ground truth has label value 1 (PyTorch convention)
      if (ground_truth[i] != label_sample[i] && ground_truth[i] != 1) {
        metric -= 1.0;
        ground_truth_diff_set.insert(i);
      }
    }
    return metric;
  }

  /* In addition to the Python interface, maintain each sample's diff compared
  with ground truth.
  */
  float incremental_compute_metric(std::vector<LabelNode*>& sample,
      float prev_metric, std::vector<int>& prev_sample, 
      std::vector<int>& diff_indicies, 
      std::unordered_set<int>& ground_truth_diff_set) {
    float current_metric = prev_metric;
    for (int i : diff_indicies) {
      bool prev_correct = (prev_sample[i] == ground_truth[i]);
      bool current_correct = (sample[i]->current_value == ground_truth[i]);
      if (prev_correct != current_correct) {
        if (prev_correct) {
          current_metric -= 1;
          ground_truth_diff_set.insert(i);
        }
        else {
          current_metric += 1;
          ground_truth_diff_set.erase(i);
        }
      }
    }
    return current_metric;
  }
};
