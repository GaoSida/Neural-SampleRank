#include "cpp_tests.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_unary_node", &test_unary_node, "Test unary factor node.");
  m.def("test_binary_node", &test_binary_node, "Test binary factor node.");
  m.def("test_graph_adjacency_list", &test_graph_adjacency_list,
    "Test building the factor graph");
  m.def("test_factor_graph_compute_scores", &test_factor_graph_compute_scores,
    "Test computing the score of the current factor graph");
  m.def("test_init_with_oracle", &test_init_with_oracle,
    "Used for testing the initialization with oracle.");
  m.def("test_unary_node_conditional_scores", 
    &test_unary_node_conditional_scores, "Unary node conditional score table");
  m.def("test_binary_node_conditional_scores",
    &test_binary_node_conditional_scores, "Binary node conditional score");
  m.def("test_gibbs_sample_probabilities", &test_gibbs_sample_probabilities,
    "Used for testing the probability computation during sampling");
  m.def("test_gibbs_sample_forced", &test_gibbs_sample_forced,
    "Test Gibbs sampling in a forced direction");
  m.def("test_neg_hamming_loss", &test_neg_hamming_loss,
    "Test neg hamming loss");
}
