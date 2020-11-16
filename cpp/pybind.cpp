#include "cpp_sampling.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gibbs_sampling_inference", &gibbs_sampling_inference, 
    "Inference with Gibbs sampling");
  m.def("gibbs_sampling_inference_multithreading",
    &gibbs_sampling_inference_multithreading, "Multithreading inference");
  m.def("sample_rank_loss", &sample_rank_loss, "SampleRank loss and gradients");
}