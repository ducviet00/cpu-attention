// Minimal pybind module exposing vLLM's csrc/cpu attention entry points, so
// tests/bench_vs_refs.py can time "vllm" against "ours" in one process.
//
// The upstream csrc/cpu/torch_bindings.cpp registers hundreds of ops and pulls
// in dnnl/moe/quant sources; we only want the attention kernel, so this file
// repeats the three declarations from csrc/cpu/ops.h and binds them directly.
//
// Build: see scripts/box_sync.sh (it links cpu_attn.cpp, cpu_isa.cpp,
// utils.cpp + this file against torch and the pinned vLLM checkout).
//
// Keep the wrappers' argument lists in sync with tests/bench_vs_refs.py.

#include <torch/extension.h>

#include <string>

// From csrc/cpu/ops.h -- declared here so we do not need the whole header set.
void cpu_attn_reshape_and_cache(const torch::Tensor& key,
                                const torch::Tensor& value,
                                torch::Tensor& key_cache,
                                torch::Tensor& value_cache,
                                const torch::Tensor& slot_mapping,
                                const std::string& isa, const double k_scale,
                                const double v_scale,
                                const std::string& kv_cache_dtype);

torch::Tensor get_scheduler_metadata(
    const int64_t num_req, const int64_t num_heads_q,
    const int64_t num_heads_kv, const int64_t head_dim,
    const torch::Tensor& seq_lens, at::ScalarType dtype,
    const torch::Tensor& query_start_loc, const bool casual,
    const int64_t window_size, const std::string& isa_hint,
    const bool enable_kv_split,
    const std::optional<torch::Tensor>& dynamic_causal,
    const std::string& kv_cache_dtype);

void cpu_attention_with_kv_cache(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, torch::Tensor& output,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const double scale, const bool causal,
    const std::optional<torch::Tensor>& alibi_slopes,
    const int64_t sliding_window_left, const torch::Tensor& block_table,
    const double softcap, const torch::Tensor& scheduler_metadata,
    const std::optional<torch::Tensor>& s_aux,
    const std::optional<torch::Tensor>& dynamic_causal, const double k_scale,
    const double v_scale, const std::string& kv_cache_dtype);

namespace {

void reshape_and_cache(const torch::Tensor& key, const torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       const torch::Tensor& slot_mapping,
                       const std::string& isa) {
  cpu_attn_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                             isa, 1.0, 1.0, "auto");
}

torch::Tensor scheduler_metadata(
    const int64_t num_req, const int64_t num_heads_q,
    const int64_t num_heads_kv, const int64_t head_dim,
    const torch::Tensor& seq_lens, at::ScalarType dtype,
    const torch::Tensor& query_start_loc, const bool causal,
    const int64_t window_size, const std::string& isa,
    const bool enable_kv_split,
    const std::optional<torch::Tensor>& dynamic_causal,
    const std::string& kv_cache_dtype = "auto") {
  return get_scheduler_metadata(num_req, num_heads_q, num_heads_kv, head_dim,
                                seq_lens, dtype, query_start_loc, causal,
                                window_size, isa, enable_kv_split,
                                dynamic_causal, kv_cache_dtype);
}

void attention(const torch::Tensor& query, const torch::Tensor& key_cache,
               const torch::Tensor& value_cache, torch::Tensor& output,
               const torch::Tensor& query_start_loc,
               const torch::Tensor& seq_lens, const double scale,
               const bool causal,
               const std::optional<torch::Tensor>& alibi_slopes,
               const int64_t sliding_window_left,
               const torch::Tensor& block_table, const double softcap,
               const torch::Tensor& scheduler_metadata,
               const std::optional<torch::Tensor>& s_aux,
               const std::optional<torch::Tensor>& dynamic_causal,
               const double k_scale, const double v_scale,
               const std::string& kv_cache_dtype) {
  cpu_attention_with_kv_cache(query, key_cache, value_cache, output,
                              query_start_loc, seq_lens, scale, causal,
                              alibi_slopes, sliding_window_left, block_table,
                              softcap, scheduler_metadata, s_aux,
                              dynamic_causal, k_scale, v_scale,
                              kv_cache_dtype);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_attn_reshape_and_cache", &reshape_and_cache);
  m.def("get_scheduler_metadata", &scheduler_metadata,
        py::arg("num_req"), py::arg("num_heads_q"), py::arg("num_heads_kv"),
        py::arg("head_dim"), py::arg("seq_lens"), py::arg("dtype"),
        py::arg("query_start_loc"), py::arg("causal"), py::arg("window_size"),
        py::arg("isa"), py::arg("enable_kv_split"), py::arg("dynamic_causal"),
        py::arg("kv_cache_dtype") = "auto");
  m.def("cpu_attention_with_kv_cache", &attention);
}
