#include "ctranslate2/ops/flash_attention.h"

#ifdef CT2_WITH_FLASH_ATTN
#include "fmha_fwd.hpp"
#include "mask.hpp"
#endif
#include "ctranslate2/ops/transpose.h"
#include "cuda/utils.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {
#ifdef CT2_WITH_FLASH_ATTN
    inline int num_splits_heuristic_ck(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
      // If we have enough to almost fill the SMs, then just use 1 split
      if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
      max_splits = std::min({max_splits, num_SMs, num_n_blocks});
      float max_efficiency = 0.f;
      std::vector<float> efficiency;
      efficiency.reserve(max_splits);
      auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
      // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
      // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
      // (i.e. it's 11 splits anyway).
      // So we check if the number of blocks per split is the same as the previous num_splits.
      auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
      };
      for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
          efficiency.push_back(0.f);
        } else {
          float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
          float eff = n_waves / ceil(n_waves);
          // printf("num_splits = %d, eff = %f\n", num_splits, eff);
          if (eff > max_efficiency) { max_efficiency = eff; }
          efficiency.push_back(eff);
        }
      }
      for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
          // printf("num_splits chosen = %d\n", num_splits);
          return num_splits;
        }
      }
      return 1;
    }

    int override_num_splits_if_necessary(int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits) {
      int device;
      int device_id = ctranslate2::get_device_index(ctranslate2::Device::CUDA);
      auto dprops = ctranslate2::cuda::get_device_properties(device_id);

      // TODO - tile size should match the TileFmhaShape, hardcode for now
      const int kM0 = 128;
      const int kN1 = hdim_v;

      const int num_m_blocks = (max_seqlen_q + kM0 - 1) / kM0;
      const int num_n_blocks = (hdim_v + kN1 - 1) / kN1;

      if(num_splits < 1 && p_drop == 0.0f)
        return num_splits_heuristic_ck(
          batch * nhead * num_m_blocks, dprops.multiProcessorCount * 2, num_n_blocks, 128);

      return num_splits;
    }

    fmha_fwd_traits get_ck_fmha_fwd_traits(const mask_info &mask, std::string dtype, int head_size) {
      return fmha_fwd_traits{head_size,
                             head_size,
                             dtype,
                             false, // is_group_mode
                             true,  // is_v_rowmajor
                             false, // has_logits_soft_cap
                             mask.type, // mask_type
                             bias_enum::no_bias, // bias_type
                             false, // has_lse
                             false, // has_dropout
                             quant_scale_enum::no_scale}; // qscale_type
    }

    fmha_fwd_args get_ck_fmha_fwd_args(const mask_info &mask,
                                       // sizes
                                       const int b,
                                       const int seqlen_q,
                                       const int seqlen_k,
                                       const int h,
                                       const int h_k,
                                       const int d,
                                       // device pointers
                                       const StorageView& q,
                                       const StorageView& k,
                                       const StorageView& v,
                                       StorageView& out,
                                       float softmax_scale) {
        // q: (batch_size, seqlen_q, nheads, d)
        // k: (batch_size, seqlen_k, nheads_k, d)
        // v: (batch_size, seqlen_k, nheads_k, d)
        // o: (batch_size, seqlen_q, nheads, d)

        // alibi_slopes:(batch_size, nheads) or (nhead)
        // lse: (batch_size, nheads, seqlen_q)

        ck_tile::index_t stride_q = q.stride(1);
        ck_tile::index_t stride_k = k.stride(1);
        ck_tile::index_t stride_v = v.stride(1);
        ck_tile::index_t stride_o = out.stride(1);

        ck_tile::index_t nhead_stride_q = q.stride(2);
        ck_tile::index_t nhead_stride_k = k.stride(2);
        ck_tile::index_t nhead_stride_v = v.stride(2);
        ck_tile::index_t nhead_stride_o = out.stride(2);

        ck_tile::index_t batch_stride_q = q.stride(0);
        ck_tile::index_t batch_stride_k = k.stride(0);
        ck_tile::index_t batch_stride_v = v.stride(0);
        ck_tile::index_t batch_stride_o = out.stride(0);

        StorageView rng_state({4}, DataType::INT32); // idk
        auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.buffer());
        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);

        return fmha_fwd_args{q.buffer(),
                             k.buffer(),
                             v.buffer(),
                             nullptr, // bias
                             nullptr, // q_descale_ptr
                             nullptr, // k_descale_ptr
                             nullptr, // v_descale_ptr
                             nullptr, // rand_val_ptr
                             nullptr, // lse_ptr
                             out.buffer(),
                             nullptr, // seqstart_q_ptr
                             nullptr, // seqstart_k_ptr
                             nullptr, // seqlen_q_ptr
                             nullptr, // seqlen_k_ptr
                             nullptr, // cu_seqlen_q_ptr
                             nullptr, // cu_seqlen_k_ptr
                             seqlen_q,
                             seqlen_k,
                             b,
                             seqlen_q,      // max_seqlen_q
                             d,             // hdim_q
                             d,             // hdim_v
                             h,             // nhead
                             h_k,           // nhead_k
                             softmax_scale, // scale_s
                             0.0f,          // logits_soft_cap
                             stride_q,
                             stride_k,
                             stride_v,
                             0, // stride_alibi_slopes,
                             0, // stride_randval,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             0, // nhead_stride_bias, FA without bias
                             0, // nhead_stride_randval
                             0, // nhead_stride_lse
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             0, // batch_stride_bias, FA without bias
                             0, // batch_stride_randval
                             0, // batch_stride_lse
                             batch_stride_o,
                             mask.left,
                             mask.right,
                             0, // sink_size
                             static_cast<ck_tile::index_t>(mask.type),
                             0, // min_seqlen_q
                             0.0f, // p_drop
                             false, // has_dropout_randval,
                             drop_seed_offset};
    }
#endif

    template<>
    void FlashAttention::compute<Device::CUDA>(StorageView& queries,
                                               StorageView& keys,
                                               StorageView& values,
                                               StorageView& output,
                                               StorageView* cached_keys,
                                               StorageView* cached_values,
                                               StorageView* attention,
                                               bool return_normalized_attention,
                                               StorageView* rotary_cos,
                                               StorageView* rotary_sin,
                                               const bool rotary_interleave,
                                               StorageView* alibi,
                                               dim_t offset) const {
#ifdef CT2_WITH_FLASH_ATTN
      if (cached_keys || cached_values || attention || rotary_cos || rotary_sin
        || rotary_interleave || alibi || offset) {
#endif
          throw std::runtime_error("Flash attention 2 is not supported");
#ifdef CT2_WITH_FLASH_ATTN
      }

      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      std::string dtype_str = dtype == DataType::FLOAT16 ? "fp16" : "bf16";
      int window_size_left = _sliding_window > 0 ? _sliding_window : -1;
      int window_size_right = _sliding_window > 0 ? 0 : -1;

      const auto shape = queries.shape();
      const int batch_size = shape[0];
      int seqlen_q = shape[1];
      int num_heads = shape[2];
      const int head_size_og = shape[3];
      const int seqlen_k = keys.dim(1);
      const int num_heads_k = keys.dim(2);

      mask_info mask = mask_info::decode("0", seqlen_q, seqlen_k);

      output.resize(queries.shape());
      auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
      const int head_size = round_multiple(head_size_og, 8);
      const int head_size_rounded = round_multiple(head_size, 32);
      const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
      const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

      cudaStream_t stream = ctranslate2::cuda::get_cuda_stream();
      ck_tile::stream_config stream_config{stream};

      auto traits = get_ck_fmha_fwd_traits(mask, dtype_str, head_size);
      auto args = get_ck_fmha_fwd_args(mask,
                                       batch_size,
                                       seqlen_q,
                                       seqlen_k,
                                       num_heads,
                                       num_heads_k,
                                       head_size,
                                       queries,
                                       keys,
                                       values,
                                       output,
                                       _queries_scale);

      float status = fmha_fwd(traits, args, stream_config);
      if (status < 0)
        throw std::runtime_error("invalid argument for fmha_fwd");
#endif
    }

  }
}
