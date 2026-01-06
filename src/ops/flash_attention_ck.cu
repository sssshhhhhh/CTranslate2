#include "ctranslate2/ops/flash_attention.h"

#ifdef CT2_WITH_FLASH_ATTN
#include "fmha_fwd.hpp"
#include "mask.hpp"
#endif
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {
#ifdef CT2_WITH_FLASH_ATTN
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
    void FlashAttention::compute<Device::CUDA>(StorageView& queries,        // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
                                               StorageView& keys,           // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
                                               StorageView& values,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
                                               StorageView& output,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
                                               StorageView* cached_keys,    // nullptr
                                               StorageView* cached_values,  // nullptr
                                               StorageView* attention,      // nullptr
                                               bool return_normalized_attention,
                                               StorageView* rotary_cos,     // nullptr
                                               StorageView* rotary_sin,     // nullptr
                                               const bool rotary_interleave,
                                               StorageView* alibi,          // nullptr
                                               dim_t offset) const {
#ifdef CT2_WITH_FLASH_ATTN
      if (cached_keys || cached_values || attention
          || rotary_cos || rotary_sin || rotary_interleave || alibi || offset) {
        throw std::runtime_error("Model does not support FlashAttention");
      }

      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      std::string dtype_str = dtype == DataType::FLOAT16 ? "fp16" : "bf16";
      const auto shape = queries.shape();
      const int batch_size = shape[0];
      int seqlen_q = shape[1];
      int num_heads = shape[2];
      const int head_size_og = shape[3];
      const int seqlen_k = keys.dim(1);
      const int num_heads_k = keys.dim(2);

      std::string mask_identify = _is_causal && seqlen_q != 1 ? "b:-1,0" : "0";
      mask_info mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k);;

      output.resize(queries.shape());
      auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
      const int head_size = round_multiple(head_size_og, 8);

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
#else
      throw std::invalid_argument("This CTranslate2 package was not compiled with FlashAttention");
#endif
    }

  }
}
