#include "ctranslate2/ops/gemm.h"

#ifdef CT2_USE_HIP
#include <hipblaslt/hipblaslt.h>
#endif
#include "cuda/helpers.h"

#include <spdlog/spdlog.h>

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    constexpr hipDataType hip_data_type() {
      if constexpr (std::is_same_v<T, float>)
        return HIP_R_32F;
      else if constexpr (std::is_same_v<T, float16_t>)
        return HIP_R_16F;
      else if constexpr (std::is_same_v<T, bfloat16_t>)
        return HIP_R_16BF;
      else if constexpr (std::is_same_v<T, float8_t>)
        return HIP_R_8F_E4M3;
      else if constexpr (std::is_same_v<T, bfloat8_t>)
        return HIP_R_8F_E5M2;
    }

    template <typename T>
    constexpr bool is_lowp() {
      return std::is_same_v<T, float8_t> || std::is_same_v<T, bfloat8_t>;
    }

    template <Device D, typename In, typename Out, typename Bias>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       StorageView& c,
                       const StorageView*,
                       const StorageView* bias,
                       const StorageView* residual,
                       const StorageView* scale_a,
                       const StorageView* scale_b,
                       const float scale_c) const {
      // c = _activation_type(_alpha * (scale_a * a) @ (scale_b * b) + bias + residual) * scale_c
      // If _beta != 0, c = _activation_type(_alpha * (scale_a * a) @ (scale_b * b) + bias + _beta * c) + residual
      // scale_a/b/c is only supported with low precision matrices a/b/c
      // scale_a/b is supported with 2 modes:
      // - Scalar applied to the whole tensor
      // - Vector outer-dim len (m/n) [m,k]@[k,n]=[m,n] for each row/col

      const dim_t k = a.dim(_trans_a ? -2 : -1);
      const dim_t n = b.dim(_trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.
      const dim_t lda = _trans_a ? m : k;
      const dim_t ldb = _trans_b ? k : n;
      const dim_t ldc = n;

      {
        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 2] = a.dim(_trans_a ? -1 : -2); // m
        output_shape[output_shape.size() - 1] = n;
        c.resize(std::move(output_shape));
      }

#ifdef CT2_USE_HIP
      // Find supported solutions in msgpack hipblaslt/library/TensileLibrary_lazy_gfx0000.dat
      // e.g. gfx1201 vector scale on b (weights) only supported with bf16 output
      // hipBLAS assumes column-major storage, so swap a and b accordingly.
      // Silently fails:
      // HIPBLASLT_ORDER_ROW doesn't do anything
      // SCALE_OUTER_VEC_32F on DESC_B_SCALE_MODE stays on scalar mode
      // hipblasLtMatmulHeuristicResult_t.workspaceSize is always 0
      if constexpr (!std::is_same_v<Out, int32_t>) {
        hipStream_t stream = cuda::get_cuda_stream();
        hipblasLtHandle_t handle = cuda::get_cublas_handle();
        void* workspace = cuda::get_cublas_workspace();
        hipblasOperation_t trans_a = _trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        hipblasOperation_t trans_b = _trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;

        constexpr hipDataType in_type = hip_data_type<In>();
        constexpr hipDataType out_type = hip_data_type<Out>();
        constexpr hipDataType bias_type = hip_data_type<Bias>();

        hipblasLtMatrixLayout_t mat_a, mat_b, mat_c;
        CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, in_type, ldb, _trans_b ? n : k, ldb));
        CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, in_type, lda, _trans_a ? k : m, lda));
        CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, out_type, n, m, ldc));

        hipblasLtMatmulDesc_t matmul;
        CUBLAS_CHECK(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                     HIPBLASLT_MATMUL_DESC_TRANSA,
                                                     &trans_a,
                                                     sizeof(int32_t)));
        CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                     HIPBLASLT_MATMUL_DESC_TRANSB,
                                                     &trans_b,
                                                     sizeof(int32_t)));

        bool act_fused = true;
        if (bias) {
          hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
          if (_activation_type) {
            switch (*_activation_type) {
            case ActivationType::ReLU:
              epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
              break;
            case ActivationType::GELU:
            case ActivationType::GELUTanh:
            case ActivationType::GELUSigmoid: // only approx in low precision
              if (*_activation_type == ActivationType::GELUTanh || is_lowp<In>())
                epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
              else
                act_fused = false;
              break;
            case ActivationType::Swish:
              epilogue = HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT;
              break;
            default:
              act_fused = false;
            }
          }

          CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                       HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                       &epilogue,
                                                       sizeof(epilogue)));
          CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                       HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                       &bias_type,
                                                       sizeof(bias_type)));
          const Bias* d_bias = bias->data<Bias>();
          CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                       HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                       &d_bias,
                                                       sizeof(void*)));
        } else if (_activation_type) {
          hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
          switch (*_activation_type) {
          case ActivationType::ReLU:
            epilogue = HIPBLASLT_EPILOGUE_RELU;
            break;
          case ActivationType::GELU:
          case ActivationType::GELUTanh:
          case ActivationType::GELUSigmoid: // only approx in low precision
            if (*_activation_type == ActivationType::GELUTanh || is_lowp<In>())
              epilogue = HIPBLASLT_EPILOGUE_GELU;
            else
              act_fused = false;
            break;
          case ActivationType::Swish:
            epilogue = HIPBLASLT_EPILOGUE_SWISH_EXT;
            break;
          default:
            act_fused = false;
          }
          CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                       HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                       &epilogue,
                                                       sizeof(epilogue)));
        }

        float scale = 1.0f; // scalars are on cpu
        if (scale_a) {
          if (scale_a->is_scalar()) {
            scale *= scale_a->as_scalar<float>();
          } else {
            hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                         HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                         &mode,
                                                         sizeof(uint32_t)));
            const float* d_scale_a = scale_a->data<float>();
            CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                         HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                         &d_scale_a,
                                                         sizeof(float*)));
          }
        }
        if (scale_b) {
          if (scale_b->is_scalar()) {
            scale *= scale_b->as_scalar<float>();
          } else {
            hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                         HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                         &mode,
                                                         sizeof(uint32_t)));
            const float* d_scale_b = scale_b->data<float>();
            CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                         HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                         &d_scale_b,
                                                         sizeof(float*)));
          }
        }
        StorageView scale_d(DataType::FLOAT32, D);
        if (scale_c != 1.f) {
          if (!act_fused || residual && _beta != 0.f)
            throw std::invalid_argument("Gemm scale_c unsupported, remove input_scale from model");
          scale_d.resize({});
          float* d_scale_d = scale_d.data<float>();
          cross_device_primitives<Device::CPU, D>::copy(&scale_c, d_scale_d, 1);
          CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
                                                       HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                       &d_scale_d,
                                                       sizeof(float*)));
        }

        hipblasLtMatmulPreference_t pref;
        CUBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                           HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &cuda::max_workspace_size,
                                                           sizeof(cuda::max_workspace_size)));
        const int request_solutions = 1;
        hipblasLtMatmulHeuristicResult_t heuristic_result[request_solutions];
        int returned_algo_count = 0;
        CUBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                     matmul,
                                                     mat_a,
                                                     mat_b,
                                                     mat_c,
                                                     mat_c,
                                                     pref,
                                                     request_solutions,
                                                     heuristic_result,
                                                     &returned_algo_count));

        if (returned_algo_count > 0) {
          const float alpha = _alpha * scale;
          const float beta = residual && _beta == 0.f ? 1.f : _beta;
          using DeviceIn = cuda::device_type<In>;
          using DeviceOut = cuda::device_type<Out>;
          const DeviceIn* d_a = cuda::device_cast(b.data<In>());
          const DeviceIn* d_b = cuda::device_cast(a.data<In>());
          DeviceOut* d_d = cuda::device_cast(c.data<Out>());
          const DeviceOut* d_c = residual && _beta == 0.f ? cuda::device_cast(residual->data<Out>()) : d_d;

          CUBLAS_CHECK(hipblasLtMatmul(handle,
                                       matmul,
                                       &alpha,
                                       d_a,
                                       mat_a,
                                       d_b,
                                       mat_b,
                                       &beta,
                                       d_c,
                                       mat_c,
                                       d_d,
                                       mat_c,
                                       &heuristic_result[0].algo,
                                       workspace,
                                       cuda::max_workspace_size,
                                       stream));
        }
        CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
        CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
        CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
        CUBLAS_CHECK(hipblasLtMatmulDescDestroy(matmul));
        CUBLAS_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
        if (returned_algo_count > 0) {
          if (residual && _beta != 0.f || !act_fused) {
            const StorageView* post_residual = _beta == 0.f ? nullptr : residual;
            const ActivationType* post_act = act_fused ? nullptr : _activation_type;
            apply_bias_and_activation(c, nullptr, post_act, post_residual);
          }
          return;
        }

        spdlog::warn("No valid solution found for gemm.");
        return;
      }
#endif
      if constexpr (is_lowp<In>())
        throw std::invalid_argument("FP8 not supported");

      primitives<D>::gemm(_a_is_packed, _b_is_packed,
                          _trans_a, _trans_b,
                          m, n, k,
                          _alpha,
                          a.data<In>(), lda,
                          b.data<In>(), ldb,
                          _beta,
                          c.data<Out>(), ldc);

      apply_bias_and_activation(c, bias, _activation_type, residual);
    }

#define DECLARE_IMPL(In, Out, Bias)                                                     \
    template void                                                                       \
    Gemm::compute<Device::CUDA, In, Out, Bias>(const StorageView& a,                    \
                                               const StorageView& b,                    \
                                               StorageView& c,                          \
                                               const StorageView* a_shift_compensation, \
                                               const StorageView* bias,                 \
                                               const StorageView* residual,             \
                                               const StorageView* scale_a,              \
                                               const StorageView* scale_b,              \
                                               const float scale_c) const;

    DECLARE_IMPL(int8_t, int32_t, int32_t)
    DECLARE_IMPL(float, float, float)
    DECLARE_IMPL(float16_t, float16_t, float16_t)
    DECLARE_IMPL(bfloat16_t, bfloat16_t, bfloat16_t)
    DECLARE_IMPL(float8_t, float, float)
    DECLARE_IMPL(float8_t, float16_t, float16_t)
    DECLARE_IMPL(float8_t, bfloat16_t, bfloat16_t)
    DECLARE_IMPL(float8_t, float8_t, float)
    DECLARE_IMPL(float8_t, float8_t, float16_t)
    DECLARE_IMPL(float8_t, float8_t, bfloat16_t)
    DECLARE_IMPL(bfloat8_t, float, float)
    DECLARE_IMPL(bfloat8_t, float16_t, float16_t)
    DECLARE_IMPL(bfloat8_t, bfloat16_t, bfloat16_t)
    DECLARE_IMPL(bfloat8_t, bfloat8_t, float)
    DECLARE_IMPL(bfloat8_t, bfloat8_t, float16_t)
    DECLARE_IMPL(bfloat8_t, bfloat8_t, bfloat16_t)

  }
}
