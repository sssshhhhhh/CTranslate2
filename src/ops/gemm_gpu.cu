#include "ctranslate2/ops/gemm.h"

#ifdef CT2_USE_HIP
#include <hipblaslt/hipblaslt-ext.hpp>
#endif
#include "cuda/helpers.h"
#include "type_dispatch.h"

#include <spdlog/spdlog.h>

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename In, typename Out>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       StorageView& c,
                       const dim_t m,
                       const dim_t n,
                       const dim_t k,
                       const dim_t lda,
                       const dim_t ldb,
                       const dim_t ldc,
                       const StorageView*,
                       const StorageView* bias,
                       const StorageView* residual) const {
#ifdef CT2_USE_HIP
      // hipBLAS assumes column-major storage, so swap a and b accordingly.
      if constexpr (!std::is_same_v<Out, int32_t>) {
        hipStream_t stream = cuda::get_cuda_stream();
        hipblasLtHandle_t handle = cuda::get_cublas_handle();
        void* workspace = cuda::get_cublas_workspace();
        hipblasOperation_t trans_a = _trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        hipblasOperation_t trans_b = _trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;

        hipblaslt_ext::GemmPreference gemm_pref;
        gemm_pref.setMaxWorkspaceBytes(cuda::max_workspace_size);
        hipDataType data_type;
        if constexpr (std::is_same_v<In, float>)
          data_type = HIP_R_32F;
        else if constexpr (std::is_same_v<In, float16_t>)
          data_type = HIP_R_16F;
        else if constexpr (std::is_same_v<In, bfloat16_t>)
          data_type = HIP_R_16BF;
        hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
        // HIPBLAS_COMPUTE_16F not supported
        hipblaslt_ext::Gemm gemm(handle,
                                 trans_b,
                                 trans_a,
                                 data_type,
                                 data_type,
                                 data_type,
                                 data_type,
                                 compute_type);

        hipblaslt_ext::GemmEpilogue gemm_epilogue;
        if (bias) {
          gemm_epilogue.setMode(HIPBLASLT_EPILOGUE_BIAS);
          gemm_epilogue.setBiasDataType(data_type);
        }

        const dim_t batch_count = 1;
        hipblaslt_ext::GemmInputs inputs;
        inputs.setA(cuda::device_cast(b.data<In>()));
        inputs.setB(cuda::device_cast(a.data<In>()));
        if (residual && _beta == 0.f)
          inputs.setC(cuda::device_cast(residual->data<Out>()));
        else
          inputs.setC(cuda::device_cast(c.data<Out>()));
        inputs.setD(cuda::device_cast(c.data<Out>()));
        if (bias)
          inputs.setBias(bias->data<Out>());
        const float alpha = _alpha;
        const float beta = residual && _beta == 0.f ? 1.f : _beta;
        inputs.setAlpha(&alpha);
        inputs.setBeta(&beta);
        CUBLAS_CHECK(gemm.setProblem(n, m, k, batch_count, gemm_epilogue, inputs));

        const int request_solutions = 1;
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
        CUBLAS_CHECK(gemm.algoGetHeuristic(request_solutions, gemm_pref, heuristic_result));
        // heuristic workspace size is always 0?

        if (!heuristic_result.empty()) {
          gemm.setMaxWorkspaceBytes(cuda::max_workspace_size);
          CUBLAS_CHECK(gemm.initialize(heuristic_result[0].algo, workspace));
          CUBLAS_CHECK(gemm.run(stream));

          if (residual && _beta != 0.f || _activation_type) {
            const StorageView* post_residual = _beta == 0.f ? nullptr : residual;
            apply_bias_and_activation(c, nullptr, _activation_type, post_residual);
          }

          return;
        }
        spdlog::warn("No valid solution found for gemm.");
      }
#endif
      primitives<D>::gemm(_a_is_packed, _b_is_packed,
                          _trans_a, _trans_b,
                          m, n, k,
                          _alpha,
                          a.data<In>(), lda,
                          b.data<In>(), ldb,
                          _beta,
                          c.data<Out>(), ldc,
                          static_cast<const Out*>(nullptr));

      apply_bias_and_activation(c, bias, _activation_type, residual);
    }

#define DECLARE_IMPL(In, Out)                                                       \
    template void                                                                   \
    Gemm::compute<Device::CUDA, In, Out>(const StorageView& a,                      \
                                         const StorageView& b,                      \
                                         StorageView& c,                            \
                                         const dim_t m,                             \
                                         const dim_t n,                             \
                                         const dim_t k,                             \
                                         const dim_t lda,                           \
                                         const dim_t ldb,                           \
                                         const dim_t ldc,                           \
                                         const StorageView* a_shift_compensation,   \
                                         const StorageView* bias,                   \
                                         const StorageView* residual) const;

    DECLARE_IMPL(int8_t, int32_t)
    DECLARE_IMPL(float, float)
    DECLARE_IMPL(float16_t, float16_t)
    DECLARE_IMPL(bfloat16_t, bfloat16_t)

  }
}
