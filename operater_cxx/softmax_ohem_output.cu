/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file softmax_output.cu
 * \brief
 * \author Bing Xu
*/

#include "softmax_ohem_output-inl.h"
#include "../../../include/mxnet/base.h"
#include "../mxnet_op.h"
#include "../../common/cuda_utils.h"

#define SOFTMAXOHEM_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
    namespace cuda {

        template<typename DType>
        __global__ void SoftmaxOHEMGradKernel(DType *dst,
                                                    const DType *src,
                                                    const DType *label,
                                                    const DType ignore_label,
                                                    const float thresh,
                                                    const int class_num,
                                                    const int sample_out_size,
                                                    const int num) {
            int page_size = sample_out_size * class_num;
            CUDA_KERNEL_LOOP(i, num){

                int n = i % sample_out_size;
                int y = i / sample_out_size;

                const index_t k = static_cast<int>(label[y * sample_out_size + n]);

                if (k == static_cast<int>(ignore_label)||src[y * page_size + k * sample_out_size + n]>thresh) {
                    for (index_t x = 0; x < class_num; ++x) {
                        dst[y * page_size + x * sample_out_size + n] = DType(0.0f);
                    }
                } else {
                    for (index_t x = 0; x < class_num; ++x) {
                        if (x == k) {
                            dst[y * page_size + k * sample_out_size + n] =  src[y * page_size + k * sample_out_size + n] - 1.0f;
                        } else {
                            dst[y * page_size + x * sample_out_size + n] =   src[y * page_size + x * sample_out_size + n];
                        }
                    }
                }
            }
        }

        template<typename DType>
        __global__ void  GetPKernel(DType *dst,
                                    const DType *src,
                                    const DType *label,
                                    const DType ignore_label,
                                    const int class_num,
                                    const int sample_out_size,
                                    const int num)
        {
            int page_size = sample_out_size * class_num;
            CUDA_KERNEL_LOOP(i, num){
                int n = i % sample_out_size;
                int y = i / sample_out_size;
                const index_t k = static_cast<int>(label[y * sample_out_size + n]);
                if (k == static_cast<int>(ignore_label)){
                    dst[y * sample_out_size + n] = 1.0f;
                }else{
                    dst[y * sample_out_size + n] =  src[y * page_size + k * sample_out_size + n];
                }
            }
        }
    }
}

namespace mshadow {
    template<typename DType>
    inline void SoftmaxOHEMGrad(Tensor<gpu, 3, DType> dst,
                                 const Tensor<gpu, 3, DType> &src,
                                 const Tensor<gpu, 2, DType> &label,
                                 const DType ignore_label,
                                 const float thresh) {

        CHECK_EQ(dst.CheckContiguous(), true);
        cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
        int num_kernels = dst.size(0) * dst.size(2);

        DType *out_ptr = dst.dptr_;
        using namespace mxnet::op::mxnet_op;

//        LOG(INFO) << dst.size(0) << ","<< dst.size(1)<<","<<dst.size(2);
//        LOG(INFO) << src.size(0) << ","<< src.size(1)<<","<<src.size(2);
//        LOG(INFO) << label.size(0) << ","<< label.size(1);

        cuda::SoftmaxOHEMGradKernel<DType> << < cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
                0, stream >> > (out_ptr,
                src.dptr_,
                label.dptr_,
                ignore_label,
                thresh,
                dst.size(1),
                dst.size(2),
                num_kernels);
        SOFTMAXOHEM_CUDA_CHECK(cudaPeekAtLastError());

    }

    template<typename DType>
    inline void CopyProbality( Tensor<gpu, 2, DType> dst,
                                const Tensor<gpu, 3, DType> &src,
                                const Tensor<gpu, 2, DType> &label,
                                const DType ignore_label) {

        CHECK_EQ(dst.CheckContiguous(), true);
        cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
        int num_kernels = dst.size(0) * dst.size(1);

        DType *out_ptr = dst.dptr_;
        using namespace mxnet::op::mxnet_op;


        cuda::GetPKernel<DType> << < cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
                0, stream >> > (out_ptr,
                src.dptr_,
                label.dptr_,
                ignore_label,
                src.size(1),
                src.size(2),
                num_kernels);
        SOFTMAXOHEM_CUDA_CHECK(cudaPeekAtLastError());

    }
}


namespace mxnet {
    namespace op {
        template<>
        Operator *CreateOp<gpu>(SoftmaxOHEMOutputParam param, int dtype) {
            Operator *op = NULL;
            MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {op = new SoftmaxOHEMOutputOp<gpu, DType>(param);})
            return op;
        }
    }  // namespace op
}  // namespace mxnet


