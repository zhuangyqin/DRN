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
 * \file softmax_output-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SOFTMAX_OHEM_OUTPUT_INL_H_
#define MXNET_OPERATOR_SOFTMAX_OHEM_OUTPUT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../tensor/ordering_op-inl.h"

namespace mxnet {
    namespace op {

        namespace softmaxOHEMout_enum {
            enum SoftmaxOHEMOutputOpInputs {
                kData, kLabel
            };
            enum SoftmaxOHEMOutputOpOutputs {
                kOut
            };
            enum SoftmaxOHEMOutputNormType {
                kNull, kBatch, kValid
            };
            enum SoftmaxOHEMOutputOpResource {
                kTempSpace, kTempSpace1
            };
        }  // namespace softmaxout_enum

        struct SoftmaxOHEMOutputParam : public dmlc::Parameter<SoftmaxOHEMOutputParam> {

            float grad_scale;
            float ignore_label;
            float thresh;
            int min_keep;
            bool multi_output;
            bool use_ignore;
            bool preserve_shape;
            int normalization;
            bool out_grad;
            uint64_t workspace;


            DMLC_DECLARE_PARAMETER(SoftmaxOHEMOutputParam) {

                    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
                            .describe("Scales the gradient by a float factor.");

                    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f).describe(
                    "The instances whose `labels` == `ignore_label` will be ignored "
                    "during backward, if `use_ignore` is set to ``true``).");

                    DMLC_DECLARE_FIELD(thresh).set_default(-1.0f)
                    .describe("the threshhold for sampling the weight");

                    DMLC_DECLARE_FIELD(min_keep)
                    .set_default(256).describe("min_keep back grad num ");

                    DMLC_DECLARE_FIELD(multi_output).set_default(true)
                    .describe("If set to ``true``, the softmax function will be computed along "
                    "axis ``1``. This is applied when the shape "
                    "of input array differs from the shape of label array.");

                    DMLC_DECLARE_FIELD(use_ignore).set_default(false)
                    .describe("If set to ``true``, the `ignore_label` value will not contribute "
                    "to the backward gradient.");

                    DMLC_DECLARE_FIELD(preserve_shape).set_default(false)
                    .describe("If set to ``true``, the softmax function will be computed along "
                    "the last axis (``-1``).");

                    DMLC_DECLARE_FIELD(normalization)
                    .add_enum("null", softmaxOHEMout_enum::kNull)
                    .add_enum("batch", softmaxOHEMout_enum::kBatch)
                    .add_enum("valid", softmaxOHEMout_enum::kValid)
                    .set_default(softmaxOHEMout_enum::kNull)
                    .describe("Normalizes the gradient.");

                    DMLC_DECLARE_FIELD(out_grad)
                    .set_default(false)
                    .describe("Multiplies gradient with output gradient element-wise.");

                    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
                    .describe("Maximum temporary workspace allowed for convolution (MB).");

            };
        };

        using namespace mshadow;

        template<typename xpu, typename DType>
        class SoftmaxOHEMOutputOp : public Operator {
        public:
            explicit SoftmaxOHEMOutputOp(SoftmaxOHEMOutputParam param) {
                this->param_ = param;
                // convert MBytes first to Bytes and then to elements.
                param_.workspace = (param_.workspace << 20) / sizeof(DType);
            }

            virtual void Forward(const OpContext &ctx,
                                 const std::vector <TBlob> &in_data,
                                 const std::vector <OpReqType> &req,
                                 const std::vector <TBlob> &out_data,
                                 const std::vector <TBlob> &aux_args) {

                using namespace mshadow;
                using namespace mshadow::expr;
                CHECK_EQ(in_data.size(), 2U) << "SoftmaxOutput Input: [data, label]";
                CHECK_EQ(out_data.size(), 1U) << "SoftmaxOutput Output: [output]";
                Stream <xpu> *s = ctx.get_stream<xpu>();

                if (param_.multi_output) {
                    int n = in_data[softmaxOHEMout_enum::kData].size(0);
                    int k = in_data[softmaxOHEMout_enum::kData].size(1);
                    Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[softmaxOHEMout_enum::kData].Size() / n / k));
                    Tensor<xpu, 3, DType> data =
                            in_data[softmaxOHEMout_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);
                    Tensor<xpu, 3, DType> out =
                            out_data[softmaxOHEMout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
                    Softmax(out, data);
//                    LOG(INFO) << "Softmax Forward OK!";
                    // sort the probality
                } else {
                    LOG(ERROR) << "Now is only support for the multi_outputs";
                }

            }


            virtual void Backward(const OpContext &ctx,
                                  const std::vector <TBlob> &out_grad,
                                  const std::vector <TBlob> &in_data,
                                  const std::vector <TBlob> &out_data,
                                  const std::vector <OpReqType> &req,
                                  const std::vector <TBlob> &in_grad,
                                  const std::vector <TBlob> &aux_args) {

                using namespace mshadow;
                using namespace mxnet::op;
                using namespace mshadow::expr;

                CHECK_EQ(in_data.size(), 2U);
                CHECK_EQ(out_grad.size(), 1U);
                CHECK_GE(in_grad.size(), 1U);
                CHECK_GE(req.size(), 1U);

                Stream <xpu> *s = ctx.get_stream<xpu>();

                if (param_.multi_output && param_.thresh > -1) {

                    int n = out_data[softmaxOHEMout_enum::kOut].size(0);
                    int k = out_data[softmaxOHEMout_enum::kOut].size(1);

                    Shape<3> s3 = Shape3(n, k, static_cast<int>(out_data[softmaxOHEMout_enum::kOut].Size() / n / k));
                    Shape<2> s2 = Shape2(s3[0], s3[2]);
                    Shape<1> s1 = Shape1(s3[0] * s3[2]);

                    Tensor<xpu, 2, DType> label = in_data[softmaxOHEMout_enum::kLabel].get_with_shape<xpu, 2, DType>(s2,
                                                                                                                     s);
                    Tensor<xpu, 3, DType> out = out_data[softmaxOHEMout_enum::kOut].get_with_shape<xpu, 3, DType>(s3,
                                                                                                                  s);
                    Tensor<xpu, 3, DType> grad = in_grad[softmaxOHEMout_enum::kData].get_with_shape<xpu, 3, DType>(s3,
                                                                                                                   s);

                    index_t total_size = 5 * label.size(0) * label.size(1);

                    Tensor<xpu, 1, DType> total_workspace = ctx.requested[softmaxOHEMout_enum::kTempSpace].get_space_typed<xpu, 1, DType>(
                            mshadow::Shape1(total_size), s);

                    Tensor<xpu, 2, DType> Prob_2D = Tensor<xpu, 2, DType>(total_workspace.dptr_, s2, s);

                    Tensor<xpu, 1, DType> indice_1D = Tensor<xpu, 1, DType>(
                            total_workspace.dptr_ + label.size(0) * label.size(1), s1, s);

                    Tensor<xpu, 1, DType> sort_workspace = Tensor<xpu, 1, DType>(total_workspace.dptr_ + 2 * label.size(0) * label.size(1),Shape1(3 * label.size(0) * label.size(1)), s);

                    index_t valid_cnt = label.size(0) * label.size(1);

                    float thresh = GetThresh(Prob_2D, indice_1D, sort_workspace, out, label, ctx, param_.thresh,
                                             param_.min_keep,
                                             static_cast<index_t>(param_.ignore_label), valid_cnt);

                    SoftmaxOHEMGrad(grad, out, label, static_cast<DType>(param_.ignore_label), thresh);

                    grad *= DType(param_.grad_scale / valid_cnt);

                } else {
                    LOG(ERROR) << "Not Support!";
                }
            }

        private:
            SoftmaxOHEMOutputParam param_;

            template<typename Dtype>
            inline float
            GetThresh(Tensor<cpu, 2, Dtype> prob, Tensor<cpu, 1, DType> indice, Tensor<cpu, 1, DType> sort_workspace,
                      const Tensor<cpu, 3, Dtype> out,
                      const Tensor<cpu, 2, Dtype> label, const OpContext &ctx, float thresh, int min_keep,
                      int ignore_label, index_t &val_cnt) {

                for (index_t i = 0; i < label.size(0); i++)
                    for (index_t j = 0; j < label.size(1); j++) {
                        index_t class_id = label[i][j];
                        if (class_id == static_cast<index_t>(ignore_label)) {
                            prob[i][j] = 1.0f;
                        } else {
                            prob[i][j] = out[i][class_id][j];
                        }
                    }

                Tensor<cpu, 1, Dtype> prob1d = prob.FlatTo1D();

//                for(index_t i =0 ;i< label.size(0) * label.size(1);i++){
//                    LOG(INFO) << prob1d[i];
//                }

                TopKParam topk_param;
                topk_param.axis = -1;
                topk_param.is_ascend = true;

                topk_param.k = 0;
                topk_param.ret_typ = topk_enum::kReturnIndices;

                TBlob input = TBlob(prob1d);
                TBlob output = TBlob(indice);

                std::vector <TBlob> outputs;
                outputs.push_back(output);

                TopKImpl<cpu>(ctx.run_ctx,&sort_workspace, input, outputs, topk_param);

//                LOG(INFO) << "Backward: Sort OK!";
                if (min_keep > label.size(0) * label.size(1)) {
                    min_keep = label.size(0) * label.size(1);
                }

                float tmp_thresh = prob1d[indice[min_keep-1]];
//                LOG(INFO) << tmp_thresh;
                if (tmp_thresh > thresh) {
                    thresh = tmp_thresh;
                }

                val_cnt = label.size(0) * label.size(1);
                for (index_t i = 0; i < prob.size(0); i++)
                    for (index_t j = 0; j < prob.size(1); j++) {
                        if ((prob[i][j] > thresh) || (label[i][j] == ignore_label)) {
                            val_cnt--;
                        }
                    }

                LOG(INFO) << val_cnt << "," << thresh;
                val_cnt = val_cnt == 0 ? 1 : val_cnt;

                return thresh;
            }

            template<typename Dtype>
            inline float
            GetThresh(Tensor<gpu, 2, Dtype> prob, Tensor<gpu, 1, DType> indice, Tensor<gpu, 1, DType> sort_workspace,
                      const Tensor<gpu, 3, Dtype> out, const Tensor<gpu, 2, Dtype> label, const OpContext &ctx,
                      float thresh, int min_keep,  int ignore_label, index_t &val_cnt) {

                using namespace mshadow;

                CopyProbality(prob, out, label, static_cast<DType>(ignore_label));

                int total_size = 3*label.size(0)* label.size(1);
//
                Tensor<cpu, 1, DType> total_workspace = ctx.requested[softmaxOHEMout_enum::kTempSpace].get_host_space_typed<1, DType>(
                        mshadow::Shape1(total_size));

                Tensor<cpu, 2, DType> label_cpu =  Tensor<cpu, 2, DType>(total_workspace.dptr_,mshadow::Shape2(label.size(0), label.size(1)));
                Copy(label_cpu, label, label.stream_);
                MSHADOW_CUDA_CALL(cudaStreamSynchronize(label.stream_->stream_));

                Tensor<cpu, 2, DType> prob2d_cpu =  Tensor<cpu, 2, DType>(total_workspace.dptr_ + label.size(0)* label.size(1),mshadow::Shape2(label.size(0), label.size(1)));
                Copy(prob2d_cpu, prob, prob.stream_);
                MSHADOW_CUDA_CALL(cudaStreamSynchronize(prob.stream_->stream_));

                Tensor<cpu, 1, DType> indice_cpu =  Tensor<cpu, 1, DType>(total_workspace.dptr_ + 2* label.size(0)* label.size(1),mshadow::Shape1(label.size(0)*label.size(1)));

                TopKParam topk_param;
                topk_param.axis = -1;
                topk_param.is_ascend = true;

                topk_param.k = 0;
                topk_param.ret_typ = topk_enum::kReturnIndices;
                Tensor<gpu, 1, Dtype> prob1D = prob.FlatTo1D();

                TBlob input = TBlob(prob1D);
                TBlob output = TBlob(indice);
                std::vector <TBlob> outputs;
                outputs.push_back(output);

                TopKImpl<gpu>(ctx.run_ctx,&sort_workspace, input, outputs, topk_param);

                Copy(indice_cpu, indice, indice.stream_);
                MSHADOW_CUDA_CALL(cudaStreamSynchronize(indice.stream_->stream_));

                if (min_keep > label.size(0) * label.size(1)) {
                    min_keep = label.size(0) * label.size(1);
                }

                index_t index = indice_cpu[min_keep-1];

                Tensor<cpu, 1, Dtype> prob1d_cpu = prob2d_cpu.FlatTo1D();
                float tmp_thresh = prob1d_cpu[index];

//                for (index_t i = 0; i < indice_cpu.size(0); i++)
//                    LOG(INFO) << indice_cpu[i]<<","<<prob1d_cpu[i];

                if (tmp_thresh > thresh) {
                    thresh = tmp_thresh;
                }

                val_cnt = label.size(0)* label.size(1);
                for(index_t i = 0;i< label.size(0);++i){
                    for(index_t j=0;j<label.size(1);++j) {
                        if (prob2d_cpu[i][j] > thresh || label_cpu[i][j] == ignore_label) {
                            val_cnt--;
                        }
                    }
                }
                LOG(INFO) << val_cnt << "," << thresh;

                val_cnt = val_cnt==0?1:val_cnt;
                return thresh;
            }
        };  // class SoftmaxOutputOp

// Decalre Factory function, used for dispatch specialization
        template<typename xpu>
        Operator *CreateOp(SoftmaxOHEMOutputParam param, int dtype);

#if DMLC_USE_CXX11

        class SoftmaxOHEMOutputProp : public OperatorProperty {
        public:
            std::vector<std::string> ListArguments() const override {
                return {"data", "label"};
            }

            void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
                param_.Init(kwargs);
            }

            std::map<std::string, std::string> GetParams() const override {
                return param_.__DICT__();
            }

            bool InferShape(std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape,
                            std::vector<TShape> *aux_shape) const override {
                using namespace mshadow;
                CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
                const TShape &dshape = in_shape->at(0);
                if (dshape.ndim() == 0) return false;

                // label.shape == data.shape: use probability as label
                if (dshape != (*in_shape)[softmaxOHEMout_enum::kLabel]) {
                    if (param_.multi_output) {
                        TShape lshape1 = Shape2(dshape[0], dshape.Size() / dshape[0] / dshape[1]);
                        TShape lshape2(dshape.ndim() - 1);
                        lshape2[0] = dshape[0];
                        for (index_t i = 2; i < dshape.ndim(); ++i)
                            lshape2[i - 1] = dshape[i];
                        TShape lshape3 = dshape;
                        lshape3[1] = 1;
                        if (in_shape->at(softmaxOHEMout_enum::kLabel).ndim() == 0) {
                            in_shape->at(softmaxOHEMout_enum::kLabel) = lshape1;
                        } else if (in_shape->at(softmaxOHEMout_enum::kLabel) == lshape1) {
                        } else if (in_shape->at(softmaxOHEMout_enum::kLabel) == lshape2) {
                        } else if (in_shape->at(softmaxOHEMout_enum::kLabel) == lshape3) {
                        } else {
                            std::ostringstream os;
                            os << "Expecting " << lshape1 << " or " << lshape2
                               << ". But got " << in_shape->at(softmaxOHEMout_enum::kLabel);
                            throw InferShapeError(os.str(), softmaxOHEMout_enum::kLabel);
                        }
                    } else {
                        TShape label_shape(dshape.ndim() - 1);
                        for (index_t i = 0; i + 1 < dshape.ndim(); ++i)
                            label_shape[i] = dshape[i];
                        SHAPE_ASSIGN_CHECK(*in_shape, softmaxOHEMout_enum::kLabel, label_shape);
                    }
                }
                out_shape->clear();
                out_shape->push_back(dshape);
                return true;
            }

            bool InferType(std::vector<int> *in_type,
                           std::vector<int> *out_type,
                           std::vector<int> *aux_type) const override {
                CHECK_GE(in_type->size(), 1U);
                int dtype = (*in_type)[0];
                CHECK_NE(dtype, -1) << "First input must have specified type";
                for (index_t i = 0; i < in_type->size(); ++i) {
                    if ((*in_type)[i] == -1) {
                        (*in_type)[i] = dtype;
                    } else {
                        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                                       << "Expected " << dtype << " v.s. given "
                                                       << (*in_type)[i] << " at " << ListArguments()[i];
                    }
                }
                out_type->clear();
                out_type->push_back(dtype);
                return true;
            }

            OperatorProperty *Copy() const override {
                auto ptr = new SoftmaxOHEMOutputProp();
                ptr->param_ = param_;
                return ptr;
            }

            std::string TypeString() const override {
                return "SoftmaxOHEMOutput";
            }

            std::vector<int> DeclareBackwardDependency(
                    const std::vector<int> &out_grad,
                    const std::vector<int> &in_data,
                    const std::vector<int> &out_data) const override {
                if (param_.out_grad) {
                    return {in_data[softmaxOHEMout_enum::kLabel], out_data[softmaxOHEMout_enum::kOut],
                            out_grad[softmaxOHEMout_enum::kOut]};
                } else {
                    return {in_data[softmaxOHEMout_enum::kLabel], out_data[softmaxOHEMout_enum::kOut]};
                }
            }

            std::vector<std::pair<int, void *> > BackwardInplaceOption(
                    const std::vector<int> &out_grad,
                    const std::vector<int> &in_data,
                    const std::vector<int> &out_data,
                    const std::vector<void *> &in_grad) const override {
                return {{out_data[softmaxOHEMout_enum::kOut], in_grad[softmaxOHEMout_enum::kData]}};
            }

            std::vector<std::pair<int, void *> > ForwardInplaceOption(
                    const std::vector<int> &in_data,
                    const std::vector<void *> &out_data) const override {
                return {{in_data[softmaxOHEMout_enum::kData], out_data[softmaxOHEMout_enum::kOut]}};
            }

            std::vector<ResourceRequest> BackwardResource(
                    const std::vector<TShape> &in_shape) const override {
                return {ResourceRequest::kTempSpace};
            }

            Operator *CreateOperator(Context ctx) const override {
                LOG(FATAL) << "Not Implemented.";
                return NULL;
            }

            Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const override;

        protected:
            SoftmaxOHEMOutputParam param_;
        };  // class SoftmaxOutputProp
#endif  // DMLC_USE_CXX11

    }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
