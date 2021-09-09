/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <nncase/importer/importer.h>
#include <nncase/ir/call.h>
#include <nncase/ir/debug.h>
#include <nncase/ir/module.h>
//#include <nncase/ir/op_utils.h>
//#include <nncase/ir/ops/dequantize.h>
//#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/tensors/transpose.h>
#include <schema_generated.h>
#include <span>
#include <unordered_map>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace nncase::importer {
class tflite_importer {
  public:
    tflite_importer(std::span<const uint8_t> model);

    ir::module_t import(const import_options &options);

  private:
    void convert_op(const tflite::Operator &op);

#define DEFINE_OPCODE(opcode)                                                  \
    void convert_op_##opcode(const tflite::Operator &op);
#include "opcode.def"
#undef DEFINE_OPCODE

    void convert_pool2d(const tflite::Operator &op, reduce_op_t reduce_op,
                        float init_value);
    void convert_binary(const tflite::Operator &op, binary_op_t binary_op,
                        tflite::ActivationFunctionType activation =
                            tflite::ActivationFunctionType_NONE);
    void convert_reduce(const tflite::Operator &op, reduce_op_t reduce_op,
                        float init_value);
    void convert_unary(const tflite::Operator &op, unary_op_t unary_op);
    void convert_resize_image(const tflite::Operator &op,
                              image_resize_mode_t mode);

    ir::shape_t get_ir_shape(const flatbuffers::Vector<int32_t> *shape);
    ir::type get_ir_type(const flatbuffers::Vector<int32_t> *shape,
                         tflite::TensorType tflite_type);
    ir::expr get_tensor_expr(const flatbuffers::Vector<int32_t> *ids,
                             int32_t offset);
    ir::expr get_input_expr(const tflite::Operator &op, int32_t offset);

    ir::call activate(ir::expr input,
                      tflite::ActivationFunctionType activation);

    template <class... TArgs, class = std::enable_if_t<std::conjunction_v<
                                  std::is_convertible<TArgs, int32_t>...>>>
    auto get_input_exprs(const tflite::Operator &op, TArgs &&...offsets) {
        constexpr size_t N = sizeof...(offsets);
        std::array<int32_t, N> offsets_arr{offsets...};
        std::array<ir::expr, N> exprs;
        for (size_t i = 0; i < N; i++) {
            exprs[i] = get_input_expr(op, offsets_arr[i]);
        }
        return exprs;
    }

    void set_tensor_expr(const flatbuffers::Vector<int32_t> *ids,
                         int32_t offset, ir::expr ex);
    void set_output_expr(const tflite::Operator &op, int32_t offset,
                         ir::expr ex);

    static datatype_t to_datatype(tflite::TensorType tflite_type);

    // ir::transpose *nhwc_to_nchw(datatype_t type, const ir::shape_t
    // &input_shape)
    //{
    //    return graph_.emplace<ir::transpose>(type, input_shape, ir::axis_t {
    //    0, 3, 1, 2 });
    //}

    // ir::transpose *nchw_to_nhwc(datatype_t type, const ir::shape_t
    // &input_shape)
    //{
    //    return graph_.emplace<ir::transpose>(type, input_shape, ir::axis_t {
    //    0, 2, 3, 1 });
    //}

    // ir::shape_t nhwc_to_nchw(const ir::shape_t &input_shape)
    //{
    //    return ir::get_transposed_shape(input_shape, { 0, 3, 1, 2 });
    //}

    // ir::shape_t nchw_to_nhwc(const ir::shape_t &input_shape)
    //{
    //    return ir::get_transposed_shape(input_shape, { 0, 2, 3, 1 });
    //}

    value_range<float>
    to_float_clamp_range(tflite::ActivationFunctionType func) {
        switch (func) {
        case tflite::ActivationFunctionType_NONE:
            return value_range<float>::full();
        case tflite::ActivationFunctionType_RELU:
            return {0.f, std::numeric_limits<float>::infinity()};
        case tflite::ActivationFunctionType_RELU_N1_TO_1:
            return {-1.f, 1.f};
        case tflite::ActivationFunctionType_RELU6:
            return {0.f, 6.f};
        default:
            throw std::runtime_error(
                std::string("Unsupported activation: ") +
                tflite::EnumNameActivationFunctionType(func));
        }
    }

    // void link_input_tensor(ir::input_connector *conn, int32_t tf_id)
    //{
    //    input_tensors_.emplace(conn, tf_id);
    //    auto tf_tensor = subgraph_->tensors()->Get(tf_id);
    //    if (to_data_type(tf_tensor->type()) != conn->type())
    //    {
    //        throw std::runtime_error(
    //            "Type must be same: \n"
    //            + conn->owner().name() + "[" +
    //            std::string(conn->owner().runtime_opcode().name) + "] != "
    //            + std::string(tf_tensor->name()->string_view()) + "[input]"
    //            + "\n has type mismatch: \n["
    //            + std::string(datatype_names(conn->type())) + "] != ["
    //            + std::string(datatype_names(to_data_type(tf_tensor->type())))
    //            + "]");
    //    }

    //    if (get_shape(tf_tensor->shape()) != conn->shape())
    //    {
    //        throw std::runtime_error(
    //            "Shape must be same: \n"
    //            + conn->owner().name() + "[" +
    //            std::string(conn->owner().runtime_opcode().name) + "] != "
    //            + std::string(tf_tensor->name()->string_view()) + "[output]"
    //            + "\n has shape mismatch: \n"
    //            + ir::to_string(conn->shape()) + " != "
    //            + ir::to_string(get_shape(tf_tensor->shape())) + "");
    //    }
    //}

    // void link_output_tensor(int32_t tf_id, ir::output_connector *conn)
    //{
    //    output_tensors_.emplace(tf_id, conn);
    //    auto tf_tensor = subgraph_->tensors()->Get(tf_id);
    //    if (to_data_type(tf_tensor->type()) != conn->type())
    //    {
    //        throw std::runtime_error(
    //            "Type must be same: \n"
    //            + conn->owner().name() + "[" +
    //            std::string(conn->owner().runtime_opcode().name) + "] != "
    //            + std::string(tf_tensor->name()->string_view()) + "[output]"
    //            + "\n has type mismatch: \n["
    //            + std::string(datatype_names(conn->type())) + "] != ["
    //            + std::string(datatype_names(to_data_type(tf_tensor->type())))
    //            + "]");
    //    }

    //    if (get_shape(tf_tensor->shape()) != conn->shape())
    //    {
    //        throw std::runtime_error(
    //            "Shape must be same: \n"
    //            + conn->owner().name() + "[" +
    //            std::string(conn->owner().runtime_opcode().name) + "] != "
    //            + std::string(tf_tensor->name()->string_view()) + "[output]"
    //            + "\n has shape mismatch: \n"
    //            + ir::to_string(conn->shape()) + " != "
    //            + ir::to_string(get_shape(tf_tensor->shape())) + "");
    //    }
    //}

    // void add_convert(ir::input_connector &next_input, const tflite::Tensor
    // &tensor, int32_t tf_id, datatype_t to_type);

    // void input_convert_to_type(ir::input_connector &next_input, const
    // tflite::Tensor &tensor, int32_t tf_id, datatype_t to_type);

    // void with_quantize(datatype_t ty, std::vector<ir::input_connector *>
    // &inputs, std::vector<quant_param_t> &input_dequant_params,
    //    std::vector<ir::output_connector *> &outputs,
    //    std::vector<quant_param_t> &output_quant_params)
    //{
    //    std::vector<ir::input_connector *> new_inputs;
    //    for (size_t i = 0; i < inputs.size(); i++)
    //    {
    //        auto deq = graph_.emplace<ir::dequantize>(ty, inputs[i]->shape(),
    //        dt_float32, input_dequant_params[i]);
    //        deq->name(inputs[i]->owner().name());
    //        inputs[i]->connect(deq->output());
    //        new_inputs.push_back(&deq->input());
    //    }
    //    inputs = new_inputs;

    //    std::vector<ir::output_connector *> new_outputs;
    //    for (size_t i = 0; i < outputs.size(); i++)
    //    {
    //        auto q = graph_.emplace<ir::quantize>(dt_float32,
    //        outputs[i]->shape(), ty, output_quant_params[i]);
    //        q->name(outputs[i]->owner().name());
    //        outputs[i]->connect(q->input());
    //        new_outputs.push_back(&q->output());
    //    }
    //    outputs = new_outputs;
    //}

  private:
    const tflite::Model *model_;
    const tflite::SubGraph *subgraph_;
    ir::module_t module_;
    std::unordered_map<int32_t, ir::expr> output_tensors_;
};
} // namespace nncase::importer

#define DEFINE_TFLITE_LOWER(opcode)                                            \
    void nncase::importer::tflite_importer::convert_op_##opcode(               \
        const tflite::Operator &op)
