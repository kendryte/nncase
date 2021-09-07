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
#include <nncase/ir/connectors.h>
#include <nncase/ir/debug.h>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/transpose.h>
#include <schema_generated.h>
#include <span>
#include <unordered_map>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace nncase::importer
{
class tflite_importer
{
public:
    tflite_importer(std::span<const uint8_t> model, ir::graph &graph);

    void import(const import_options &options, std::string &real_inlayout, std::string &real_outlayout);

private:
    void convert_op(const tflite::Operator &op);

#define DEFINE_OPCODE(opcode) void convert_op_##opcode(const tflite::Operator &op);
#include "opcode.def"
#undef DEFINE_OPCODE

    void convert_pool2d(const tflite::Operator &op, reduce_op_t reduce_op, float init_value);
    void convert_binary(const tflite::Operator &op, binary_op_t binary_op, tflite::ActivationFunctionType activation);
    void convert_reduce(const tflite::Operator &op, reduce_op_t reduce_op, float init_value);
    void convert_unary(const tflite::Operator &op, unary_op_t unary_op);
    void convert_resize_image(const tflite::Operator &op, image_resize_mode_t mode);

    const tflite::Tensor &get_tensor(const flatbuffers::Vector<int32_t> *ids, int32_t offset)
    {
        return *subgraph_->tensors()->Get(ids->Get(offset));
    }

    template <class T>
    const tflite::Buffer &get_buffer(const tflite::Tensor &tensor)
    {
        auto expect_type = to_tensor_type<T>();
        auto actual_type = tensor.type();
        if (actual_type != expect_type)
            throw std::runtime_error(std::string("Tensor (") + tensor.name()->str() + std::string(") Expect ") + tflite::EnumNameTensorType(expect_type) + " tensor but got " + tflite::EnumNameTensorType(actual_type));

        auto buffer = model_->buffers()->Get(tensor.buffer());
        if (!buffer)
            throw std::runtime_error("Cannot read buffer");
        return *buffer;
    }

    ir::shape_t get_shape(const flatbuffers::Vector<int32_t> *shape)
    {
        if (shape && shape->size())
            return { std::begin(*shape), std::end(*shape) };
        else
            return { 1 };
    }

    template <class TCast, class T, size_t... I>
    auto get_vector_elements(const flatbuffers::Vector<T> &vector, std::index_sequence<I...>)
        -> std::array<TCast, sizeof...(I)>
    {
        return { (TCast)vector.Get(I)... };
    }

    template <size_t N, class Indices = std::make_index_sequence<N>>
    std::array<size_t, N> get_shape(const flatbuffers::Vector<int32_t> *shape)
    {
        if (shape && shape->size())
            return get_vector_elements<size_t>(*shape, Indices {});
        else
            return { 1 };
    }

    ir::axis_t get_axis(const flatbuffers::Vector<int32_t> &shape)
    {
        return { std::begin(shape), std::end(shape) };
    }

    template <class T>
    std::vector<T> to_vector(const flatbuffers::Vector<T> &shape)
    {
        return { std::begin(shape), std::end(shape) };
    }

    quant_param_t to_quant_param(const tflite::QuantizationParameters *param);

    template <class T>
    xt::xarray<T> load_array(const tflite::Tensor &tensor)
    {
        auto &buffer = get_buffer<T>(tensor);
        auto shape = get_shape(tensor.shape());
        return xt::adapt(reinterpret_cast<const T *>(buffer.data()->data()), shape);
    }

    template <class T, size_t N>
    xt::xtensor<T, N> load_tensor(const tflite::Tensor &tensor)
    {
        auto &buffer = get_buffer<T>(tensor);
        auto shape = get_shape(tensor.shape());
        return xt::adapt(reinterpret_cast<const T *>(buffer.data()->data()), shape);
    }

    scalar load_scalar(const tflite::Tensor &tensor)
    {
        auto buffer = model_->buffers()->Get(tensor.buffer());
        if (!buffer)
            throw std::runtime_error("Cannot read buffer");
        scalar s;
        s.type = to_data_type(tensor.type());
        std::memcpy(&s.storage, buffer->data()->data(), buffer->data()->size());
        return s;
    }

    template <size_t N>
    xt::xtensor<float, N> dequantize_tensor(const tflite::Tensor &tensor)
    {
        auto tensor_type = tensor.type();
        if (tensor_type == tflite::TensorType_FLOAT32)
        {
            return load_tensor<float, N>(tensor);
        }
        else if (tensor_type == tflite::TensorType_INT8)
        {
            auto src_tensor = load_tensor<int8_t, N>(tensor);
            auto &quant = *tensor.quantization();
            auto scale = quant.scale()->Get(0);
            auto bias = quant.zero_point()->Get(0);

            xt::xtensor<float, N> dest_tensor(get_shape<4>(tensor.shape()));
            auto src_it = src_tensor.begin();
            auto dest_it = dest_tensor.begin();
            while (src_it != src_tensor.end())
                *dest_it++ = (*src_it++ - bias) * scale;
            return dest_tensor;
        }
        else
        {
            throw std::runtime_error(std::string("Tensor (") + tensor.name()->str() + std::string(") of type ") + tflite::EnumNameTensorType(tensor_type) + " is not supported");
        }
    }

    template <class T>
    ir::axis_t load_axis(const tflite::Tensor &tensor)
    {
        auto &buffer = get_buffer<T>(tensor);
        auto begin = buffer.data()->data();
        auto end = begin + buffer.data()->size();
        return { reinterpret_cast<const T *>(begin), reinterpret_cast<const T *>(end) };
    }

    template <class T>
    ir::shape_t load_shape(const tflite::Tensor &tensor)
    {
        auto &buffer = get_buffer<T>(tensor);
        auto begin = buffer.data()->data();
        auto end = begin + buffer.data()->size();
        return { reinterpret_cast<const T *>(begin), reinterpret_cast<const T *>(end) };
    }

    template <class T, size_t N>
    xt::xtensor<T, N> load_weights(const tflite::Tensor &tensor)
    {
        auto data = load_tensor<T, N>(tensor);
        if constexpr (N == 4)
            data = xt::transpose(data, { 0, 3, 1, 2 });
        return data;
    }

    template <class T>
    constexpr tflite::TensorType to_tensor_type()
    {
        if constexpr (std::is_same_v<T, float>)
            return tflite::TensorType_FLOAT32;
        else if constexpr (std::is_same_v<T, float>)
            return tflite::TensorType_FLOAT32;
        else if constexpr (std::is_same_v<T, int32_t>)
            return tflite::TensorType_INT32;
        else if constexpr (std::is_same_v<T, int8_t>)
            return tflite::TensorType_INT8;
        else
        {
            assert(!"Invalid element type");
            std::terminate();
        }
    }

    constexpr datatype_t to_data_type(tflite::TensorType type)
    {
        switch (type)
        {
        case tflite::TensorType_FLOAT32:
            return dt_float32;
        case tflite::TensorType_INT8:
            return dt_int8;
        case tflite::TensorType_UINT8:
            return dt_uint8;
        case tflite::TensorType_INT32:
            return dt_int32;
        case tflite::TensorType_INT64:
            return dt_int64;
        default:
            throw std::runtime_error("Invalid tensor type");
        }
    }

    ir::transpose *nhwc_to_nchw(datatype_t type, const ir::shape_t &input_shape)
    {
        return graph_.emplace<ir::transpose>(type, input_shape, ir::axis_t { 0, 3, 1, 2 });
    }

    ir::transpose *nchw_to_nhwc(datatype_t type, const ir::shape_t &input_shape)
    {
        return graph_.emplace<ir::transpose>(type, input_shape, ir::axis_t { 0, 2, 3, 1 });
    }

    ir::shape_t nhwc_to_nchw(const ir::shape_t &input_shape)
    {
        return ir::get_transposed_shape(input_shape, { 0, 3, 1, 2 });
    }

    ir::shape_t nchw_to_nhwc(const ir::shape_t &input_shape)
    {
        return ir::get_transposed_shape(input_shape, { 0, 2, 3, 1 });
    }

    value_range<float> to_float_clamp_range(tflite::ActivationFunctionType func)
    {
        switch (func)
        {
        case tflite::ActivationFunctionType_NONE:
            return value_range<float>::full();
        case tflite::ActivationFunctionType_RELU:
            return { 0.f, std::numeric_limits<float>::infinity() };
        case tflite::ActivationFunctionType_RELU_N1_TO_1:
            return { -1.f, 1.f };
        case tflite::ActivationFunctionType_RELU6:
            return { 0.f, 6.f };
        default:
            throw std::runtime_error(std::string("Unsupported activation: ") + tflite::EnumNameActivationFunctionType(func));
        }
    }

    void link_input_tensor(ir::input_connector *conn, int32_t tf_id)
    {
        input_tensors_.emplace(conn, tf_id);
        auto tf_tensor = subgraph_->tensors()->Get(tf_id);
        if (to_data_type(tf_tensor->type()) != conn->type())
        {
            throw std::runtime_error(
                "Type must be same: \n"
                + conn->owner().name() + "[" + std::string(conn->owner().runtime_opcode().name) + "] != "
                + std::string(tf_tensor->name()->string_view()) + "[input]"
                + "\n has type mismatch: \n["
                + std::string(datatype_names(conn->type())) + "] != ["
                + std::string(datatype_names(to_data_type(tf_tensor->type()))) + "]");
        }

        if (get_shape(tf_tensor->shape()) != conn->shape())
        {
            throw std::runtime_error(
                "Shape must be same: \n"
                + conn->owner().name() + "[" + std::string(conn->owner().runtime_opcode().name) + "] != "
                + std::string(tf_tensor->name()->string_view()) + "[output]"
                + "\n has shape mismatch: \n"
                + ir::to_string(conn->shape()) + " != "
                + ir::to_string(get_shape(tf_tensor->shape())) + "");
        }
    }

    void link_output_tensor(int32_t tf_id, ir::output_connector *conn)
    {
        output_tensors_.emplace(tf_id, conn);
        auto tf_tensor = subgraph_->tensors()->Get(tf_id);
        if (to_data_type(tf_tensor->type()) != conn->type())
        {
            throw std::runtime_error(
                "Type must be same: \n"
                + conn->owner().name() + "[" + std::string(conn->owner().runtime_opcode().name) + "] != "
                + std::string(tf_tensor->name()->string_view()) + "[output]"
                + "\n has type mismatch: \n["
                + std::string(datatype_names(conn->type())) + "] != ["
                + std::string(datatype_names(to_data_type(tf_tensor->type()))) + "]");
        }

        if (get_shape(tf_tensor->shape()) != conn->shape())
        {
            throw std::runtime_error(
                "Shape must be same: \n"
                + conn->owner().name() + "[" + std::string(conn->owner().runtime_opcode().name) + "] != "
                + std::string(tf_tensor->name()->string_view()) + "[output]"
                + "\n has shape mismatch: \n"
                + ir::to_string(conn->shape()) + " != "
                + ir::to_string(get_shape(tf_tensor->shape())) + "");
        }
    }

    void add_convert(ir::input_connector &next_input, const tflite::Tensor &tensor, int32_t tf_id, datatype_t to_type);

    void input_convert_to_type(ir::input_connector &next_input, const tflite::Tensor &tensor, int32_t tf_id, datatype_t to_type);

    void with_quantize(datatype_t ty, std::vector<ir::input_connector *> &inputs, std::vector<quant_param_t> &input_dequant_params,
        std::vector<ir::output_connector *> &outputs, std::vector<quant_param_t> &output_quant_params)
    {
        std::vector<ir::input_connector *> new_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            auto deq = graph_.emplace<ir::dequantize>(ty, inputs[i]->shape(), dt_float32, input_dequant_params[i]);
            deq->name(inputs[i]->owner().name());
            inputs[i]->connect(deq->output());
            new_inputs.push_back(&deq->input());
        }
        inputs = new_inputs;

        std::vector<ir::output_connector *> new_outputs;
        for (size_t i = 0; i < outputs.size(); i++)
        {
            auto q = graph_.emplace<ir::quantize>(dt_float32, outputs[i]->shape(), ty, output_quant_params[i]);
            q->name(outputs[i]->owner().name());
            outputs[i]->connect(q->input());
            new_outputs.push_back(&q->output());
        }
        outputs = new_outputs;
    }

private:
    const tflite::Model *model_;
    const tflite::SubGraph *subgraph_;
    ir::graph &graph_;
    std::unordered_map<ir::input_connector *, int32_t> input_tensors_;
    std::unordered_map<int32_t, ir::output_connector *> output_tensors_;
};
}

#define DEFINE_TFLITE_LOWER(opcode) \
    void nncase::importer::tflite_importer::convert_op_##opcode(const tflite::Operator &op)
