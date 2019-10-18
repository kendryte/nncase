/* Copyright 2019 Canaan Inc.
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
#include "tflite_schema.h"
#include <ir/connectors.h>
#include <ir/graph.h>
#include <ir/op_utils.h>
#include <ir/ops/transpose.h>
#include <unordered_map>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace importer
{
    class tflite_importer
    {
    public:
        tflite_importer(xtl::span<const uint8_t> model, ir::graph &graph);

        void import();

    private:
        void convert_op(const tflite::Operator &op);

#define DEFINE_OPCODE(opcode) void convert_op_##opcode(const tflite::Operator &op);
#include "opcode.def"
#undef DEFINE_OPCODE

        void convert_pool_2d(const tflite::Operator &op, reduce_op_t reduce_op, float init_value);
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

        ir::axis_t get_axis(const flatbuffers::Vector<int32_t> &shape)
        {
            return { std::begin(shape), std::end(shape) };
        }

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

        template <class T>
        ir::axis_t load_axis(const tflite::Tensor &tensor)
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
            else
                assert(!"Invalid element type");
        }

        constexpr datatype_t to_data_type(tflite::TensorType type)
        {
            switch (type)
            {
            case tflite::TensorType_FLOAT32:
                return dt_float32;
            default:
                throw std::runtime_error("Invalid tesnor type");
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
                return { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max() };
            case tflite::ActivationFunctionType_RELU:
                return { 0.f, std::numeric_limits<float>::max() };
            case tflite::ActivationFunctionType_RELU_N1_TO_1:
                return { -1.f, 1.f };
            case tflite::ActivationFunctionType_RELU6:
                return { 0.f, 6.f };
            default:
                throw std::runtime_error(std::string("Not supported activation: ") + tflite::EnumNameActivationFunctionType(func));
            }
        }

    private:
        const tflite::Model *model_;
        const tflite::SubGraph *subgraph_;
        ir::graph &graph_;
        std::unordered_map<ir::input_connector *, int32_t> input_tensors_;
        std::unordered_map<int32_t, ir::output_connector *> output_tensors_;
    };
}
}

#define DEFINE_TFLITE_LOWER(opcode) \
    void nncase::importer::tflite_importer::convert_op_##opcode(const tflite::Operator &op)
