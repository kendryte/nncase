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
#include "caffe.pb.h"
#include <nncase/ir/connectors.h>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/transpose.h>
#include <unordered_map>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace importer
{
    class caffe_importer
    {
    public:
        caffe_importer(std::span<const uint8_t> model, std::span<const uint8_t> prototxt, ir::graph &graph);

        void import(const struct import_options &options);

    private:
        void convert_op(const caffe::LayerParameter &op, caffe::NetParameter caffemodel);

#define DEFINE_OPCODE(opcode) void convert_op_##opcode(const caffe::LayerParameter &op, caffe::NetParameter caffemodel);
#include "opcode.def"
#undef DEFINE_OPCODE

        void load_tensor(std::string_view name, uint8_t *begin, uint8_t *end);

    private:
        static ir::shape_t get_shape(const caffe::BlobShape &shape)
        {
            ir::shape_t result;
            result.reserve(shape.dim_size());
            for (int i = 0; i < shape.dim_size(); i++)
                result.push_back((size_t)shape.dim(i));
            return result;
        }

        static ir::axis_t get_axis(const caffe::BlobShape &shape)
        {
            ir::axis_t result;
            result.reserve(shape.dim_size());
            for (int i = 0; i < shape.dim_size(); i++)
                result.push_back((int32_t)shape.dim(i));
            return result;
        }

        template <class TGetter>
        static uint32_t get_or_default(TGetter &&getter, int32_t size, int32_t index, uint32_t default_val)
        {
            if (size == 0)
                return default_val;
            if (size == 1)
                return getter(0);
            if (index < size)
                return getter(index);
            return default_val;
        }

        caffe::LayerParameter get_op_data(caffe::LayerParameter op, caffe::NetParameter caffemodel)
        {
            for (int32_t i = 0; i < caffemodel.layer_size(); i++)
            {
                if (op.name() == caffemodel.layer(i).name())
                    return caffemodel.layer(i);
            }
            throw std::runtime_error("can't find related data from caffemodel");
        }

        // input names are different when there are bn/scale/relu nodes above
        std::vector<std::string> get_real_input_names(caffe::LayerParameter op)
        {
            std::vector<std::string> input_names;
            for (int32_t i = 0; i < op.bottom_size(); i++)
            {
                std::string input_name_i = op.bottom(i) + "/clamp";
                if (output_tensors_.find(input_name_i) == output_tensors_.end())
                {
                    input_name_i = op.bottom(i) + "/add";
                    if (output_tensors_.find(input_name_i) == output_tensors_.end())
                    {
                        input_name_i = op.bottom(i) + "/mul";
                        if (output_tensors_.find(input_name_i) == output_tensors_.end())
                            input_name_i = op.bottom(i);
                    }
                }
                input_names.push_back(input_name_i);
            }
            return input_names;
        }

        template <size_t N>
        xt::xtensor<float, N> load_tensor(const caffe::BlobProto &blob)
        {
            auto &buffer = blob.data();
            auto shape = get_shape(blob.shape());
            return xt::adapt(reinterpret_cast<const float *>(buffer.data()), shape);
        }

    private:
        caffe::NetParameter model_;
        caffe::NetParameter prototxt_;
        ir::graph &graph_;
        std::unordered_map<ir::input_connector *, std::string> input_tensors_;
        std::unordered_map<std::string_view, ir::output_connector *> output_tensors_;
    };
}
}

#define DEFINE_CAFFE_LOWER(opcode) \
    void nncase::importer::caffe_importer::convert_op_##opcode([[maybe_unused]]const caffe::LayerParameter &op, [[maybe_unused]]caffe::NetParameter caffemodel)
