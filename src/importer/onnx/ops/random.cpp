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

#include "../onnx_importer.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/random_normal.h>
#include <nncase/ir/ops/random_uniform.h>
#include <nncase/runtime/debug.h>
#include <time.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_RandomNormal(const onnx::NodeProto &node)
{
    assert(node.input().size() == 0);
    assert(node.output().size() == 1);

    // dtype
    auto dtype_attr = get_attribute<int>(node, "dtype");
    TensorProto_DataType dtype = static_cast<TensorProto_DataType>(dtype_attr ? dtype_attr.value() : TensorProto_DataType_FLOAT);
    auto output_type = get_datatype(dtype).value();
    if (output_type != dt_float32)
    {
        throw std::runtime_error("RandomNormal supports float only, but got " + std::string(datatype_names(output_type)));
    }

    // mean
    auto mean_attr = get_attribute<float>(node, "mean");
    float mean = mean_attr ? mean_attr.value() : 0.0f;

    // scale
    auto scale_attr = get_attribute<float>(node, "scale");
    float scale = scale_attr ? scale_attr.value() : 1.0f;

    // seed
    auto seed_attr = get_attribute<float>(node, "seed");
    float seed = seed_attr ? seed_attr.value() : time(nullptr);

    // shape
    auto shape_attr = get_attribute<std::vector<int>>(node, "shape");
    assert(shape_attr);
    shape_t output_shape(shape_attr.value().begin(), shape_attr.value().end());

    // random_normal
    auto op = graph_.emplace<random_normal>(output_type, output_shape, mean, scale, seed);
    op->name(generate_name(node));

    output_tensors_.emplace(node.output()[0], &op->output());
}

void onnx_importer::convert_op_RandomNormalLike(const onnx::NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];

    // dtype
    auto dtype_attr = get_attribute<int>(node, "dtype");
    datatype_t output_type = dtype_attr ? get_datatype(static_cast<TensorProto_DataType>(dtype_attr.value())).value()
                                        : get_datatype(input).value();
    if (output_type != dt_float32)
    {
        throw std::runtime_error("RandomNormalLike supports float only, but got " + std::string(datatype_names(output_type)));
    }

    // mean
    auto mean_attr = get_attribute<float>(node, "mean");
    float mean = mean_attr ? mean_attr.value() : 0.0f;

    // scale
    auto scale_attr = get_attribute<float>(node, "scale");
    float scale = scale_attr ? scale_attr.value() : 1.0f;

    // seed
    auto seed_attr = get_attribute<float>(node, "seed");
    float seed = seed_attr ? seed_attr.value() : time(nullptr);

    // random_normal
    auto op = graph_.emplace<random_normal>(output_type, get_shape(input), mean, scale, seed);
    op->name(generate_name(node));

    output_tensors_.emplace(node.output()[0], &op->output());
}

void onnx_importer::convert_op_RandomUniform(const onnx::NodeProto &node)
{
    assert(node.input().size() == 0);
    assert(node.output().size() == 1);

    // dtype
    auto dtype_attr = get_attribute<int>(node, "dtype");
    TensorProto_DataType dtype = static_cast<TensorProto_DataType>(dtype_attr ? dtype_attr.value() : TensorProto_DataType_FLOAT);
    auto output_type = get_datatype(dtype).value();
    if (output_type != dt_float32)
    {
        throw std::runtime_error("RandomUniform supports float only, but got " + std::string(datatype_names(output_type)));
    }

    // low
    auto low_attr = get_attribute<float>(node, "low");
    float low = low_attr ? low_attr.value() : 0.0f;

    // high
    auto high_attr = get_attribute<float>(node, "high");
    float high = high_attr ? high_attr.value() : 1.0f;

    // seed
    auto seed_attr = get_attribute<float>(node, "seed");
    float seed = seed_attr ? seed_attr.value() : time(nullptr);

    // shape
    auto shape_attr = get_attribute<std::vector<int>>(node, "shape");
    assert(shape_attr);
    shape_t output_shape(shape_attr.value().begin(), shape_attr.value().end());

    // random_uniform
    auto op = graph_.emplace<random_uniform>(output_type, output_shape, low, high, seed);
    op->name(generate_name(node));

    output_tensors_.emplace(node.output()[0], &op->output());
}

void onnx_importer::convert_op_RandomUniformLike(const onnx::NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];

    // dtype
    auto dtype_attr = get_attribute<int>(node, "dtype");
    datatype_t output_type = dtype_attr ? get_datatype(static_cast<TensorProto_DataType>(dtype_attr.value())).value()
                                        : get_datatype(input).value();
    if (output_type != dt_float32)
    {
        throw std::runtime_error("RandomUniformLike supports float only, but got " + std::string(datatype_names(output_type)));
    }

    // low
    auto low_attr = get_attribute<float>(node, "low");
    float low = low_attr ? low_attr.value() : 0.0f;

    // high
    auto high_attr = get_attribute<float>(node, "high");
    float high = high_attr ? high_attr.value() : 1.0f;

    // seed
    auto seed_attr = get_attribute<float>(node, "seed");
    float seed = seed_attr ? seed_attr.value() : time(nullptr);

    // random_uniform
    auto op = graph_.emplace<random_uniform>(output_type, get_shape(input), low, high, seed);
    op->name(generate_name(node));

    output_tensors_.emplace(node.output()[0], &op->output());
}

