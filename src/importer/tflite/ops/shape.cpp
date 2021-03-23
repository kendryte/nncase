/* Copyright 2020 Canaan Inc.
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
#include "../tflite_importer.h"
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

namespace
{
template <class T>
std::vector<std::byte> as_shape_output(const flatbuffers::Vector<int32_t> &shape)
{
    std::vector<std::byte> result(shape.size() * sizeof(T));
    for (size_t i = 0; i < shape.size(); i++)
    {
        auto value = static_cast<T>(shape[i]);
        std::memcpy(result.data() + i * sizeof(T), &value, sizeof(T));
    }

    return result;
}
}

#define AS_SHAPE_IMPL(type)                                            \
    case type:                                                         \
        output = as_shape_output<to_cpp_type_t<type>>(*input.shape()); \
        break

DEFINE_TFLITE_LOWER(SHAPE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_ShapeOptions();
    std::vector<std::byte> output;
    switch (to_data_type(options.out_type()))
    {
        AS_SHAPE_IMPL(dt_int32);
        AS_SHAPE_IMPL(dt_float32);
    default:
        throw std::runtime_error(std::string("Unsupported SHAPE's out_type: ") + tflite::EnumNameTensorType(options.out_type()));
    }

    auto node = graph_.emplace<constant>(to_data_type(options.out_type()), shape_t { input.shape()->size() }, output);
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_output_tensor(op.outputs()->Get(0), &node->output());
}
