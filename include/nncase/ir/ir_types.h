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
#include <nncase/runtime/datatypes.h>
#include <span>
#include <type_traits>
#include <xtensor/xshape.hpp>

namespace nncase::ir
{
using shape_t = xt::dynamic_shape<std::size_t>;
using axis_t = xt::dynamic_shape<int32_t>;

enum node_attributes
{
    node_attr_none = 0,
    node_attr_action = 1,
    node_attr_need_quantize = 2,
    node_attr_fuse_input_slice = 4,
    node_attr_fuse_output_concat = 8,
    node_attr_skip_constant_folding = 16,
    node_attr_skip_quantize = 32,
};

enum connector_attributes
{
    cnctr_attr_none = 0,
    cnctr_attr_need_quantize = 1,
    cnctr_attr_no_layout_strides = 2,
    cnctr_attr_no_buffer_fusion = 4,
    cnctr_attr_buffer_slice = 8,
    cnctr_attr_no_dummy_for_benchmark = 16
};

DEFINE_ENUM_BITMASK_OPERATORS(node_attributes)
DEFINE_ENUM_BITMASK_OPERATORS(connector_attributes)

template <class T, class = std::enable_if_t<std::is_pointer_v<T>>>
std::vector<std::decay_t<T>> dup(std::span<T> source)
{
    return { source.begin(), source.end() };
}
}
