/* Copyright 2019-2020 Canaan Inc.
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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/pad.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

pad::pad(datatype_t type, shape_t input_shape, xt::svector<padding> paddings, pad_mode_t pad_mode, scalar pad_value)
    : paddings_(std::move(paddings)), pad_mode_(pad_mode), pad_value_(std::move(pad_value))
{
    add_input("input", type, input_shape);
    add_output("output", type, get_padded_shape(input_shape, paddings_));
}

bool pad::properties_equal(node &other) const
{
    auto &r = static_cast<pad &>(other);
    return paddings() == r.paddings() && pad_mode() == r.pad_mode() && pad_value() == r.pad_value();
}
