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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/clamp.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

clamp::clamp(shape_t input_shape, shape_t input_low_shape, shape_t input_high_shape)
{
    add_input("input", dt_float32, input_shape);
    add_input("input_low", dt_float32, input_low_shape);
    add_input("input_high", dt_float32, input_high_shape);
    add_output("output", dt_float32, input_shape);
}
