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
#include <hlir/op_utils.h>
#include <llir/ops/memory_copy.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::llir;

memory_copy::memory_copy(datatype_t input_type, shape_t input_shape, datatype_t output_type, shape_t new_shape)
{
    add_input("input", input_type, input_shape);
    add_output("output", output_type, new_shape);
}
