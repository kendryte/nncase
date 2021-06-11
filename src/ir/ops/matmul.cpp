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
#include <nncase/ir/ops/matmul.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

matmul::matmul(shape_t input_a_shape, shape_t input_b_shape, value_range<float> fused_activation)
    : fused_activation_(fused_activation)
{
    std::cout<<"test3"<<std::endl;
    for (size_t i =0;i<input_a_shape.size();i++)
        std::cout<<input_a_shape[i]<<std::endl;
    std::cout<<"test4"<<std::endl;
    for (size_t i =0;i<input_b_shape.size();i++)
        std::cout<<input_b_shape[i]<<std::endl;

    // workaround currently since we don't know lstm output shape now.
    // if (input_a_shape.size() != 2 || input_b_shape.size() != 2)
    //     throw std::invalid_argument("inputs must be 2 rank");
    // if (input_a_shape[1] != input_b_shape[0])
    //     throw std::invalid_argument("input a's cols must be equal to input b's rows");
    add_input("input_a", dt_float32, input_a_shape);
    add_input("input_b", dt_float32, input_b_shape);
    add_input("bias", dt_float32, shape_t { input_b_shape[1] });
    add_output("output", dt_float32, shape_t { input_a_shape[0], input_b_shape[1] });
}

bool matmul::properties_equal(node &other) const
{
    auto &r = static_cast<matmul &>(other);
    return fused_activation() == r.fused_activation();
}
