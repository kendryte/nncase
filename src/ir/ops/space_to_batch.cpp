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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/space_to_batch.h>

using namespace nncase;
using namespace nncase::ir;

space_to_batch::space_to_batch(datatype_t input_type, shape_t input_shape, int32_t block_h, int32_t block_w, padding padding_h, padding padding_w, scalar pad_value)
    : block_size_h_(block_h), block_size_w_(block_w), padding_h_(padding_h), padding_w_(padding_w), pad_value_(std::move(pad_value))
{
    add_input("input", input_type, input_shape);
    add_output("output", input_type,
        shape_t {
            input_shape[0] * block_h * block_w,
            input_shape[1],
            (input_shape[2] + padding_h_.sum()) / block_h,
            (input_shape[3] + padding_w_.sum()) / block_w });
}

bool space_to_batch::properties_equal(node &other) const
{
    auto &r = static_cast<space_to_batch &>(other);
    return block_size_h() == r.block_size_h() && block_size_w() == r.block_size_w()
        && padding_h() == r.padding_h() && padding_w() == r.padding_w()
        && pad_value() == r.pad_value();
}
