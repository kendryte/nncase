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
#pragma once
#include "../node.h"
#include <xtensor/xarray.hpp>

namespace nncase::ir
{
class NNCASE_API space_to_batch : public node
{
public:
    DEFINE_NODE_OPCODE(op_space_to_batch);

    input_connector &input() { return input_at(0); }
    output_connector &output() { return output_at(0); }

    int32_t block_size_h() const noexcept { return block_size_h_; }
    int32_t block_size_w() const noexcept { return block_size_w_; }
    padding padding_h() const noexcept { return padding_h_; }
    padding padding_w() const noexcept { return padding_w_; }
    const scalar &pad_value() const noexcept { return pad_value_; }

    space_to_batch(datatype_t input_type, shape_t input_shape, int32_t block_shape_h, int32_t block_shape_w, padding padding_h, padding padding_w, scalar pad_value);

protected:
    bool properties_equal(node &other) const override;

private:
    int32_t block_size_h_;
    int32_t block_size_w_;
    padding padding_h_;
    padding padding_w_;
    scalar pad_value_;
};
}
