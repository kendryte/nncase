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
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase::ir
{
class NNCASE_API gru : public node
{
public:
    DEFINE_NODE_OPCODE(op_gru);

    input_connector &input() { return input_at(0); }
    input_connector &w() { return input_at(1); }
    input_connector &r() { return input_at(2); }
    input_connector &b() { return input_at(3); }
    input_connector &initial_h() { return input_at(4); }
    input_connector &initial_c() { return input_at(5); }
    output_connector &output() { return output_at(0); }
    output_connector &output_h() { return output_at(1); }

    lstm_direction direction() const noexcept { return direction_; }
    std::string framework() const noexcept { return framework_; }
    bool linear_before_reset() const noexcept { return linear_before_reset_; }

    gru(shape_t input_shape, shape_t w_shape, shape_t r_shape, shape_t b_shape, shape_t output_shape,
        shape_t output_h_shape, lstm_direction direction, std::string framework, bool linear_before_reset);

protected:
    bool properties_equal(node &other) const override;

private:
    lstm_direction direction_;
    std::string framework_;
    bool linear_before_reset_;
};
}
