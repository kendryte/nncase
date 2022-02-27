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
class NNCASE_API lstm : public node
{
public:
    DEFINE_NODE_OPCODE(op_lstm);

    input_connector &input() { return input_at(0); }
    input_connector &w() { return input_at(1); }
    input_connector &r() { return input_at(2); }
    input_connector &b() { return input_at(3); }
    input_connector &initial_h() { return input_at(4); }
    input_connector &initial_c() { return input_at(5); }
    input_connector &w_static() { return input_at(6); }
    output_connector &output() { return output_at(0); }
    output_connector &output_h() { return output_at(1); }
    output_connector &output_c() { return output_at(2); }

    bool has_static() const noexcept { return has_static_; }
    lstm_direction direction() const noexcept { return direction_; }
    std::string framework() const noexcept { return framework_; }

    lstm(shape_t input_shape, shape_t w_shape, shape_t r_shape, shape_t b_shape, shape_t output_shape,
        shape_t output_h_shape, shape_t output_c_shape, bool has_static, lstm_direction direction, std::string framework);

protected:
    bool properties_equal(node &other) const override;

private:
    bool has_static_;
    lstm_direction direction_;
    std::string framework_;
};
}
