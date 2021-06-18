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
    input_connector &w_xc() { return input_at(1); }
    input_connector &b_xc() { return input_at(2); }
    input_connector &w_hc() { return input_at(3); }
    input_connector &w_static() { return input_at(4); }
    output_connector &output() { return output_at(0); }

    int32_t num_output() const noexcept { return num_output_; }
    bool has_static() const noexcept { return has_static_; }

    lstm(shape_t input_shape, shape_t w_xc_shape, shape_t b_xc_shape, shape_t w_hc_shape, int32_t num_output, bool has_static);

protected:
    bool properties_equal(node &other) const override;

private:
    int32_t num_output_;
    bool has_static_;
};
}
