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
class NNCASE_API scale : public node
{
public:
    DEFINE_NODE_OPCODE(op_scale);

    input_connector &input() { return input_at(0); }
    output_connector &output() { return output_at(0); }

    std::vector<float> gamma() const noexcept { return gamma_; }
    std::vector<float> beta() const noexcept { return beta_; }

    scale(datatype_t input_type, shape_t input_shape, std::vector<float> gamma, std::vector<float> beta);

protected:
    bool properties_equal(node &other) const override;

private:
    std::vector<float> gamma_;
    std::vector<float> beta_;
};
}
