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

namespace nncase
{
namespace llir
{
    class nnil_unary_method : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_nnil_unary_method);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const std::vector<uint8_t> &body() const noexcept { return body_; };

        nnil_unary_method(shape_t input_shape, std::vector<uint8_t> body);

    private:
        std::vector<uint8_t> body_;
    };
}
}
