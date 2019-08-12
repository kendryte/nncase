/* Copyright 2019 Canaan Inc.
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
#include "../../node.h"
#include <runtime/k210/k210_sim_types.h>
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    namespace k210
    {
        class fake_piecewise_linear : public node
        {
        public:
            DEFINE_NODE_OPCODE(op_k210_fake_piecewise_linear);

            input_connector &input() { return input_at(0); }
            output_connector &output() { return output_at(0); }

            const xt::svector<runtime::k210::piecewise_linear_segment> &segments() const noexcept { return segments_; }

            fake_piecewise_linear(shape_t input_shape, xt::svector<runtime::k210::piecewise_linear_segment> segments);

        private:
            xt::svector<runtime::k210::piecewise_linear_segment> segments_;
        };
    }
}
}
