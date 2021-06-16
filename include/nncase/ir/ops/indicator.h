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
// #pragma once
// #include "../node.h"
// #include <xtensor/xtensor.hpp>

// namespace nncase::ir
// {
// class NNCASE_API indicator : public node
// {
// public:
//     DEFINE_NODE_OPCODE(op_indicator);

//     output_connector &output() { return output_at(0); }

//     int32_t time_step() const noexcept { return time_step_; }
//     int32_t batch_size() const noexcept { return batch_size_; }

//     template <class TShape>
//     indicator(datatype_t type, TShape &&shape, int32_t time_step, int32_t batch_size)
//         : time_step_(time_step), batch_size_(batch_size)
//     {
//         add_output("output", type, std::forward<TShape>(shape), mem_input);
//     }

// protected:
//     bool properties_equal(node &other) const override;

// private:
//     int32_t time_step_;
//     int32_t batch_size_;
// };
// }
