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
#include "../op.h"
#include "nncase/runtime/datatypes.h"
#include "opcode.h"

namespace nncase::ir::tensors
{
/** @brief GatherND operator node */
class NNCASE_API gather_nd_node : public op_node
{
public:
    DEFINE_NODE_OPCODE(op_tensors_gather_nd);

    gather_nd_node(int32_t axis, int32_t batch_dims);

    /** @brief Get the axis of the gather_nd expression */
    int32_t axis() const noexcept { return axis_; }
    /** @brief Set the axis of the gather_nd expression */
    void axis(int32_t value) noexcept { axis_ = value; }

    /** @brief Get the batch dims of the gather_nd expression */
    int32_t batch_dims() const noexcept { return batch_dims_; }
    /** @brief Set the batch dims of the gather_nd expression */
    void batch_dims(int32_t value) noexcept { batch_dims_ = value; }

private:
    int32_t axis_;
    int32_t batch_dims_;
};

using gather_nd = expr_t<gather_nd_node>;
}
