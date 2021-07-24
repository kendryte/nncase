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
#include "../transform.h"
#include <unordered_set>

namespace nncase::ir::transforms
{
class NNCASE_API add_quant_checkpoints_transform : public transform
{
public:
    template <class TIt>
    add_quant_checkpoints_transform(TIt begin, TIt end)
        : opcodes_(begin, end)
    {
    }

    template <class... TArgs>
    add_quant_checkpoints_transform(std::in_place_t, TArgs &&...opcodes)
        : opcodes_({ std::forward<TArgs>(opcodes)... })
    {
    }

    void process(transform_context &context) override;

protected:
    bool skip_self_contained_check() const noexcept override { return true; }
    bool on_try_match(ir::node &node, transform_context &context) override;

private:
    std::unordered_set<ir::node_opcode> opcodes_;
};
}
