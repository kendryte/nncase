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
#include <nncase/ir/ops/constant.h>

namespace nncase::ir::transforms
{

class NNCASE_API remove_nonsense_binary : public transform
{
public:
    void process(transform_context &context) override;

    template <datatype_t DT>
    bool constant_equal_to(std::span<const std::byte> data, float value) noexcept
    {
        using T = to_cpp_type_t<DT>;
        const T value_ref = static_cast<T>(value);
        std::span<const T> data_ref = std::span(reinterpret_cast<const T *>(data.data()), data.size_bytes() / sizeof(T));

        std::vector<bool> eq_seq;
        bool ret = true;
        for (auto &&x : data_ref)
        {
            ret &= (x == value_ref);
            if (not ret)
                break;
        }
        return ret;
    }

    bool constant_equal_to(constant *node, float value) noexcept;

protected:
    bool skip_self_contained_check() const noexcept override { return true; }
    bool on_try_match(ir::node &node, transform_context &context) override;
};

}