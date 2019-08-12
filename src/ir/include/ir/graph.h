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
#include "node.h"
#include "placeholders.h"
#include <memory>
#include <vector>

namespace nncase
{
namespace ir
{
    class graph
    {
    public:
        xtl::span<std::unique_ptr<node>> nodes() noexcept { return nodes_; }
        xtl::span<input_node *> inputs() noexcept { return inputs_; }
        xtl::span<output_node *> outputs() noexcept { return outputs_; }

        template <class T, class... TArgs>
        T *emplace(TArgs &&... args)
        {
            auto node = static_cast<T *>(nodes_.emplace_back(new T(std::forward<TArgs>(args)...)).get());
            if constexpr (std::is_same_v<T, input_node>)
                inputs_.emplace_back(node);
            else if constexpr (std::is_same_v<T, output_node>)
                outputs_.emplace_back(node);
            return node;
        }

        void assign_names();
        void collect();

    private:
        std::vector<std::unique_ptr<node>> nodes_;
        std::vector<input_node *> inputs_;
        std::vector<output_node *> outputs_;
    };
}
}
