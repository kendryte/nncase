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
#include <cassert>
#include <ir/graph.h>
#include <scheduler/scheduler.h>
#include <unordered_map>

namespace nncase
{
namespace ir
{
    class evaluate_context
    {
    public:
        evaluate_context(const std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, const std::unordered_map<ir::output_connector *, scheduler::memory_allocation> &allocations);

        xtl::span<uint8_t> memory_at(const scheduler::memory_allocation &allocation);

        template <class T>
        xtl::span<T> memory_at(const scheduler::memory_allocation &allocation)
        {
            auto span = memory_at(allocation);
            return { reinterpret_cast<T *>(span.data()), span.size_bytes() / sizeof(T) };
        }

        template <class T>
        xtl::span<T> memory_at(ir::output_connector &connector)
        {
            auto &alloc = allocations_.at(&connector);
            return memory_at<T>(alloc);
        }

        template <class T>
        xtl::span<T> memory_at(ir::input_connector &connector)
        {
            auto conn = connector.connection();
            assert(conn);
            return memory_at<T>(*conn);
        }

    private:
        const std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators_;
        const std::unordered_map<ir::output_connector *, scheduler::memory_allocation> &allocations_;
        std::unordered_map<memory_type_t, std::unique_ptr<uint8_t[]>> memory_pools_;
    };

    class quantizer;

    class evaluator
    {
    public:
        evaluator(evaluate_context &context, xtl::span<ir::node *> compute_sequence);

        template <class T>
        xtl::span<T> input_at(size_t index)
        {
            return context_.memory_at<T>(*inputs_[index]);
        }

        template <class T>
        xtl::span<T> output_at(size_t index)
        {
            return context_.memory_at<T>(*outputs_[index]);
        }

        void evaluate(quantizer *quantizer = nullptr, bool add_input_stat = false);

    private:
        evaluate_context &context_;
        xtl::span<ir::node *> compute_sequence_;
        std::vector<ir::output_connector *> inputs_;
        std::vector<ir::input_connector *> outputs_;
    };

    void register_evaluator(ir::node_opcode opcode, std::function<void(ir::node &, evaluate_context &)> evaluator);
}
}
